"""Collect activations for multiple layers in a single forward pass.

Saves per-layer activation tensors and shared metadata to disk so that SAE
training can be run without re-running the forward pass.

Uses numpy memory-mapped files to write activations directly to disk as they
are computed, avoiding the need to hold all activations in RAM.  Peak Python
RSS ≈ model weights + one batch of activations, regardless of dataset size.

Output layout:
    {output_dir}/
        layer_{N}/
            activations.pt   # float16 tensor [n_rows, d_model]
            metadata.json
        collection_info.json

Usage:
    python sae_experiments/scripts/collect_activations.py \
        --config configs/clevr_lite/sae_layer0_attn_out_question.yaml \
        --layers 0,10,11,12,13,14 \
        --output_dir "/run/media/ron/External SSD/clevr_lite_activations/question" \
        --batch_size 8 \
        --num_workers 4 \
        --show_progress true
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    IMAGE_TOKEN_INDEX = -200

from sae_experiments.config.sae_config import load_config
from sae_experiments.utils.hook_utils import HookManager, get_target_module
from sae_experiments.utils.knockout_utils import estimate_image_token_count
from sae_experiments.utils import token_utils
from sae_experiments.utils.script_utils import load_llava_components


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def _make_batched_collate(pad_id: int):
    """Right-padding collate for batches of variable-length LLaVA inputs.

    Each item from CustomDataset is (input_ids [1,seq], image_tensor list,
    image_sizes list, prompt str, mask_tensor).  We pad input_ids on the
    right so that question-token positions are unaffected by causal masking,
    and return the original per-item input_ids for position computation.
    """
    def collate(batch):
        input_ids_list, image_tensors_list, image_sizes_list, prompts, _ = zip(*batch)

        max_len = max(ids.shape[-1] for ids in input_ids_list)
        B = len(input_ids_list)
        padded = torch.full((B, max_len), pad_id, dtype=input_ids_list[0].dtype)
        orig_input_ids = []
        for i, ids in enumerate(input_ids_list):
            l = ids.shape[-1]
            padded[i, :l] = ids[0]
            orig_input_ids.append(ids[0])

        stacked_images = [img_list[0] for img_list in image_tensors_list]
        sizes = [sz[0] for sz in image_sizes_list]

        return padded, stacked_images, sizes, prompts, orig_input_ids

    return collate


def _normalize_acts_batch(acts, item_idx: int):
    """Extract activations for item *item_idx* from a batch hook output."""
    if acts is None:
        return None
    if isinstance(acts, (tuple, list)):
        acts = acts[0]
    if acts is None:
        return None
    while acts.ndim > 3 and acts.shape[0] == 1:
        acts = acts[0]
    if acts.ndim == 3:
        return acts[item_idx]   # [seq, d_model]
    if acts.ndim == 2:
        return acts             # already [seq, d_model] (batch_size=1 path)
    return None


def _select_positions(
    position_type: str,
    input_ids: torch.Tensor,
    image_token_count: int,
    question_text: str,
    tokenizer,
    line: dict,
) -> List[int]:
    if position_type == "all":
        return list(range(input_ids.shape[-1] - 1 + image_token_count))

    question_range = token_utils.get_question_token_range(
        input_ids[0], image_token_count, question_text, tokenizer, IMAGE_TOKEN_INDEX
    )

    if position_type == "question":
        return question_range
    if position_type == "last":
        ntoks = input_ids.shape[-1] + image_token_count - 1
        return [max(0, ntoks - 1)]
    if position_type == "attribute":
        if not question_range:
            return []
        start, end = question_range[0], question_range[-1]
        positions = []
        for attr in line.get("attribute_tokens", []):
            positions.extend(start + p for p in attr.get("positions", []))
        return sorted(set(p for p in positions if start <= p <= end))
    if position_type == "image":
        ids_list = input_ids[0].tolist()
        try:
            img_idx = ids_list.index(IMAGE_TOKEN_INDEX)
        except ValueError:
            return []
        return list(range(img_idx, img_idx + image_token_count))

    return question_range


def _batched_forward(model, orig_input_ids, stacked_images, device):
    """Manual batched forward pass for LLaVA.

    LLaVA's prepare_inputs_labels_for_multimodal only processes the first item
    in a batch (returns inputs_embeds [1, seq, d_model] regardless of B).
    This function:
      1. Runs the vision encoder + projector once on all B images.
      2. Builds per-item inputs_embeds by splicing image features into token embeddings.
      3. Right-pads and stacks into a [B, max_seq, d_model] batch.
      4. Runs the language model backbone — hooks capture the full batch.

    Returns image_token_count (int) so callers can resolve position indices.
    """
    B = len(orig_input_ids)

    stacked = torch.stack(stacked_images, dim=0)
    image_features = model.encode_images(stacked)
    image_token_count = image_features.shape[1]

    inputs_embeds_list = []
    for i in range(B):
        item_ids = orig_input_ids[i].to(device)
        safe_ids = item_ids.clone()
        safe_ids[safe_ids < 0] = 0
        text_emb = model.get_model().embed_tokens(safe_ids)

        img_pos = (item_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if len(img_pos) > 0:
            pos = img_pos[0].item()
            expanded = torch.cat(
                [text_emb[:pos], image_features[i], text_emb[pos + 1:]], dim=0
            )
        else:
            expanded = text_emb
        inputs_embeds_list.append(expanded)

    max_len = max(e.shape[0] for e in inputs_embeds_list)
    d = inputs_embeds_list[0].shape[-1]
    dtype = inputs_embeds_list[0].dtype
    batched = torch.zeros(B, max_len, d, dtype=dtype, device=device)
    for i, emb in enumerate(inputs_embeds_list):
        batched[i, : emb.shape[0]] = emb

    model.model(inputs_embeds=batched, use_cache=False)

    return image_token_count


# ---------------------------------------------------------------------------
# Memory-mapped activation sink
# ---------------------------------------------------------------------------

class MemmapActivationSink:
    """Streams activation rows directly to disk via numpy memory-mapped files.

    Manages one memmap per layer.  Starts with a generous capacity estimate
    and truncates to the actual row count at the end.  Peak RAM cost is
    negligible — the OS pages data out as it's written.

    Parameters
    ----------
    output_dir : str
        Root directory; each layer gets a subdirectory.
    layers : list[int]
        Layer indices to allocate files for.
    d_model : int
        Hidden dimension of the model.
    capacity : int
        Upper-bound number of rows per layer.  Over-estimating is fine — the
        file is sparse on filesystems that support it, and truncated at close.
    """

    def __init__(self, output_dir: str, layers: List[int], d_model: int, capacity: int):
        self.output_dir = output_dir
        self.d_model = d_model
        self._mmaps: Dict[int, np.memmap] = {}
        self._npy_paths: Dict[int, str] = {}
        self._offsets: Dict[int, int] = {l: 0 for l in layers}

        for layer_idx in layers:
            layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)
            npy_path = os.path.join(layer_dir, "activations.npy")
            self._npy_paths[layer_idx] = npy_path
            self._mmaps[layer_idx] = np.memmap(
                npy_path, dtype=np.float16, mode="w+", shape=(capacity, d_model)
            )

    def write(self, layer_idx: int, acts_tensor: torch.Tensor) -> int:
        """Write a [N, d_model] float16 tensor to the memmap for *layer_idx*.

        Returns the start offset of the written block (for metadata).
        """
        arr = acts_tensor.numpy()  # already cpu float16
        mmap = self._mmaps[layer_idx]
        offset = self._offsets[layer_idx]
        n = arr.shape[0]
        mmap[offset : offset + n] = arr
        self._offsets[layer_idx] = offset + n
        return offset

    def close(self) -> Dict[int, int]:
        """Flush all memmaps, truncate to actual size, and convert to .pt files.

        Returns a dict mapping layer_idx → actual row count.
        """
        row_counts: Dict[int, int] = {}
        for layer_idx, mmap in self._mmaps.items():
            actual_rows = self._offsets[layer_idx]
            row_counts[layer_idx] = actual_rows

            # Flush and release the memmap
            mmap.flush()
            del mmap

            npy_path = self._npy_paths[layer_idx]
            layer_dir = os.path.dirname(npy_path)

            if actual_rows == 0:
                os.remove(npy_path)
                print(f"[layer {layer_idx}] No activations collected — skipping.")
                continue

            # Re-open at the correct shape (read-only) and convert to .pt
            final = np.memmap(npy_path, dtype=np.float16, mode="r",
                              shape=(actual_rows, self.d_model))
            pt_path = os.path.join(layer_dir, "activations.pt")
            torch.save(torch.from_numpy(final.copy()), pt_path)
            del final

            # Remove the intermediate .npy — downstream code expects .pt
            os.remove(npy_path)

            mb = actual_rows * self.d_model * 2 / 1024 ** 2
            print(f"[layer {layer_idx}] Saved {actual_rows:,} rows  ({mb:.0f} MB)  → {layer_dir}")

        self._mmaps.clear()
        return row_counts


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_multilayer(
    model,
    tokenizer,
    dataset,
    layers: List[int],
    activation_site: str,
    position_type: str,
    output_dir: str,
    max_samples: Optional[int],
    show_progress: bool,
    batch_size: int = 1,
    num_workers: int = 0,
) -> None:
    device = next(model.parameters()).device
    d_model = model.config.hidden_size

    # ── Register one hook per layer ────────────────────────────────────────
    storage: Dict[int, torch.Tensor] = {}
    hook_manager = HookManager(model)
    for layer_idx in layers:
        module = get_target_module(model, layer_idx, activation_site)
        hook_manager.register_forward_hook(
            module,
            (lambda idx: lambda m, i, o: storage.__setitem__(idx, o))(layer_idx),
        )

    # ── Build batched dataloader ───────────────────────────────────────────
    from InformationFlow import CustomDataset as _CustomDataset
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    raw_dataset = _CustomDataset(
        dataset.questions,
        dataset.image_folder,
        tokenizer,
        dataset.image_processor,
        dataset.model_config,
        "CLEVRLite",
        dataset.conv_mode,
    )
    data_loader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_make_batched_collate(pad_id),
        pin_memory=(num_workers > 0),
    )

    questions = dataset.questions
    dataset_dict = dataset.dataset_dict
    total = min(len(questions), max_samples) if max_samples else len(questions)

    # ── Allocate memory-mapped sinks ───────────────────────────────────────
    # Upper-bound capacity: total samples × generous max positions per sample.
    # For "question" positions ~26 tokens is typical; 64 gives headroom.
    # For "all" or "image" positions this could be ~600+; adjust accordingly.
    POS_CAPACITY_MAP = {"question": 64, "last": 1, "attribute": 16,
                        "image": 640, "all": 700}
    max_positions_per_sample = POS_CAPACITY_MAP.get(position_type, 64)
    capacity = total * max_positions_per_sample

    sink = MemmapActivationSink(output_dir, layers, d_model, capacity)

    # ── Per-layer metadata (small — stays in RAM) ──────────────────────────
    layer_metadata: Dict[int, List[dict]] = {l: [] for l in layers}

    iterator = enumerate(data_loader)
    if show_progress:
        iterator = tqdm(iterator, total=(total + batch_size - 1) // batch_size,
                        desc="Collecting activations", unit="batch")

    n_processed = 0
    try:
        for batch_idx, batch in iterator:
            padded_input_ids, stacked_images, sizes, prompts, orig_input_ids = batch

            q_start = batch_idx * batch_size
            if max_samples is not None and q_start >= max_samples:
                break

            B = padded_input_ids.shape[0]
            batch_lines = questions[q_start : q_start + B]

            stacked_images = [img.to(device) for img in stacked_images]

            storage.clear()
            with torch.no_grad():
                image_token_count = _batched_forward(
                    model, orig_input_ids, stacked_images, device
                )

            for item_idx, line in enumerate(batch_lines):
                if max_samples is not None and n_processed >= max_samples:
                    break

                item_ids = orig_input_ids[item_idx].unsqueeze(0)
                question_text = dataset_dict[line["q_id"]]["question"]

                positions = _select_positions(
                    position_type, item_ids, image_token_count,
                    question_text, tokenizer, line,
                )
                if not positions:
                    n_processed += 1
                    continue

                for layer_idx in layers:
                    acts = _normalize_acts_batch(storage.get(layer_idx), item_idx)
                    if acts is None:
                        continue

                    # Slice → CPU → float16 → numpy → memmap (no accumulation)
                    selected = acts[positions].cpu().to(torch.float16)
                    start_offset = sink.write(layer_idx, selected)

                    layer_metadata[layer_idx].append({
                        "question_id": line["q_id"],
                        "question": question_text,
                        "answer": dataset_dict[line["q_id"]].get("answer", ""),
                        "positions": positions,
                        "attribute_tokens": line.get("attribute_tokens", []),
                        "start_idx": start_offset,
                        "count": len(positions),
                    })

                n_processed += 1

    finally:
        hook_manager.remove_hooks()

    # ── Finalize: flush memmaps → .pt, write metadata ─────────────────────
    row_counts = sink.close()

    for layer_idx in layers:
        meta = layer_metadata[layer_idx]
        if not meta:
            continue
        layer_dir = os.path.join(output_dir, f"layer_{layer_idx}")
        with open(os.path.join(layer_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Provenance record ──────────────────────────────────────────────────
    info = {
        "collected_at": datetime.now().isoformat(),
        "layers": layers,
        "activation_site": activation_site,
        "position_type": position_type,
        "n_samples": n_processed,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "rows_per_layer": {str(l): row_counts.get(l, 0) for l in layers},
    }
    with open(os.path.join(output_dir, "collection_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCollection info written to {output_dir}/collection_info.json")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/clevr_lite/sae_layer0_attn_out_question.yaml",
                        help="Any one of the layer configs (model/dataset settings are shared).")
    parser.add_argument("--layers", type=str, default="0,10,11,12,13,14",
                        help="Comma-separated layer indices.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write per-layer activation files.")
    parser.add_argument("--position_type", type=str, default=None,
                        help="Override position_type from config.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Forward-pass batch size. CLEVR-Lite images are uniform so "
                             "no padding overhead. Use 8-16 on a 24GB GPU.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes for image prefetching.")
    parser.add_argument("--show_progress", type=_parse_bool, default=True)
    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",")]

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    position_type = args.position_type or training_cfg.get("position_type", "question")
    activation_site = model_cfg.get("activation_site", "attn_out")

    print(f"Layers       : {layers}")
    print(f"Site         : {activation_site}")
    print(f"Position type: {position_type}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Num workers  : {args.num_workers}")
    print(f"Output dir   : {args.output_dir}")
    print()

    tokenizer, model, image_processor = load_llava_components(
        model_cfg, attn_implementation="sdpa"
    )

    dataset_format = data_cfg.get("format", "csv")
    if dataset_format == "clevr_lite":
        from sae_experiments.data.clevr_lite_dataset import CLEVRLiteVQADataset
        dataset = CLEVRLiteVQADataset(
            data_dir=data_cfg.get("data_dir", "datasets/clevr_lite"),
            split=data_cfg.get("split", "train"),
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
        )
    else:
        from sae_experiments.data.attribute_dataset import AttributeVQADataset
        from sae_experiments.utils.config_utils import resolve_primary_task_type
        dataset = AttributeVQADataset(
            refined_dataset=data_cfg.get("refined_dataset", ""),
            image_folder=data_cfg.get("image_folder", ""),
            tokenizer=tokenizer,
            image_processor=image_processor,
            model_config=model.config,
            task_type=resolve_primary_task_type(data_cfg.get("task_types")),
            conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
        )

    collect_multilayer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        layers=layers,
        activation_site=activation_site,
        position_type=position_type,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        show_progress=args.show_progress,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()