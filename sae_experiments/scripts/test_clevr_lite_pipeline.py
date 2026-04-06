"""Fast validation test for the CLEVR-Lite knockout pipeline.

Runs a small number of samples (default 10) end-to-end and performs
explicit checks to catch silent errors before a full multi-hour run.

Checks performed:
  1. Dataset adapter structure (all required keys, no missing fields)
  2. true option != false option for every sample
  3. Images load without error (first batch)
  4. Baseline logprobs are finite and non-NaN
  5. Baseline accuracy > 50% (model prefers correct answer)
  6. Knockout effect is non-zero for Image->Question at layer 0
  7. Results file and checkpoint written and valid
  8. Summary has expected structure

Usage:
    python sae_experiments/scripts/test_clevr_lite_pipeline.py \
        --config configs/clevr_lite/knockout_llava15_7b.yaml \
        --max_samples 10
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

_failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not condition:
        _failures.append(label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/clevr_lite/knockout_llava15_7b.yaml")
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()

    from sae_experiments.config.sae_config import load_config, save_config
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("dataset", {})
    knockout_cfg = config.get("knockout", {})

    print(f"\n{'='*60}")
    print(f"CLEVR-Lite pipeline test  (max_samples={args.max_samples})")
    print(f"Config: {args.config}")
    print(f"{'='*60}\n")

    # ── CHECK 1: Dataset adapter structure ───────────────────────────────────
    print("[ 1 ] Dataset adapter structure")
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    model_path = os.path.expanduser(model_cfg.get("name", ""))
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_cfg.get("model_base"), model_name,
        device_map="auto", attn_implementation=None,
    )
    model.eval()

    from sae_experiments.data.clevr_lite_dataset import CLEVRLiteVQADataset
    dataset = CLEVRLiteVQADataset(
        data_dir=data_cfg.get("data_dir", "datasets/clevr_lite"),
        split=data_cfg.get("split", "val"),
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        conv_mode=model_cfg.get("conv_mode", "vicuna_v1"),
    )

    n_questions = len(dataset.questions)
    check("Dataset loaded (>0 questions)", n_questions > 0, f"{n_questions} questions")

    required_keys = {"q_id", "question", "answer", "img_id", "true option", "false option", "attribute_tokens"}
    missing_keys_any = [
        k for k in required_keys
        if any(k not in q for q in dataset.questions[:20])
    ]
    check("All required keys present in questions", len(missing_keys_any) == 0,
          f"missing: {missing_keys_any}" if missing_keys_any else "")

    mismatched = sum(1 for q in dataset.questions if q.get("true option") == q.get("false option"))
    check("true option != false option for all samples", mismatched == 0,
          f"{mismatched} mismatches" if mismatched else "")

    samples_without_attr = sum(1 for q in dataset.questions[:50] if not q.get("attribute_tokens"))
    check("attribute_tokens populated (spot-check 50)", samples_without_attr == 0,
          f"{samples_without_attr}/50 missing" if samples_without_attr else "")

    # ── CHECK 2: Image loading ───────────────────────────────────────────────
    print("\n[ 2 ] Image loading")
    data_loader = dataset.create_dataloader()
    first_batch = next(iter(data_loader))
    input_ids, image_tensor, image_sizes, _, _ = first_batch
    check("First batch loads without error", True)
    check("image_tensor is a list", isinstance(image_tensor, list),
          f"type={type(image_tensor)}")
    check("image_tensor[0] is a tensor", hasattr(image_tensor[0], "shape"),
          f"shape={image_tensor[0].shape if hasattr(image_tensor[0], 'shape') else '?'}")

    # ── CHECK 3: Baseline logprobs ────────────────────────────────────────────
    print("\n[ 3 ] Baseline logprobs (first 10 samples)")
    from sae_experiments.utils.knockout_utils import sequence_logprob, estimate_inputs_embeds_shape

    device = next(model.parameters()).device
    n_correct_baseline = 0
    n_finite = 0
    n_tested = 0

    for idx, (batch, line) in enumerate(zip(data_loader, dataset.questions)):
        if idx >= args.max_samples:
            break
        input_ids, image_tensor, image_sizes, _, _ = batch
        input_ids = input_ids.to(device)
        image_tensor = [img.to(device) for img in image_tensor]

        q_id = line["q_id"]
        detail = dataset.dataset_dict[q_id]
        true_opt = detail["true option"]
        false_opt = detail["false option"]

        lp_true = sequence_logprob(model, tokenizer, input_ids, image_tensor, image_sizes,
                                   true_opt, normalize=True)
        lp_false = sequence_logprob(model, tokenizer, input_ids, image_tensor, image_sizes,
                                    false_opt, normalize=True)

        if lp_true is not None and lp_false is not None:
            n_tested += 1
            if math.isfinite(lp_true) and math.isfinite(lp_false):
                n_finite += 1
            if lp_true > lp_false:
                n_correct_baseline += 1
            print(f"    q_id={q_id}  true_lp={lp_true:.3f}  false_lp={lp_false:.3f}  "
                  f"margin={lp_true - lp_false:.3f}  "
                  f"q={detail['question']!r}  answer={true_opt!r}  foil={false_opt!r}")

    check("All logprobs finite", n_finite == n_tested,
          f"{n_finite}/{n_tested} finite")
    check(f"Baseline accuracy > 50% ({n_correct_baseline}/{n_tested})",
          n_tested > 0 and n_correct_baseline / n_tested > 0.5,
          f"{n_correct_baseline/n_tested:.0%}" if n_tested else "n=0")

    # ── CHECK 4: Knockout effect at layer 0 ──────────────────────────────────
    print("\n[ 4 ] Knockout effect at layer 0 (Image->Question)")
    from sae_experiments.knockout.knockout_runner import run_knockout_sweep

    test_dir = "output/test_clevr_lite_pipeline"
    os.makedirs(test_dir, exist_ok=True)
    checkpoint_path = os.path.join(test_dir, "checkpoint.jsonl")
    # Remove stale checkpoint to get a fresh run
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    results, summaries = run_knockout_sweep(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset.dataset_dict,
        questions=dataset.questions,
        data_loader=dataset.create_dataloader(),
        flows=["Image->Question"],
        model_name=model_name,
        window=1,
        max_samples=args.max_samples,
        filter_correct=True,
        normalize_logprob=True,
        progress_desc=f"Test sweep (n={args.max_samples})",
        checkpoint_path=checkpoint_path,
    )

    check("run_knockout_sweep returned results", len(results) > 0,
          f"{len(results)} rows")
    check("Summaries computed", len(summaries) > 0,
          f"{len(summaries)} entries")

    layer0_rows = [r for r in results if r["layer"] == 0 and r["flow"] == "Image->Question"]
    if layer0_rows:
        mean_drop_l0 = sum(r["margin_drop"] for r in layer0_rows) / len(layer0_rows)
        print(f"    Layer-0 Image->Question mean_margin_drop = {mean_drop_l0:.4f}  "
              f"(n={len(layer0_rows)})")
        check("Layer-0 margin_drop > 0 (knockout has effect)",
              mean_drop_l0 > 0,
              f"mean_drop={mean_drop_l0:.4f}")
    else:
        check("Layer-0 rows present", False, "no rows for layer 0 / Image->Question")

    nan_rows = [r for r in results if not all(
        math.isfinite(v) for v in [r.get("margin_drop", 0), r.get("base_margin", 0)]
    )]
    check("No NaN/inf in results", len(nan_rows) == 0,
          f"{len(nan_rows)} bad rows" if nan_rows else "")

    # ── CHECK 5: Checkpoint file ──────────────────────────────────────────────
    print("\n[ 5 ] Checkpoint file")
    check("checkpoint.jsonl created", os.path.exists(checkpoint_path))
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        check("Checkpoint has at least one line", len(lines) > 0,
              f"{len(lines)} lines")
        try:
            first_entry = json.loads(lines[0])
            check("First checkpoint entry is valid JSON with 'question_id' and 'rows'",
                  "question_id" in first_entry and "rows" in first_entry)
        except json.JSONDecodeError as e:
            check("First checkpoint entry is valid JSON", False, str(e))

    # ── CHECK 6: Results/summary files ───────────────────────────────────────
    print("\n[ 6 ] Results and summary files")
    results_path = os.path.join(test_dir, "knockout_results.json")
    summary_path = os.path.join(test_dir, "knockout_summary.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    check("knockout_results.json written", os.path.exists(results_path))
    check("knockout_summary.json written", os.path.exists(summary_path))

    with open(summary_path) as f:
        loaded_summary = json.load(f)
    summary_keys = {"flow", "layer", "samples", "mean_margin_drop", "effect_size", "p_value"}
    if loaded_summary:
        missing_summary_keys = summary_keys - set(loaded_summary[0].keys())
        check("Summary entries have expected keys", len(missing_summary_keys) == 0,
              f"missing: {missing_summary_keys}" if missing_summary_keys else "")

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if _failures:
        print(f"  {FAIL}  {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  {PASS}  All checks passed. Safe to run the full sweep.")
    print(f"  Test outputs: {test_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
