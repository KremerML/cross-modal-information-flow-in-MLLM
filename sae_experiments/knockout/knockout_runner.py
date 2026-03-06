"""Per-layer attention knockout runner for multimodal flows."""

from typing import Dict, Iterable, List, Optional, Tuple

import math
import os

import torch
from tqdm import tqdm

from sae_experiments.ablation import statistical_analysis
from sae_experiments.evaluation import metrics as eval_metrics
from sae_experiments.utils import knockout_utils
from sae_experiments.utils.knockout_utils import sequence_logprob as _sequence_logprob


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def run_knockout_sweep(
    model,
    tokenizer,
    dataset_dict: Dict,
    questions: List[Dict],
    data_loader,
    flows: List[str],
    model_name: str,
    window: int = 1,
    max_samples: Optional[int] = None,
    filter_correct: bool = True,
    normalize_logprob: bool = True,
    progress_desc: str = "Knockout sweep",
) -> Tuple[List[Dict], List[Dict]]:
    num_layers = model.config.num_hidden_layers
    results: List[Dict] = []
    summaries: List[Dict] = []

    total = len(questions)
    if max_samples is not None:
        total = min(total, max_samples)

    total_steps = total * num_layers * max(1, len(flows))
    progress = tqdm(total=total_steps, desc=progress_desc, unit="step")

    for idx, (batch, line) in enumerate(zip(data_loader, questions)):
        if max_samples is not None and idx >= max_samples:
            break

        input_ids, image_tensor, image_sizes, _, _ = batch
        input_ids = input_ids.to(device=next(model.parameters()).device)
        image_tensor = [img.to(device=next(model.parameters()).device) for img in image_tensor]

        question_id = line["q_id"]
        detail = dataset_dict[question_id]
        question_text = detail.get("question", "")
        true_option = detail.get("true option", "").strip()
        false_option = detail.get("false option", "").strip()
        if not true_option or not false_option:
            progress.update(num_layers * max(1, len(flows)))
            continue

        inputs_embeds_shape = knockout_utils.estimate_inputs_embeds_shape(
            model, input_ids, image_tensor, image_sizes
        )
        if inputs_embeds_shape is None:
            progress.update(num_layers * max(1, len(flows)))
            continue

        base_true_lp = _sequence_logprob(
            model,
            tokenizer,
            input_ids,
            image_tensor,
            image_sizes,
            true_option,
            normalize=normalize_logprob,
        )
        base_false_lp = _sequence_logprob(
            model,
            tokenizer,
            input_ids,
            image_tensor,
            image_sizes,
            false_option,
            normalize=normalize_logprob,
        )
        if base_true_lp is None or base_false_lp is None:
            progress.update(num_layers * max(1, len(flows)))
            continue
        base_margin = base_true_lp - base_false_lp
        if filter_correct and base_margin <= 0:
            progress.update(num_layers * max(1, len(flows)))
            continue

        for flow in flows:
            source_range, target_range = knockout_utils.resolve_flow_ranges(
                flow,
                input_ids,
                inputs_embeds_shape,
                question_text,
                tokenizer,
                model_name,
            )
            if not source_range or not target_range:
                progress.update(num_layers)
                continue
            src_tgt_pairs = [(tgt, src) for src in source_range for tgt in target_range]

            for layer in range(num_layers):
                block_config = knockout_utils.build_block_config(
                    layer, num_layers, window, src_tgt_pairs
                )
                new_true_lp = _sequence_logprob(
                    model,
                    tokenizer,
                    input_ids,
                    image_tensor,
                    image_sizes,
                    true_option,
                    normalize=normalize_logprob,
                    block_config=block_config,
                )
                new_false_lp = _sequence_logprob(
                    model,
                    tokenizer,
                    input_ids,
                    image_tensor,
                    image_sizes,
                    false_option,
                    normalize=normalize_logprob,
                    block_config=block_config,
                )
                if new_true_lp is None or new_false_lp is None:
                    progress.update(1)
                    continue
                new_margin = new_true_lp - new_false_lp
                margin_drop = base_margin - new_margin
                results.append(
                    {
                        "question_id": question_id,
                        "flow": flow,
                        "layer": layer,
                        "base_true_logprob": base_true_lp,
                        "base_false_logprob": base_false_lp,
                        "base_margin": base_margin,
                        "new_true_logprob": new_true_lp,
                        "new_false_logprob": new_false_lp,
                        "new_margin": new_margin,
                        "margin_drop": margin_drop,
                    }
                )
                progress.update(1)

    progress.close()

    if not results:
        return results, summaries

    # Aggregate per-flow, per-layer summaries
    by_key: Dict[Tuple[str, int], List[Dict]] = {}
    for row in results:
        key = (row["flow"], row["layer"])
        by_key.setdefault(key, []).append(row)

    for (flow, layer), rows in sorted(by_key.items()):
        base_margins = [r["base_margin"] for r in rows]
        new_margins = [r["new_margin"] for r in rows]
        t_stat, p_val = statistical_analysis.paired_t_test(base_margins, new_margins)
        effect = statistical_analysis.effect_size_cohens_d(base_margins, new_margins)
        summaries.append(
            {
                "flow": flow,
                "layer": layer,
                "samples": len(rows),
                "mean_base_margin": _mean(base_margins),
                "mean_new_margin": _mean(new_margins),
                "mean_margin_drop": eval_metrics.mean_margin_drop(base_margins, new_margins),
                "t_stat": t_stat,
                "p_value": p_val,
                "effect_size": effect,
            }
        )

    return results, summaries
