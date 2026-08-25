# Handoff: active development moved to `vlm-flow-probe`

**2026-08-25.** The model-agnostic core of this repo was extracted into
[`KremerML/vlm-flow-probe`](https://github.com/KremerML/vlm-flow-probe) — an installable package
(`vlmflowprobe`, src layout, console scripts) with a **model adapter interface**: everything
model-specific (prompt construction, preprocessing, token geometry, attention-knockout mechanics,
module resolution) lives in one adapter class per model family, HF-transformers-native. This repo
is now the **frozen archive of record** for the published outputs (`output/`), the paper
(`overleaf/`), and the research log (`docs/`).

## Equivalence

The port is behavior-preserving, established by a four-stage gate (`vfp-gate`) run on the
RTX 4090 against this repo's artifacts:

| stage | result |
|---|---|
| 0 geometry | 256/256 samples: byte-exact prompts, 576 image tokens, question spans exactly equal to the archived `sample_cache.json` positions |
| 1 baselines | margin correlation r = 0.999975; mean \|Δmargin\| = 0.008; 255/256 pred agreement (one flip on a 0.418-vs-0.411 near-tied generation); new-harness self-jitter exactly 0 |
| 2 knockout | Image→Question vs the archived 10-sample sweep: Spearman ρ = 0.9978, identical top-3 layers {0, 11, 14}, no top-5 drift |
| 3 A0 regression | archived layer-11 SAE + catalog, replace-mode 200-feature ablation: mean margin drop **0.21358 vs archived 0.21307** (Δ = 0.0005), per-sample r = 0.99959, all 15 archived control feature sets rerun |

The gate passed **18/18 checks** at `vlm-flow-probe` commit
`323cbc3ac2f9c38d8089bac40d4558897b7448fb` (report committed there as
`gate/reports/gate_report_2026-08-25.json`; reference provenance in `gate/references/SOURCES.md`).
There were no accepted numeric deviations — one criterion refinement (the pred-flip confidence
proxy) is recorded in that repo's `docs/deviation_ledger.md`.

## Two findings about THIS repo made during the port

1. **The "question span" for CLEVR-Lite was always the fallback span.** The sublist match in
   `token_utils.get_question_token_range` never hit for CLEVR-Lite — in context the question
   follows `<image>\n`, and SentencePiece tokenizes `\nwhat` differently from bare `what` — so
   every published "question-position" intervention actually covered the span from the end of the
   image block through the final token, i.e. question + answer-format suffix + `ASSISTANT:`.
   This does not invalidate anything (knockout and ablation used the same span, and the paper's
   claims are about that intervention locus), but "question positions" in the writeup should be
   read as "post-image text positions".
2. **"Layer 31 Image→Question knockout always yields margin_drop = 0.0" (CLAUDE.md) is
   GQA-specific.** It holds only when the span excludes the final token. For CLEVR-Lite the span
   includes the final token — whose position predicts the first answer token — so L31 knockout has
   a small real effect; this repo's own `exp_default` sweep recorded −0.0025 (n = 7084).

## Old module → new module

| here | vlm-flow-probe |
|---|---|
| `sae_experiments/core/*` | `vlmflowprobe/core/*` |
| `sae_experiments/training/*` | `vlmflowprobe/training/*` (trainer takes an adapter) |
| `sae_experiments/ablation/*` | `vlmflowprobe/ablation/*` (rewired to adapter + `ModelBatch`) |
| `sae_experiments/feature_analysis/{causal_feature_identifier,feature_catalog}` | `vlmflowprobe/features/*` |
| `sae_experiments/hooks/hook_utils` | `vlmflowprobe/hooks` (`get_target_module` → `adapter.layer_module`, silent fallback now raises) |
| `sae_experiments/hooks/attention_hooks` (self_attn.forward monkeypatch) | `vlmflowprobe/knockout/mask_hooks` (decoder-layer pre-hooks editing the eager 4D mask) |
| `sae_experiments/hooks/knockout_utils` | split: flow grammar → `knockout/block_config`, scoring → `knockout/scoring`; the `prepare_inputs_labels_for_multimodal` dry-pass machinery deleted (HF input_ids are post-expansion) |
| the three `_select_positions`/`_resolve_positions` copies | one `vlmflowprobe/positions.py` with pinned `COLLECTION_POLICY`/`ABLATION_POLICY` |
| `sae_experiments/data/llava_loader` | `adapters/hf_llava.build_inputs` (prompt reproduced byte-exactly) |
| `sae_experiments/data/clevr_lite*` | `vlmflowprobe/data/{clevr_lite/,datasets}` |
| `sae_experiments/data/activation_collector` | `vlmflowprobe/data/collection` |
| `sae_experiments/pipeline/00–04` | `vfp-knockout`, `vfp-train-sae`, `vfp-identify`, `vfp-ablate`, `vfp-analyze` |
| `tools/{run_multilayer_ablation,analyze_multilayer_ablation,distill_results,collect_activations,generate_clevr_lite,knockout_runner}` | `vfp-multilayer`, `vfp-analyze-multilayer`, `vfp-distill`, `vfp-collect`, `vfp-gen-clevr`, `vlmflowprobe/knockout/runner` |

Archive bugs fixed in the port (unchanged here): stage 00/04 never seeded; stage 02 lacked
`--experiment_dir`/`--experiment_name`; stage 04's default `--results` pointed at a file 03 never
writes; the multilayer runner's GQA branch passed a nonexistent `csv_path` kwarg; `controls_for`
dispatched on condition-id string prefixes; `random_control` defaults silently degraded matched
sampling to uniform (new repo defaults: `activation_mean` + `strict_matching: true`).

## Deliberately left behind

`InformationFlow.py`, `methods.py`, `archive/`, `tools/{knockout_sae_pipeline,
analyze_causal_features,knockout_guided_features}.py`, v1 `feature_analysis/feature_identifier.py`
(+ its test), GQA `data/attribute_dataset.py` + `configs/gqa/`, the notebooks (they read this
repo's committed summaries), `LLM_TECHNICAL_SUMMARY.*` (hand-maintained), and the LLaVA-NeXT
vendored install. The `paligemma-2` branch is superseded by the adapter design.
