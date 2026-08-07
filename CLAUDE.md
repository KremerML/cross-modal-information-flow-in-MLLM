# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Mechanistic-interpretability research on LLaVA-v1.5-7b (32 layers, d_model=4096, `conv_mode="vicuna_v1"`).
Two intervention techniques are combined to study how visual information reaches language representations:

1. **Attention knockout** — block an attention flow (e.g. `Image->Question`) at one layer, measure the drop in
   `margin = log P(true option) - log P(false option)`. Identifies *which layers* carry causal image→text information.
2. **SAE feature ablation** — train a sparse autoencoder on activations at those layers, identify causally
   important features, zero them, measure the same margin drop. Tests whether the flow is mediated by
   *sparse interpretable features*.

Upstream is the CVPR paper repo (`README.md`, `archive/`); everything under `sae_experiments/` is this fork's work.

## Environment

**All Python must run through the LLaVA virtualenv** — the `llava` package is only importable there:

```bash
LLaVA-NeXT/.venv/bin/python ...        # or: source LLaVA-NeXT/.venv/bin/activate
```

Python 3.10.20, torch 2.10.0+cu128, transformers 4.57.6. `LLaVA-NeXT/` and `datasets/` are gitignored
(the LLaVA install and image data live there). Extra deps beyond the LLaVA env: `requirements_sae.txt`.

## Artifacts not in git

A fresh clone is ~24 MB and contains **all the results needed to read, cite, and write up this
project** — but none of the bulk artifacts. Nothing here needs regenerating to understand the
findings; regenerate only to re-run experiments.

| Absent | Size | How to get it back | Cost |
|---|---|---|---|
| `LLaVA-NeXT/` (install + venv) | 8.7 GB | Upstream LLaVA-NeXT install + `requirements_sae.txt` | minutes |
| `datasets/images/` (GQA) | 21 GB | Public GQA download | download-bound |
| `datasets/clevr_lite/` | 413 MB | `tools/generate_clevr_lite.py` with `datasets/clevr_lite/config.json` (seed 32) — **deterministic, reproduces exactly** | ~1h |
| `output/**/sae_checkpoint.pt` (20 files) | ~20 GB | `pipeline/01_train_sae.py` per layer | ~20 GPU-hours total |
| `output/**/knockout_results.json` (per-sample) | 58 MB | `pipeline/00_knockout_sweep.py` | 33h for the n=7084 sweep |
| `output/**/_activation_cache/` | varies | `tools/collect_activations.py` | hours |
| `output/**/feature_<N>.png` (220 files) | 247 MB | `04_analyze_results.py` | minutes; all v1-era, superseded |

**Distilled, not lost.** The bulk result JSONs are gitignored but each has a committed
`*.summary.json` sibling carrying the top-500 features, the full score distribution, and
aggregate statistics — ~4% of the size, and the form the writeup actually cites. Regenerate
with `tools/distill_results.py --root output` after any new run. Two fields in those summaries
exist *only* there, because the runner never stored them: per-sample `margin_drop` (derived
from `baseline_margin − ablated_margin`) and `*_prediction_changes` (flip counts).

`knockout_results.json` is the exception — it already had a complete `knockout_summary.json`
sibling (n, means, t-stat, p, Cohen's d) before any of this, so it was simply untracked.

**Start here, in this order:**
1. `output/sae_experiments/LLM_TECHNICAL_SUMMARY.md` — every verified result number, per-layer
   tables, trust levels per run, and which files to load. Its `.json` twin is the same data,
   machine-readable, generated from the result files.
2. `docs/MEMORY.md` — accumulated project context and the caveats that must survive into the writeup.
3. `docs/CLAUDE.md` — the research log and the reasoning behind the current pipeline.

## Commands

```bash
# Tests (23 unittest-style tests, run under pytest; CPU-only, no model download, ~4s)
LLaVA-NeXT/.venv/bin/python -m pytest tests/ -q
LLaVA-NeXT/.venv/bin/python -m pytest tests/test_sae.py::TestSparseAutoencoder::test_loss_and_grad -q

# Fast end-to-end smoke check of the CLEVR-Lite knockout path (10 samples, loads the model)
LLaVA-NeXT/.venv/bin/python sae_experiments/tools/test_clevr_lite_pipeline.py \
    --config configs/clevr_lite/knockout.yaml --max_samples 10
```

Run `pytest` as `python -m pytest` from the repo root — there is no `conftest.py`, `pyproject.toml`, or
installed package, so `sae_experiments` is importable only via the CWD on `sys.path`.

### Pipeline

Stages are `sae_experiments/pipeline/NN_*.py`, run in order, each driven by the same YAML config.
The numeric filename prefix means they cannot be imported as modules — always invoke by path
(each appends the repo root to `sys.path` itself).

```bash
PY=LLaVA-NeXT/.venv/bin/python
CFG=configs/clevr_lite/sae_layer11_attn_out_question.yaml

$PY sae_experiments/pipeline/00_knockout_sweep.py --config configs/clevr_lite/knockout.yaml
$PY sae_experiments/pipeline/01_train_sae.py --config $CFG --show_progress true
$PY sae_experiments/pipeline/02_identify_features_causal.py --config $CFG --target margin --position_type question --top_k 200
$PY sae_experiments/pipeline/03_run_ablation.py --config $CFG --skip_passthrough --max_samples 256
$PY sae_experiments/pipeline/04_analyze_results.py --config $CFG --results <path from stage 03>
```

Common flags across stages: `--config` (required), `--experiment_dir` / `--experiment_name` (override output
location), `--max_samples`. Progress is `--show_progress true|false` on 00/01 but `--no_progress` on 02/03 —
they are not uniform.

Multi-layer orchestration lives in `scripts/*.sh` (resumable — each loop skips layers whose output already
exists): `run_full_pipeline.sh` (train + causal feature ID for layers 0,10,11,12,13,14),
`run_ablation_all_layers.sh` (ablation for the same set), `collect_activations_clevr_lite_question.sh`.

### Standalone tools

`sae_experiments/tools/` (unnumbered, no pipeline ordering):

```bash
# Pre-collect activations for many layers in one forward pass, then train from cache (much faster
# than re-running the model per layer — 01_train_sae skips model loading entirely with --activations_path)
$PY sae_experiments/tools/collect_activations.py --config $CFG --layers 0,10,11,12,13,14 --output_dir <dir>
$PY sae_experiments/pipeline/01_train_sae.py --config $CFG --activations_path <dir>

$PY sae_experiments/tools/generate_clevr_lite.py --output_dir datasets/clevr_lite --num_train 50000 --num_val 2000
```

## Architecture

### Config-driven everything

`sae_experiments/core/config.py` holds a `DEFAULT_CONFIG` dict; `load_config(path)` deep-merges the YAML over it
and returns a `Config` wrapper with a shallow `.get(section, default)`. **A config file only needs to state its
diffs** — every section (`model`, `sae`, `training`, `ablation`, `random_control`, `evaluation`, `knockout`,
`feature_identification`, `reproducibility`, `experiment`, `dataset`) is always present at runtime.

Consequence: a key you don't see in a YAML is still active with its `DEFAULT_CONFIG` value. Check
`core/config.py` before assuming a behaviour is off.

### Experiment directories

`utils/checkpoint_utils.resolve_experiment_dir` resolves, in order: `--experiment_dir` →
`experiment.output_dir` → `{experiment.output_base}/{experiment.name}[_timestamp]`. Output paths therefore come
from `experiment.name` **in the config, not from the config filename**.

The stage-02/03 contract is a naming convention, not a flag:

- 01 writes `{experiment_dir}/sae_checkpoint.pt`
- 02 writes `{experiment_dir}_causal/` (suffix from `--output_suffix`, default `causal`):
  `causal_feature_catalog.json`, `causal_feature_stats.json`, `causal_summary.json`
- 03 reads `{experiment_dir}_causal/causal_feature_catalog.json` (falling back to
  `{experiment_dir}/feature_catalog.json`) and writes `{experiment_dir}_causal/results/ablation_v2_results.json`

### Dataset adapter contract

`data/attribute_dataset.AttributeVQADataset` (GQA CSV) and `data/clevr_lite_dataset.CLEVRLiteVQADataset`
(synthetic) are interchangeable and selected by `dataset.format` (`"csv"` vs `"clevr_lite"`) — that branch is
duplicated inline in stages 00–03. Both expose:

- `.questions` — list of dicts, each with `q_id`, `question`, `true option`, `false option`
- `.dataset_dict` — `q_id -> detail` (the same dicts)
- `.tokenizer`, `.create_dataloader()` yielding `(input_ids, image_tensor, image_sizes, ...)`

CLEVR-Lite synthesises a `false option` by deterministically sampling a distractor from the closed attribute
set, so open-ended CLEVR questions still support the forced-choice `margin` metric used everywhere.

### Intervention layer (`hooks/`, and the two things that hook the model)

`hook_utils.HookManager` is a context manager that registers and guarantees removal of forward hooks.
`get_target_module(model, layer_idx, site)` maps an **activation site** to a module:

| `activation_site` | module |
|---|---|
| `residual` | the decoder layer itself (post-layer residual stream) |
| `attn_out` | `layer.self_attn` |
| `mlp_out` | `layer.mlp` |

**Position types** decide which token indices an intervention touches. Implemented twice, in near-identical
`_select_positions` (`data/activation_collector.py`) and `_resolve_positions`
(`ablation/feature_ablator.py`, `feature_analysis/causal_feature_identifier.py`) — change one, check the others:

| `position_type` | positions |
|---|---|
| `question` | question token span (via `utils/token_utils.get_question_token_range`) — the default for all current work |
| `image` | `[img_placeholder_idx, img_placeholder_idx + image_token_count)` — the expanded visual patches |
| `attribute` | attribute-token subspan of the question (GQA only; falls back to the whole question span) |
| `last` | final position of the expanded sequence |
| `all` | everything (`None` positions → no masking) |

Position indices are into the **post-expansion** sequence: LLaVA's `prepare_inputs_labels_for_multimodal`
replaces the single `IMAGE_TOKEN_INDEX = -200` placeholder with `image_token_count` patch embeddings, so
`input_ids` indices and hidden-state indices differ. `hooks/knockout_utils.estimate_image_token_count` does a
dry expansion to get that count.

### Ablation modes (`ablation/feature_ablator.py:create_ablation_hook`)

- `replace` — `out = decode(feats_with_selected_zeroed)`; discards SAE reconstruction error. Hard intervention.
  **Every active config uses this.**
- anything else (i.e. `residual`) — `out = acts + (decode(feats_mod) - decode(feats_full))`; the reconstruction
  error cancels, so only the selected features' contribution is removed. Soft/"error-preserving delta" mode.

`ablation/ablation_experiments.AblationExperiment.run_three_condition_test` is the core comparison: binding
features vs. N random control sets vs. baseline. Configs request `random_sampling: "matched"`, which is *meant*
to sample random features matched on `random_control.matched_metric` — a random set with the same activation
profile, not uniformly random indices.

**It silently did not do that in any run to date.** Configs set `matched_metric: "correct_mean"`, but v2 stats
files carry only `causal_score` / `activation_mean` / `gradient_mean`. `_extract_metric_value` falls back
through `correct_mean → ratio → diff → incorrect_mean`, finds none of them, returns `None`, and
`_sample_matched_random_features` takes its `rng.choice(sorted(available))` branch — **uniform**. At layer 11
that means controls with median activation 6.1e-08 against the binding set's 0.117, i.e. the same near-dead
"ghost features" the project diagnosed in v1 *selection*, surviving in the *control* arm. Every published
z-score is inflated by this. Use `matched_metric: "activation_mean"` and `random_control.strict_matching: true`
(which raises instead of falling back) for new work; the default stays permissive so existing runs reproduce,
but it now warns loudly when it falls back.

### Feature identification: v1 vs v2

Two generations coexist and must not be confused.

- **v1** (`feature_analysis/feature_identifier.py`) — statistical: score features by `ratio` / `abs_diff` of mean
  activation on correct vs. incorrect samples. Produced 18 consecutive null ablation results because it selected
  "ghost" features with ~1e-5 activations. Kept for reference; the `feature_identification` config section
  (`selection_method`, `score_key`, `discrimination_threshold`, …) belongs to it.
- **v2** (`feature_analysis/causal_feature_identifier.py`, stage 02) — gradient attribution: insert the SAE into
  the forward pass, backprop the target (`margin` or `correct_logit`) to feature activations, score
  `|grad| * |activation|` averaged over samples. This is the current method and the one that produced a positive
  result. Grounded in Marks et al. 2024 (Sparse Feature Circuits) / Agrawal et al. 2025.

### Legacy surface

`InformationFlow.py` and `methods.py` at the repo root are backwards-compat re-export shims for the original
paper code and the notebooks; the canonical implementations are `sae_experiments/data/llava_loader.py` and
`sae_experiments/hooks/attention_hooks.py`. `archive/` holds superseded Gen 0–2 scripts and configs with their
original structure — read it for history, don't wire new code to it.

## Conventions

- Pipeline scripts: `NN_verb_object.py`, numbered by dependency order, no dataset/layer/attribute in the name.
- Tool scripts: `verb_object.py`, unnumbered.
- Configs: grouped by dataset (`configs/clevr_lite/` active, `configs/gqa/` reference); pattern
  `{component}_{layer}_{site}_{position}.yaml`; one `knockout.yaml` per dataset dir; variants append
  `_v2` / `_replace` / `_causal` / `_holdout`.
- Shell scripts: `scripts/{verb}_{object}_{dataset}_{position}.sh`.
- Existing output dirs are never renamed; conventions apply to new runs only.

## Gotchas

- `03_run_ablation.py`'s module docstring says it "always uses error-preserving delta mode" — **it doesn't**.
  `run_three_condition_test` reads `ablation.mode` from the config, and every active config sets `replace`.
  Only the pass-through baseline in that script hardcodes delta mode (which is why it is ~0 by construction).
- `04_analyze_results.py` defaults `--results` to `{experiment_dir}/results/ablation_results.json`, but stage 03
  writes `{experiment_dir}_causal/results/ablation_v2_results.json`. Pass `--results` explicitly.
- After the `setup_experiment()` refactor, don't reference `reproducibility_cfg` as a local in stage scripts —
  use `config.get("reproducibility", {})`.
- Long knockout sweeps are resumable: `tools/knockout_runner.py` appends one fsync'd JSONL line per completed
  sample and skips already-seen `q_id`s on restart. `output/**/checkpoint.jsonl` is gitignored.
- `01_train_sae.py` deletes the LLaVA model and empties the CUDA cache before training, and reassembles chunked
  activations only afterwards — keep that ordering if you touch it, it is what makes 32768-feature SAEs fit.
- Layer 31 `Image->Question` knockout always yields `margin_drop=0.0` (nothing downstream to affect).
- Negative `margin_drop` values are real signal, not bugs: several layers carry inhibitory/distracting
  information and blocking them slightly improves accuracy.

## Research context

`docs/CLAUDE.md` is the research log — knockout result tables, the v1 null-result post-mortem, dataset quality
analysis, and methodology decisions. Read it before designing an experiment; it is the reason the current
pipeline looks the way it does. Companion writeups: `docs/feature_id_v2_breakthrough_report.md`,
`docs/ablation_methodology_review.md`, `docs/REPORT_SAE_EXPERIMENT.md`.
`docs/README_SAE_EXPERIMENT.md` documents the archived Gen 1–2 GQA pipeline and its paths are stale.
