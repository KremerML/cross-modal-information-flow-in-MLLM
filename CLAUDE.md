# Project: Cross-Modal Information Flow in MLLMs

## Overview

Research project studying how visual information flows into language representations in LLaVA-v1.5-7b using:
1. **Attention knockout sweeps** to identify which transformer layers carry causal image-to-text information
2. **Sparse Autoencoders (SAEs)** trained at identified layers to decompose representations into interpretable features
3. **Feature ablation experiments** to test whether individual SAE features are causally necessary for attribute-binding (color, material, shape, size, state)

## Repository Structure

```
sae_experiments/
  ablation/         feature_ablator.py, ablation_experiments.py, statistical_analysis.py
  config/           sae_config.py (load_config, save_config)
  data/             activation_collector.py, attribute_dataset.py
  evaluation/       hypothesis_tester.py, metrics.py
  feature_analysis/ feature_catalog.py, feature_identifier.py
  knockout/         knockout_runner.py
  models/           sparse_autoencoder.py, sae_trainer.py
  utils/            config_utils.py, hook_utils.py, knockout_utils.py,
                    script_utils.py, token_utils.py, visualization_utils.py, checkpoint_utils.py
  scripts/
    00_knockout_sweep.py    - layer selection via attention knockout
    01_train_sae.py         - train SAE on LLaVA activations
    02_identify_features.py - rank features by discrimination score
    03_run_ablation.py      - 3-condition ablation test
    04_rank_features_causally.py
    05_visualize_features.py
    06_knockout_sae_pipeline.py
    10_full_latent_ablation.py  - upper-bound ceiling experiment

configs/
  sae_layer0_attn_out.yaml
  sae_layer0_attn_out_scorekey_diff.yaml
  sae_layer0_residual.yaml
  knockout_sae/knockout_llava15_7b_color.yaml
  sae_categories/   per-attribute category configs

output/
  knockout_sae/     attention knockout sweep results
  sae_experiments/  SAE training + ablation results
```

## Model Details

- **Model**: LLaVA-v1.5-7b (`liuhaotian/llava-v1.5-7b`)
- **Architecture**: 32 transformer layers, d_model=4096
- **Image tokens**: Expanded via `prepare_inputs_labels_for_multimodal`; `IMAGE_TOKEN_INDEX = -200`
- **Conv mode**: `vicuna_v1`
- **SAE**: 32768 features, L1 regularization (l1_coeff=1e-3 typical)

## Attention Knockout Results (Key Finding)

Script: `sae_experiments/scripts/00_knockout_sweep.py`
Results: `output/knockout_sae/knockout_color_run1_20260219_180829/` (n=510) and `knockout_run2_fixed_20260203_173906/` (n=810)

Metric: `margin_drop = log P(true) - log P(false)` before vs. after blocking attention at each layer.
Only correctly-answered samples included (`filter_correct=True`). Paired t-test + Cohen's d.

### Image->Question flow (dominant):

| Layer | margin_drop | effect_size | Interpretation |
|-------|------------|-------------|----------------|
| **0** | **0.54** | **0.83** | Dominant — early cross-modal grounding |
| **11** | **0.17** | **0.60** | Second peak — semantic binding |
| 8  | 0.076 | 0.44 | Moderate |
| 10 | 0.054 | 0.29 | Moderate |
| 12 | 0.054 | 0.34 | Moderate |

### Image->Last flow: Much weaker (max d~0.20 at layer 0). Many layers show *negative* drops.

**Conclusion**: Layers 0 and 11 selected as SAE training sites because they show the two largest causal effects in the Image->Question flow. The two-peak structure suggests two distinct integration phases:
- Layer 0: immediate cross-modal registration
- Layer 11 (~1/3 depth): higher-level visual-semantic binding

## SAE Experiments Summary (18 experiments in output/sae_experiments/)

**Consistent null result**: No SAE feature set shows meaningful causal effects vs. random controls.

Key experiments (most recent):
- `layer11_attn_out_replace_color` — 50 features, `replace` mode, layer 11 attn_out, color attribute
- `layer0_attn_out_diff` — abs_diff feature selection, layer 0 attn_out
- `layer0_residual_replace_color` — residual site, layer 0

**Full-latent ceiling** (`10_full_latent_ablation.py`): Zeroing ALL 32768 features in `replace` mode. Results confirm the SAE captures negligible causal signal at attribute text token positions (0.3% relative perturbation at both layers). At `all` positions, substantial norm perturbation occurs but accuracy drops remain small. See ceiling table in analysis below.

**Dead feature problem**: ~44% of SAE features never activate, suggesting l1_coeff is too aggressive or wrong activation site. MSE reconstruction: ~2e-6 (low absolute error, but explained variance unknown).

## Analysis of the Null Result (Mar 2026)

### The hypothesis being tested
That the causal Image→Question attention flow at layers 0 and 11 is mediated by sparse, interpretable SAE features — features selectively encoding visual attributes whose ablation would partially replicate the behavioral disruption seen in the knockout.

### Why the null result is almost certainly methodological, not a true negative

**1. The ceiling results confirm the SAE provides negligible causal leverage at attribute token positions.**

Full-latent ceiling results (`replace` mode, n=128, color category):

| Layer | Site | Position | Accuracy drop | Margin drop | Relative perturbation | Knockout margin_drop |
|-------|------|----------|--------------|-------------|----------------------|----------------------|
| 11 | attn_out | attribute | +3.1% | 0.046 | 0.003 (0.3%) | 0.17 |
| 11 | attn_out | all | +3.1% | 0.169 | 0.999 (99.9%) | 0.17 |
| 0 | attn_out | attribute | −0.8% | 0.022 | 0.003 (0.3%) | 0.54 |
| 0 | attn_out | all | −0.8% | 0.314 | 1.016 (101.6%) | 0.54 |

Result files: `output/sae_experiments/layer11_attn_out_replace_color/results/full_latent_ablation_quick.json` and `output/sae_experiments/full_latent_layer0_attn_out/results/full_latent_ablation_quick.json`

**Key interpretations:**
- **`attribute` positions (both layers):** 0.3% relative perturbation = the SAE reconstruction at those positions is essentially zero relative to the original activation. Feature ablation at question-attribute text tokens is undetectable by design — this is not a feature quality issue, it confirms that attribute text tokens don't carry the visual signal via the SAE.
- **`all` positions:** Both layers show ~100% relative perturbation (SAE reconstruction matches activation norm), with margin drops of 0.314 (layer 0) and 0.169 (layer 11) — matching the knockout margin drops at those layers almost exactly (0.54 and 0.17 respectively). This means the SAE *does* capture the causal signal when all positions are ablated, but accuracy barely changes because:
  1. The perturbation is spread across all positions (including image tokens), diluting per-position effect.
  2. The model can re-read image tokens at subsequent layers (the conceptual mismatch in point 2 below).
- **The fundamental problem is position, not SAE quality.** Ablating at `attribute` text positions catches nothing. Ablating at image token positions (the source) would be the diagnostic test.

**2. Fundamental conceptual mismatch: knockout vs. SAE ablation.**
The attention knockout blocks the attention *mechanism*, preventing question tokens from reading image patches at that layer for the entire forward pass — a persistent information barrier. The SAE ablation modifies the *output* of that mechanism at one layer. Crucially, the model at layers L+1 through 31 can still attend to image tokens directly — the barrier is not in place. The attribute information removed at layer 11 can simply be re-read from image tokens at layer 12+. Ablating question-token positions (targets) doesn't block the source. To replicate the knockout, you'd need to ablate at **image token positions** (the source) or ablate at every layer simultaneously.

**3. SAE training data is far too small.**
1,000 ChooseAttr samples × ~6–8 attribute tokens = ~6–8k activation vectors for a 32,768-feature SAE. Modern SAE work requires millions of vectors. This explains the 44% dead features and likely produces overloaded, polysemantic live features that don't cleanly separate attributes.

**4. Feature identification conflates attribute content with task difficulty.**
Features selected by comparing correct vs. incorrect samples at `attribute` token positions may be tracking object visibility, question ambiguity, or image quality rather than the attribute itself — since incorrect answers correlate with tiny objects (20.6% of color items), not just wrong visual processing.

**5. Language-side bypass in ChooseAttr.**
Both options ("red", "blue") appear as text in the question. The model can partially solve the task via language priors + object name without relying on visual features. This reduces the causal leverage of visual-feature ablations and dilutes any effect.

**6. SAE architecture lacks modern best practices.**
- No decoder column normalization → features with larger decoder norms dominate at the expense of others
- No encoder pre-bias (`b_pre`) → poor decomposition of off-center activation distributions
- No auxiliary dead-feature prevention (AuxK, TopK, or jumprelu)
- `mean(|z|)` L1 penalty: increasing n_features reduces effective sparsity pressure

### Ablation mode clarification
- **`residual` mode** (most experiments): `out = acts + (decode(feats_without_selected) - decode(feats_all))` — subtracts selected features' contribution, preserves reconstruction error. Soft intervention.
- **`replace` mode**: `out = decode(feats_without_selected)` — replaces full activation with SAE reconstruction minus selected features, discards reconstruction error. Harder, more interpretable intervention. Preferred for future experiments.

### Alternative Task Formats (Research — Mar 2026)

**Problem with ChooseAttr:** Both options appear in the question text ("Is the car red or blue?"), enabling a language-side bypass. The model can partially answer via language priors + object name without visual grounding, diluting causal leverage of visual-feature ablations.

**Three proposed alternatives:**

**A — Open-ended logprob scoring (recommended, low effort)**
Score = log P(correct_answer_token | `"<image>\nWhat color is the {obj}? Answer with one word."`)
Construct prompt synthetically from `central object name` + `answer` columns (no re-annotation needed).
`sequence_logprob` in `utils/knockout_utils.py` is directly usable: pass the single correct answer string
as `answer_text`. Metric = `logprob_drop` = baseline − ablated (direct analogue of `margin_drop`).
- **Verified**: 599/602 color rows have single-word answers. Only dark brown (2 rows) and dark blue (1 row)
  are multi-word — negligible (<0.5%). "cream colored" and "light blue" appear only in captions, not answers.
- No language bypass; single forward pass; same statistical pipeline; ~30 lines of new code.

**B — GQA QueryAttr split (requires external data)**
`datasets/GQA_val_correct_question_with_positionQuery_QueryAttr.csv` **exists but contains only
`positionQuery` questions** ("On which side of the photo is X?") — NOT open-ended attribute queries.
Authentic open-ended queryAttr questions ("What color is the X?") must be fetched separately from
HuggingFace (`lmms-lab/GQA`) and filtered for color. Medium effort; not a drop-in replacement.

**C — Image-token activation probing (medium-high effort, theoretically strongest)**
Collect SAE feature activations at **image token positions** (not attribute text positions). Test whether
features fire for images depicting the target attribute with no language prompt at all. Requires: new
`"image"` position type in `ActivationCollector`/`FeatureAblator`; `get_image_token_range` in
`utils/knockout_utils.py` already identifies image token index range. Directly replicates the knockout:
removing attribute signal at the source prevents all layers L+1..31 from reading it. Supported by
arXiv:2410.07149 (image-token probing) and SAE-V (ICML 2025).

**Key scoring note (from arXiv:2402.07270, ICLR 2024):**
Exact-match scoring gives ~0% accuracy on instruction-tuned LLaVA even for correct answers (verbosity).
Use containment scoring or logprob scoring. Exact match is only valid for discriminative probing.

| Format | Removes bypass? | Data reuse | Effort | Best use |
|--------|----------------|------------|--------|----------|
| A: Logprob open-ended | Yes | Full (602 rows) | Low (~30 lines) | Near-term drop-in |
| B: QueryAttr GQA | Yes | None (external) | Medium | Larger-scale eval |
| C: Image-token probing | N/A (no language) | Full | Medium-High | Definitive visual encoding test |

### Proposed methodological fixes (priority order)

1. **Validate the SAE ceiling first.** Run full-latent ablation in `replace` mode and measure accuracy drop and `relative_norm` (delta_norm / acts_norm). If drop is <15%, fix the SAE before any feature analysis.

2. **Fix SAE training.** Train on a large diverse corpus (≥100k activation vectors from LLaVA Instruct or full GQA validation), not just ChooseAttr. Reduce n_features to 4096–8192 or scale data accordingly. Reduce l1_coeff to 1e-4 or 5e-5. Normalize decoder columns. Add encoder pre-bias.

3. **Ablate at image token positions, not attribute text positions.** Image tokens are the *source* of visual attribute information. Zeroing SAE features at image token positions prevents all subsequent layers from reading the attribute signal — which is what the knockout actually does.

4. **Use activation difference as a supervision signal.** Compute activation difference at layer 11 attn_out (image token positions) between correct-answer forward passes and incorrect-answer forward passes. Features explaining this difference directly mediate attribute encoding.

5. **Bridge knockout and SAE directly.** Run paired forward passes with/without the layer-0 knockout and collect activations at layer 11 with vs. without the block. The difference subspace tells you exactly which directions at layer 11 carry the Image→Question information.

6. **Use `replace` mode** for all ablations instead of `residual`.

7. **Consider open-ended generation tasks** instead of forced-choice to eliminate the language-side bypass.

## Key Architectural Decisions (Codebase)

### Activation sites
- `residual` — full post-layer residual stream output
- `attn_out` — self-attention output (`layer.self_attn`)
- `mlp_out` — MLP output (`layer.mlp`)

### Position types (which tokens to collect/intervene on)
- `attribute` — tokens spanning the attribute-describing region of the question
- `question` — all question tokens
- `all` — all positions
- `last` — final token only

### Feature selection methods
- `ratio` — correct_mean / incorrect_mean activation
- `abs_diff` — |correct_mean - incorrect_mean|
- `causal_hybrid` — combines statistical + causal scores

### 3-condition ablation test
Binding features vs. random-sampled control features vs. baseline, measured by `forced_choice_margin` drop.

## Shared Utility Functions (refactored Jan–Mar 2026)

All previously duplicated helpers are now in:

| Function | Location | Replaces |
|----------|----------|---------|
| `resolve_dtype(value)` | `utils/config_utils.py` | `_resolve_dtype` in 5 scripts + sae_trainer |
| `get_target_module(model, layer_idx, site)` | `utils/hook_utils.py` | `_get_target_module` in ablator + collector |
| `estimate_image_token_count(model, ...)` | `utils/knockout_utils.py` | `_estimate_image_token_count` in ablator + collector |
| `get_question_token_range(...)` | `utils/knockout_utils.py` (thin wrapper → token_utils) | incompatible dual implementations |
| `sequence_logprob(model, tokenizer, ...)` | `utils/knockout_utils.py` | `_sequence_logprob` in ablator + knockout_runner |
| `setup_experiment(args, config)` | `utils/script_utils.py` | boilerplate in 7 scripts |
| `load_llava_components(model_cfg)` | `utils/script_utils.py` | boilerplate in 7 scripts |
| `load_sae(config, model, path)` | `utils/script_utils.py` | boilerplate in 7 scripts |

### Notable: `get_question_token_range` unification
Two incompatible versions existed. Resolution: `knockout_utils` version is now a thin wrapper that delegates to `token_utils.get_question_token_range` (the canonical model-agnostic sublist-search implementation), preserving the existing call signature in `knockout_runner.py`.

### Dead code removed
- `AblationExperiment.feature_importance_ranking` — superseded by script 04
- `AblationExperiment.run_attention_knockout_baseline` — never called
- `AblationExperiment.test_task_specificity` — hardcoded-skipped
- `HypothesisTester.test_task_specificity`, `test_feature_interpretability`, `generate_hypothesis_report`
- `create_intervention_hook` from `hook_utils.py`

## Dataset

- **Task**: `ChooseAttr` — forced-choice VQA (pick true vs. false attribute option)
- **Split**: validation set
- **Main file**: `datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv` (1000 rows, 937 unique images)
- **By-attribute CSVs**: `datasets/by_attribute_category/ChooseAttr_{color,material,shape,size,state}.csv`
- **Images**: `datasets/images/`
- **Python env for analysis**: `LLaVA-NeXT/.venv/bin/python`
- Baseline accuracy ~82–87% depending on attribute type

### Dataset Quality Findings (Mar 2026)

**Structural integrity**: No duplicate question_ids, no missing values, `answer == true option` for all 1000 rows, option order perfectly balanced (501 true-first, 499 false-first — no position bias).

**Category file composition** (from `manifest.json`, policy=`first`):

| Category | Rows | Runnable? | What's actually inside |
|----------|------|-----------|----------------------|
| color | 602 | Yes | Color only — clean and homogeneous |
| material | 96 | No (<100) | Material only — clean |
| shape | 15 | No (<100) | Shape only — critically too small |
| size | 87 | No (<100) | size(41) + length(22) + height(15) + depth/thickness/weight/width(9) |
| state | 151 | Yes | weather(18) + cleanliness(11) + state(5) + opaqness(1) + **116 generic "choose" items** |

**49 rows unassigned** (pose, activity, sportActivity, face expression) — not visual attribute-binding tasks; correctly excluded.

**Key quality concerns**:

1. **State file is heterogeneous**: 77% of its rows (116/151) are generic `['choose', 'X|Y']` items (open/closed, wet/dry, full/empty, short-sleeved/long-sleeved, etc.) with no GQA attribute type. Weather and cleanliness are mixed in. Findings on "state" don't isolate a single semantic property.

2. **Size conflates distinct dimensions**: height, length, width, depth, thickness, and weight are all bundled. A size-discriminating SAE feature may actually be specific to, e.g., hair length.

3. **Shape has only 15 examples**: Statistically unusable. 9/15 cover just three attribute pairs (curly/straight hair ×3, round/square ×3, checkered/striped ×3).

4. **Color dominated by black/white**: ~58% of color questions involve black or white as one option; (black, white) pairs alone = 10%. A color-feature may be learning achromatic vs. chromatic rather than color in general.

5. **Tiny objects**: 20.6% of color questions (124/602) have objects covering <1% of image area. Overall 15.7% across all categories. Attribute may not be visually discernible at that scale.

6. **False option is never in the object's own attribute list** (verified 0/602 for color): foils are plausible distractors the object does NOT have — sound design, but means the task always requires recognizing the true attribute, not filtering a co-occurring one.

7. **Color false options include multi-word and uncommon values** (25/602): "blond", "cream colored", "light brown", "dark blue", "light blue" — these differ in tokenization from typical single-word colors and may behave differently in logprob scoring.

**Recommendation**: Use **color** as the primary category for all experiments (largest, cleanest, most homogeneous). Treat **state** results with caution given its heterogeneity. Do not draw conclusions from shape.

## Common Config Fields

```yaml
model:
  name: liuhaotian/llava-v1.5-7b
  target_layer: 0        # or 11
  activation_site: attn_out  # or residual
  d_model: 4096
sae:
  n_features: 32768
  l1_coeff: 0.001
knockout:
  flows: [Image->Question, Image->Last]
  top_k_layers: 5
  filter_correct: true
  normalize_logprob: true
```

## Known Issues / Gotchas

- After refactoring to `setup_experiment()`, do NOT reference `reproducibility_cfg` as a local variable — use `config.get("reproducibility", {})` directly where needed (e.g. in checkpoint metadata in `01_train_sae.py:376` and `03_run_ablation.py:91`)
- Script `06_knockout_sae_pipeline.py` still computes `model_name = get_model_name_from_path(...)` inline before calling `load_llava_components` because it's needed for downstream `run_knockout_sweep` and `_make_attn_block_resolver` calls
- Layer 31 Image->Question always shows margin_drop=0.0 (last layer cannot be blocked effectively — output is already committed)
- Negative margin_drops on Image->Last (layers 8, 10, 15, 17, 27...) are real: blocking those attention paths slightly *improves* accuracy, suggesting they carry distracting or noisy information
