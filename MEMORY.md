# Project Memory: cross-modal-information-flow-in-MLLM

Full project context is in `CLAUDE.md` at the repo root. Key points for quick recall:

## Research Goal
Study cross-modal information flow in LLaVA-v1.5-7b using attention knockout + SAE feature ablation on ChooseAttr VQA task (forced-choice attribute binding: color, material, shape, size, state).

## Key Finding: Knockout Results
- Layer 0 Image->Question: effect_size=0.83, margin_drop=0.54 — dominant layer
- Layer 11 Image->Question: effect_size=0.60, margin_drop=0.17 — second peak
- Image->Last flow is much weaker overall
- SAE experiments trained at attn_out at these two layers

## SAE Ablation Status
Consistent null result across 18 experiments — binding features show no significant causal effect vs. random.

**Full-latent ceiling results** (`replace` mode, n=128, color; `10_full_latent_ablation.py`):

| Layer | Position | Accuracy drop | Margin drop | Relative perturbation | Knockout margin_drop |
|-------|----------|--------------|-------------|----------------------|----------------------|
| 11 | attribute | +3.1% | 0.046 | 0.003 (0.3%) | 0.17 |
| 11 | all | +3.1% | 0.169 | 0.999 (99.9%) | 0.17 |
| 0 | attribute | −0.8% | 0.022 | 0.003 (0.3%) | 0.54 |
| 0 | all | −0.8% | 0.314 | 1.016 (101.6%) | 0.54 |

**Key finding:** The `attribute` position rows (0.3% perturbation at both layers) confirm that attribute text tokens carry essentially no visual signal via the SAE — the problem is *position*, not SAE quality. The `all` position rows show the SAE captures ~100% of activation norm and the margin drops match the knockout results (layer 0: 0.314 vs knockout 0.54; layer 11: 0.169 vs knockout 0.17), but accuracy barely changes because the model re-reads image tokens at subsequent layers.

**The null result is almost certainly methodological.** Key issues:
- **Position mismatch**: ablating at `attribute` text token positions (targets) catches nothing; should ablate at **image token positions** (source)
- Fundamental mismatch: knockout = persistent attention barrier; SAE ablation = single-layer output mod; model re-reads image at layers L+1 to 31
- **SAE ablation is NOT single-token**: `create_ablation_hook` receives the full `[batch, seq_len, d_model]` tensor and corrupts ALL positions simultaneously. With `position_type="all"`, all 576 image tokens + question tokens are replaced at once. The small effect is cross-layer redundancy, not single-token processing.
- **Why full-latent `all` positions still shows small accuracy drop**: Corruption at layer L modifies one additive component in the residual stream. Layers 1–31 each re-attend to image token positions whose residual stream still contains the unmodified vision encoder embeddings. The knockout has a large effect precisely because it applies a *persistent mask* across the entire forward pass — a barrier SAE ablation cannot replicate without applying at every layer simultaneously.
- SAE trained on ~6–8k vectors for 32,768 features (needs millions); 44% dead features is expected
- ChooseAttr forced-choice puts both options in question text — language-side bypass dilutes visual ablation effects
- SAE lacks: decoder normalization, encoder pre-bias, dead-feature prevention; l1_coeff=1e-3 too large

**Priority fixes:**
1. Ablate at **image token positions**, not attribute token positions (most impactful fix based on ceiling data)
2. Train SAE on ≥100k diverse activations; reduce l1_coeff to 1e-4; normalize decoder; smaller n_features
3. Use knockout-difference activations as SAE training/supervision signal
4. Use `replace` mode (not `residual`) for all ablations — already done

## Codebase State (post-refactor, Mar 2026)
- Utility functions centralized: `resolve_dtype`, `get_target_module`, `estimate_image_token_count`, `sequence_logprob`, `get_question_token_range` all in `sae_experiments/utils/`
- Setup helpers in `utils/script_utils.py`: `setup_experiment`, `load_llava_components`, `load_sae`
- Dead methods removed from `ablation_experiments.py` and `hypothesis_tester.py`
- Bug: after refactor, reference `config.get("reproducibility", {})` directly — not `reproducibility_cfg` local var

## Dataset Quality (analyzed Mar 2026)
- **color** (602 rows): best category — clean, homogeneous. ~58% involve black/white options; 20.6% tiny objects (<1% image area).
- **material** (96 rows): clean but below runnable threshold (100).
- **shape** (15 rows): too small for any analysis.
- **size** (87 rows): heterogeneous — bundles height/length/width/depth/thickness/weight.
- **state** (151 rows): very heterogeneous — 77% are generic "choose X|Y" items (open/closed, wet/dry, etc.) with no GQA attribute type; also includes weather and cleanliness.
- 49 rows unassigned (pose/activity/sportActivity) — correctly excluded, not visual attribute-binding.
- False options are never in the object's own attribute list (sound foil design).
- Option order is balanced (501 true-first, 499 false-first).
- **Use color for primary experiments; treat state results with caution; ignore shape.**
- Python env: `LLaVA-NeXT/.venv/bin/python`

## Active Plan
Staged methodology improvement plan saved at `/home/ron/.claude/plans/optimized-discovering-moon.md`.
- **Pre-requisite**: create branch `methodology-improvements` before any code/config changes
- **Stage 1** (parallel): Plan A (ceiling + replace mode configs), Plan B (research open-ended tasks), Plan C (image position type)
- **Stage 2** (parallel): Plan D (fix SAE architecture + training data), Plan E (knockout-guided feature ID design)
- **Stage 3**: Plan F (integration experiment combining all fixes)

## Important File Paths
- Configs: `configs/sae_layer0_attn_out.yaml`, `configs/knockout_sae/knockout_llava15_7b_color.yaml`
- Knockout results: `output/knockout_sae/knockout_color_run1_20260219_180829/knockout/knockout_summary.json`
- SAE outputs: `output/sae_experiments/`
- Main dataset: `datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv`
- Dataset CSVs: `datasets/by_attribute_category/ChooseAttr_{color,...}.csv`
- Images: `datasets/images/`
