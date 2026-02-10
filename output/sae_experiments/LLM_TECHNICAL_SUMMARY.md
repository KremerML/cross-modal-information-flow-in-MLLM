# LLM Technical Summary: SAE Experiment Runs (output/sae_experiments)
Last updated: 2026-02-10

## Scope
This summary compresses all **14 indexed experiment runs** under `output/sae_experiments` so another LLM can reason about setup changes, result quality, and what conclusions are reliable.

## Primary Research Context
- Base finding from attention knockout: image-to-question flow has strong layer-dependent causal effects.
- SAE extension goal: identify sparse features and show that ablating selected ("binding") features hurts performance more than matched random features.

## High-Confidence Global Takeaway
Across completed runs, the main pattern is:
1. **Knockout signal exists and is strong** (latest run).
2. **SAE feature ablation signal remains weak and usually indistinguishable from random controls**, including after switching from residual stream to attention output (`attn_out`), increasing top-k, and using matched multi-set random controls.

## Trust Levels for Runs
- **High trust (current methodology)**:
  - `first_pass_layer11_residual`
  - `rerun_layer11_attn_out_20260209_213255`
  - Layer-0 sweep completed runs (`...residual_delta1.0`, `...residual_delta2.0`, `...replace_delta1.0`)
- **Medium/low trust (older methodology)**:
  - `exp_run1`, `exp_run2_all_residual`, `exp_run3_attr_residual_weak`, `exp_run4`, `sae_q_layer0`, `sae_q_layer11`
  - Reasons: mostly single random control set, no multi-set empirical random baseline, older metric focus.
- **Incomplete / interrupted**:
  - `rerun_layer11_attn_out_20260209_211336` (config only)
  - `sweeps/...modereplace_delta_scale2.0` (ablation interrupted by KeyboardInterrupt)

## Canonical Files to Load First
- Latest strong knockout + attn_out SAE run:
  - `output/sae_experiments/rerun_layer11_attn_out_20260209_213255/knockout/knockout_summary.json`
  - `output/sae_experiments/rerun_layer11_attn_out_20260209_213255/results/ablation_results.json`
  - `output/sae_experiments/rerun_layer11_attn_out_20260209_213255/analysis/hypothesis_report.json`
  - `output/sae_experiments/rerun_layer11_attn_out_20260209_213255/config.yaml`
- Corrected residual comparison run:
  - `output/sae_experiments/first_pass_layer11_residual/results/ablation_results.json`
  - `output/sae_experiments/first_pass_layer11_residual/config.yaml`
- Layer-0 strength/mode sweep (completed):
  - `output/sae_experiments/sweeps/sae_grid_v1_layer0_train-attribute_feat-attribute_top_k50_min_activation0.0_moderesidual_delta_scale1.0/results/ablation_results.json`
  - `output/sae_experiments/sweeps/sae_grid_v1_layer0_train-attribute_feat-attribute_top_k50_min_activation0.0_moderesidual_delta_scale2.0/results/ablation_results.json`
  - `output/sae_experiments/sweeps/sae_grid_v1_layer0_train-attribute_feat-attribute_top_k50_min_activation0.0_modereplace_delta_scale1.0/results/ablation_results.json`

## Run Catalog (14 runs)
Legend:
- `B_acc_drop`: binding accuracy drop.
- `R_acc_drop`: random accuracy drop.
- `B-R acc`: binding minus random accuracy drop.
- `B_margin`: binding mean margin drop.
- `R_margin`: random mean margin drop.
- `B-R margin`: binding minus random margin drop.

| Run | Setup Summary | Status | Key Outcome |
|---|---|---|---|
| `exp_default` | residual(default), layer 12, knockout only | completed | Knockout summary exists but all effects are zero (stale/legacy behavior). |
| `exp_run1` | residual(default), layer 12, attr positions, mode=residual, delta=1.0, top_k=50 | completed | `B_acc_drop=0.0`, `R_acc_drop=0.0` (null). |
| `exp_run2_all_residual` | all positions, mode=residual, delta=1.0, top_k=50 | completed | `B_acc_drop=0.00114`, `R_acc_drop=0.00228`, old hypothesis report says significant; treat as low-trust legacy evidence. |
| `exp_run3_attr_residual_weak` | attr positions, mode=residual, delta=0.5, top_k=50 | completed | `B_acc_drop=0.00114`, `R_acc_drop=0.00114` (null). |
| `exp_run4` | attr positions, mode=residual, delta=2.0, top_k=50 | completed | Tiny effects; no robust separation. |
| `sae_q_layer0` | residual(default), fixed results file, top_k=50 | completed | `B_acc_drop=0.0`, `R_acc_drop=0.00114`; tiny margins, null interpretation. |
| `sae_q_layer11` | residual(default), fixed results file, top_k=50 | completed | `B_acc_drop=0.00114`, `R_acc_drop=0.00114`; null interpretation. |
| `sweeps/...residual_delta_scale1.0` | layer 0, attr-train/attr-feature, mode=residual, delta=1.0, top_k=50, min_act=0.0 | completed | `B_acc_drop=0.0`, `R_acc_drop=0.00114`, `B-R margin=-8.69e-05` (null). |
| `sweeps/...residual_delta_scale2.0` | same as above, delta=2.0 | completed | `B_acc_drop=0.0`, `R_acc_drop=0.0`, `B-R margin=-4.45e-04` (null). |
| `sweeps/...replace_delta_scale1.0` | same grid, mode=replace, delta=1.0 | completed | `B_acc_drop=0.00114`, `R_acc_drop=0.00114`, `B-R margin=-3.96e-05` (null). |
| `sweeps/...replace_delta_scale2.0` | same grid, mode=replace, delta=2.0 | incomplete | No `ablation_results.json`; run interrupted (KeyboardInterrupt in `run.log`). |
| `first_pass_layer11_residual` | residual, layer 11, attr positions, mode=replace, top_k=200, matched random sets | completed | `B_acc_drop=-0.00114`, `R_acc_drop=-0.00228`, `B_margin=0.00135`, `R_margin=0.00140`, empirical p: acc `0.0625`, margin `1.0` (null). |
| `rerun_layer11_attn_out_20260209_211336` | attn_out, layer 11, attr positions | incomplete | Config exists; no knockout/results artifacts. |
| `rerun_layer11_attn_out_20260209_213255` | attn_out, layer 11, attr positions, mode=replace, top_k=200, matched random sets, margin metric | completed | `B_acc_drop=0.001140`, `R_acc_drop=0.001216`, `B-R acc=-7.60e-05`; `B_margin=-0.002105`, `R_margin=-0.001967`, `B-R margin=-1.38e-04`; empirical p values both `1.0` (null). |

## Most Important Quantitative Snapshot (Latest attn_out run)
Run: `output/sae_experiments/rerun_layer11_attn_out_20260209_213255`

### Knockout (same run, strong signal)
- `Image->Question`, layer 11:
  - `mean_margin_drop = 0.168814`
  - `effect_size = 0.605366`
  - `p_value = 7.62e-57`

### SAE ablation (null vs random)
- Binding:
  - `accuracy_drop = 0.001140`
  - `mean_margin_drop = -0.002105`
  - `mean_relative_perturbation = 0.000303`
- Random (15 matched sets mean):
  - `accuracy_drop = 0.001216`
  - `mean_margin_drop = -0.001967`
  - `mean_relative_perturbation = 0.000303`
- Empirical p-values:
  - `accuracy_drop p = 1.0`
  - `margin_drop p = 1.0`
- Hypothesis report:
  - `hypothesis_supported = false`
  - `test_type = empirical_random_set`

## Method Evolution (What Changed Across Runs)
1. Residual-only early runs with limited controls.
2. Fixed indexing and updated evaluation outputs (`*_fixed.json`).
3. Layer-0 grid sweep varying intervention mode (`residual` vs `replace`) and strength (`delta_scale`).
4. Layer-11 focused runs with larger feature sets (`top_k=200`) and matched multi-set random controls.
5. Shift from residual target to attention-output target (`attn_out`) to align with knockout mechanism.

## Caveats and Data Hygiene Notes
1. Some old runs are missing `config.yaml`; reconstruct setup from result files where possible.
2. `exp_default` knockout summary appears non-informative (all zeros).
3. `sweeps/...replace_delta_scale2.0` is incomplete; do not include in aggregate efficacy conclusions.
4. Older runs often have `random_set_summaries=0`; direct statistical comparisons with newer empirical-random-set runs are not apples-to-apples.

## Bottom-Line Interpretation for Downstream LLM
Use this conclusion as prior:
- The pipeline repeatedly confirms **layer-level cross-modal dependence** via knockout.
- The current SAE feature selection + ablation interventions do **not** show stable binding-specific causal effects beyond matched random controls, even when targeting `attn_out` and using stronger controls.
