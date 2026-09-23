# LLM Technical Summary: SAE Experiment Runs

Last updated: 2026-08-06. Supersedes the 2026-02-10 version, whose headline conclusion
("SAE feature ablation signal remains weak and usually indistinguishable from random
controls") was overturned by the v2 causal method in May 2026. **Do not trust any
document in this repo dated before 2026-05 on the question of whether SAE ablation works.**

## Project phase

The project is in **writeup**. This file exists so a reader — human or model — can reconstruct
what was run, what it showed, and which numbers are safe to cite, without loading the raw
result files.

Two experiments ran after the writeup phase began: the layer-14 ablation (2026-08-06) and the
multi-layer ablation program (2026-08-07, 47 conditions). Both are folded in below.

Numbers were recomputed from the on-disk results on 2026-08-06, and the multi-layer section on
2026-08-07 — not copied from earlier prose.

## The two techniques

1. **Attention knockout** — block an attention flow at one layer, measure the drop in
   `margin = log P(true option) − log P(false option)`. Identifies which layers carry
   causal image→text information.
2. **SAE feature ablation** — train a sparse autoencoder at those layers, identify causally
   important features by gradient attribution, zero them, measure the same margin drop.
   Tests whether the flow is mediated by *sparse interpretable features*.

The knockout drop is the natural ceiling for the ablation drop at the same layer: knockout
removes the whole flow, ablation removes only what the selected features carry.

## Headline result 1 — CLEVR-Lite knockout sweep

`Image->Question` flow, n=7084 per layer, 33h run, all p < 1e-10.

| Layer | mean margin_drop | Cohen's d |
|---|---|---|
| 0 | 0.4575 | 1.042 |
| 14 | 0.3983 | 1.205 |
| 11 | 0.4273 | 0.915 |
| 10 | 0.2683 | 1.135 |
| 12 | 0.2452 | 0.764 |
| 13 | **−0.0483** | **−0.364** |
| 29 | **−0.0170** | **−1.218** |

Three-region structure: a strong early site (layer 0), a mid cluster (10–14), and late-layer
inhibition (26, 29). **Negative values are real signal, not bugs** — blocking those layers
slightly *improves* accuracy. Layer 13 is inhibitory while flanked by positive layers 12 and 14.

Full 64-row table (both flows, all 32 layers):
`output/sae_experiments/exp_default/knockout/knockout_summary.json`.

## Headline result 2 — v2 causal feature ablation

Feature identification ran on the full validation set (n=7790); ablation on a 256-sample
subsample. `attn_out` site, `question` positions, `replace` mode, top-k features vs. 15
random control sets.

> **All z-scores in this table are inflated — read the control caveat below before citing them.**

| Layer | ablation margin_drop | random mean (sd) | z | % positive | flips | % of knockout ceiling |
|---|---|---|---|---|---|---|
| 0 | 0.0200 | −0.00028 (0.00028) | 73.1 | 56.6% | 0 | **4.4%** |
| 10 | 0.1514 | 0.00004 (0.00228) | 66.5 | 81.2% | 3 | 56.4% |
| 11 | 0.2131 | −0.00040 (0.00262) | 81.6 | 84.8% | 4 | 49.9% |
| 12 | 0.1670 | 0.00059 (0.00297) | 56.0 | 82.8% | 3 | 68.1% |
| 13 | 0.1220 | 0.00023 (0.00255) | 47.7 | 64.8% | 3 | — |
| **14** | **0.2433** | 0.00076 (0.00347) | 70.0 | **90.2%** | 2 | **61.1%** |

**Layer 14 is the strongest result, not layer 11** (run 2026-08-06; this file previously said
"not run"). It has the largest ablation drop, the highest fraction of positive samples, the
largest knockout effect size in the whole sweep (d = 1.205), and a healthy SAE
(`dead_feature_fraction` 0.0016). Layer 11 retains the highest z (81.6 vs 70.0) only because
layer 14's control sets are noisier (sd 0.00347 vs 0.00262); with 15 control sets that sd is
itself a noisy estimate, so the z gap is not a meaningful difference.

### Control caveat — the random controls were uniform, not matched

Discovered 2026-08-07. The configs request `random_sampling: "matched"` with
`matched_metric: "correct_mean"`, but v2 stats files carry only `causal_score`,
`activation_mean` and `gradient_mean`. `_extract_metric_value` falls back through
`correct_mean → ratio → diff → incorrect_mean`, finds none, returns `None`, and
`_sample_matched_random_features` takes its uniform branch. Confirmed empirically: 0 of the 45
stored control features fall in the causal top-500 (matched sampling predicts nearly all;
uniform predicts ~0.7).

At layer 11 the controls therefore have median activation **6.1e-08** against the binding set's
**0.117** — the same near-dead "ghost features" this project diagnosed in v1 *selection*,
surviving in the *control* arm. **Every z above is inflated by an unknown amount.** The
direction and per-sample consistency of the results (84–90% of samples positive) are unaffected;
only the magnitude of the separation from controls is in question.

Two further reporting problems in the same table: with 15 control sets the empirical p is floored
at 1/16 = 0.0625, and a z estimated from 15 draws has standard error ≈ z/√(2(n−1)), so `81.6`
cannot carry three significant figures.

Re-runs with `matched_metric: "activation_mean"` and `strict_matching: true` will be written to
`ablation_matched_controls.json` beside each existing result file. Expect every z to fall.

**Correction to earlier prose:** the "39% of knockout ceiling" figure in older documents
compared the CLEVR-Lite ablation (0.213) against the *GQA* knockout drop (0.540). Against
CLEVR-Lite's own layer-11 knockout drop (0.4273) the correct figure is **49.9%**.

### Why the ablation captures only half the ceiling — answered 2026-08-07

The standing explanation was that single-layer ablation leaves the model free to re-read the
image at layers L+1…31. **That is now tested and false.** Full record:
`docs/multilayer_ablation_findings.md`; 47 conditions in
`output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question/`.

**The direct test.** Ablate at L, then knock out `Image->Question` at *every* downstream layer
so the model cannot re-read the image at all. If re-reading were the mechanism, the same
ablation should bite far harder. It does not:

| anchor | ablation | downstream knockout | additive | actual combined | excess |
|---|---|---|---|---|---|
| 11 | 0.2131 | 0.9110 (L12–31) | 1.1241 | 1.2068 | +0.0828 |
| 14 | 0.2432 | 0.3533 (L15–31) | 0.5964 | 0.5989 | **+0.0024** |

**What is true instead — the features are distributed across layers.** At an equal budget of
200 features, spreading 40 across each of layers 10–14 gives a margin drop of **0.6182**
against **0.2131** for 200 concentrated at layer 11 (**2.90×**) and 0.1940 for 800 at layer 11
(**3.19×**) — while perturbing the residual stream *less* (0.0100 vs 0.0186). The concentrated
count curve is flat (0.190 / 0.192 / 0.213 / 0.236 / 0.194 for k = 40/100/200/400/800): a
single layer has a hard ceiling near 0.24 that no number of features breaks.

**And the recovered share does not trend with span.** Joint ablation over layers 10–14 gives
1.0028 against a span-knockout ceiling of 1.3934 measured on the same 256 samples — R = **72.0%**
(95% CI 68.9–75.3); pooled over spans of 1–5 layers, **72.6%** (95% CI 69.6–75.8). Across span
sizes 1–5 neither curve saturates; both are linear, ablation 0.193/layer against knockout
0.269/layer, ratio 0.718.

**Do not call R constant.** That slope ratio is an aggregate; per span R runs 65.1–87.5% with no
value inside all five paired-bootstrap intervals (highest lower bound 81.5% vs lowest upper bound
68.5%, spread 22.5 points, CI 17.7–28.4). What holds is the *direction*: R shows no trend with
span size (**−0.010/layer, 95% CI −0.024 to +0.003**), where a redundancy account predicts it
should rise as the span covers more of the compensating layers. Span 2's 83.9% is a denominator
artefact — inhibitory layer 13 lowers the ceiling. The residual third still points at the feature
method — incomplete feature sets, SAE reconstruction, position selection — rather than at
compensation by other layers. Regenerate with `tools/analyze_multilayer_ablation.py`
(`redundancy_by_span`, `redundancy_trend`).

Note R = 72.0% sits *inside* the single-layer range (49.9–78.2%): layer 12 alone recovers
78.2%. And leave-one-out runs the wrong way for redundancy at layers 10–12, whose in-context
marginals are 1.14×, 1.57× and 1.79× their standalone effects.

**Methodological consequence worth stating in the writeup: any single-layer SAE analysis
systematically understates the circuit.**

## Two results that need care in the writeup

**Layer 0 is an outlier and probably a methodological artifact.** It has the largest knockout
effect (0.4575) but the smallest ablation effect (0.0200, 4.4% of ceiling). Its SAE is the
reason: `dead_feature_fraction = 0.742` — 74% of the 32768 features never activate, against
0.06–0.4% at every other layer. The layer-0 dictionary largely failed to train. Do not read
"visual information at layer 0 is not sparsely encoded" from this number; the honest claim is
that this SAE could not test it.

**Layer 13 is a genuine puzzle.** Knockout there is *inhibitory* (−0.0483, d = −0.364): blocking
the flow slightly improves accuracy. Yet ablating its causally-identified features *hurts*
(margin_drop 0.1220, z = 47.7). Blocking the whole flow and removing these specific features
push in opposite directions, so the "% of ceiling" framing is undefined here. Worth a sentence
in the writeup rather than being quietly dropped from the table.

## Why v1 produced 18 consecutive nulls

v1 (`feature_analysis/feature_identifier.py`) scored features by ratio of mean activation on
correct vs. incorrect samples. That metric is maximised by features that barely activate at
all — "ghost features" with ~1e-5 magnitudes — so ablating them changed nothing.

The distilled catalogs quantify this directly. At layer 11, the top-500 features by causal
score have mean activation **0.117**, against a median across all 32768 features of
**6.1e-08** — roughly six orders of magnitude apart. There is **zero overlap** between the v1
and v2 top-200 sets.

v2 (`feature_analysis/causal_feature_identifier.py`) instead inserts the SAE into the forward
pass, backprops the target to feature activations, and scores `|grad| × |activation|`. Grounded
in Marks et al. 2024 (Sparse Feature Circuits) and Agrawal et al. 2025.

## Feature sparsity

At layer 11 the causal score is concentrated: top-50 features carry 17.2% of total score mass,
top-200 carry 45.7%, top-500 carry 79.9% (27,810 of 32,768 features are nonzero). This
concentration is the quantitative form of the paper's central claim and is available per-layer
in each `causal_feature_stats.summary.json`.

## SAE training quality

All six SAEs: 32768 features, `attn_out`, `question` positions, 4,898,438 training rows,
l1_coeff 5e-4, 10 epochs.

| Layer | explained variance | mean L0 | dead fraction |
|---|---|---|---|
| 0 | 0.99983 | 1416 | **0.742** |
| 10 | 0.99888 | 1128 | 0.0020 |
| 11 | 0.99851 | 1180 | 0.0038 |
| 12 | 0.99936 | 1587 | 0.0006 |
| 13 | 0.99859 | 1256 | 0.0015 |
| 14 | 0.99870 | 1446 | 0.0016 |

Caveat worth stating in the writeup: **mean L0 of 1100–1600 active features is high** for an
SAE of this size. Reconstruction is near-perfect but the dictionaries are not very sparse, which
weakens any claim that individual features are cleanly interpretable units.

## Trust levels

**Cite freely (current methodology, CLEVR-Lite, v2 causal):**
- `sae_clevr_lite_layer{0,10,11,12,13,14}_attn_out_question` — SAE training
- `sae_clevr_lite_layer{0,10,11,12,13,14}_attn_out_question_causal` — feature ID + ablation
  (**margin_drop and % positive are solid; the z-scores are inflated — see the control caveat**)
- `exp_default/knockout/` — the n=7084 knockout sweep

**Reference only (superseded v1 methodology, GQA, ratio-based features):**
- `exp_run1`, `exp_run2_all_residual`, `exp_run3_attr_residual_weak`, `exp_run4`
- `sae_q_layer0`, `sae_q_layer11`, `first_pass_*`, `layer0_*`, `layer11_*`
- `sweeps/sae_grid_v1_*`
- All of these are null results caused by ghost-feature selection. They are evidence about
  the *method*, not about the model.

**Incomplete:**
- `rerun_layer11_attn_out_20260209_211336` — config only
- `sweeps/...modereplace_delta_scale2.0` — ablation interrupted

**Structural irregularity:** layer 10's results are nested one level deeper
(`results/ablation_v2/ablation_v2_results.json`) than every other layer, and layer 11 has both
an n=50 run (`ablation_v2_results.json`) and the n=256 headline run (`ablation_results.json`).
Read filenames carefully; the n=256 file is the one to cite.

## Canonical files to load first

For the headline claim:
- `sae_clevr_lite_layer11_attn_out_question_causal/results/ablation_results.summary.json`
- `sae_clevr_lite_layer11_attn_out_question_causal/causal_feature_stats.summary.json`
- `sae_clevr_lite_layer11_attn_out_question_causal/causal_summary.json`
- `exp_default/knockout/knockout_summary.json`

For cross-layer comparison, the same two `*.summary.json` files under each
`sae_clevr_lite_layer{0,10,12,13}_attn_out_question_causal/`.

Writeup figures: `output/sae_experiments/report_assets/` and
`output/knockout_sae/knockout_color_run1_20260219_180829/knockout/plots/`.

## Distilled vs. raw data

Large result files are **not in git**. Each has a committed `*.summary.json` sibling holding
the top-500 features, the full score distribution, and aggregate statistics — everything the
writeup needs, at ~4% of the size. The originals remain on the author's disk and are
regenerable by re-running the pipeline.

| Raw file (gitignored) | Committed summary |
|---|---|
| `causal_feature_stats.json` (4.5 MB × 6) | `causal_feature_stats.summary.json` (~76 KB) |
| `feature_stats.json` | `feature_stats.summary.json` |
| `ablation*results*.json` | `ablation*results*.summary.json` |
| `knockout_results.json` (per-sample) | `knockout_summary.json` (already existed) |
| `feature_<N>.png` (220 files, 247 MB) | — dropped; v1-era ghost features |

Regenerate with `sae_experiments/tools/distill_results.py --root output`.

Note the summaries carry two fields the raw runner never stored: per-sample `margin_drop`
(derived from `baseline_margin − ablated_margin`) and `*_prediction_changes` (correct→wrong
and wrong→correct flip counts). See `docs/CLAUDE.md` for the research log and
`CLAUDE.md` § "Artifacts not in git" for what else is absent from a fresh clone.
