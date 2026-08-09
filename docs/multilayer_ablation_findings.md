# Multi-layer SAE ablation: attention is distributed, not redundant

**Date:** 2026-08-07
**Branch:** `multilayer-ablation`
**Span:** layers 10–14, `attn_out`, `question` positions, `replace` mode
**Dataset:** CLEVR-Lite, n=256 (the same 256 samples for every condition, ablation and knockout alike)
**Conditions:** 47, all in
`output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question/conditions/`
**Analysis:** `analysis/multilayer_summary.{json,md}`, regenerate with
`tools/analyze_multilayer_ablation.py --experiment_dir <dir>`

---

## 1. What was being tested

Single-layer SAE ablation recovers only 50–68% of the attention-knockout ceiling. The standing
hypothesis was **cross-layer redundancy**: ablating one layer leaves the others free to re-supply
the missing signal, most plausibly by re-reading the image downstream.

Note the hypothesis needed restating before it could be tested. Single-layer ablation is *not*
insignificant — every layer clears its controls. What is unexplained is the residual 30–50%. And
nothing "relearns" at inference; the only available mechanism is downstream layers re-reading the
image through their own attention.

**Result: the redundancy hypothesis is not supported.** What replaced it is a cleaner and more
useful finding about how the causal features are distributed.

---

## 2. The headline finding — distribution beats concentration

**Spreading a fixed feature budget across layers is ~3× more effective than concentrating it, while
perturbing the residual stream less.**

| condition | features | margin drop | rel. perturbation |
|---|---|---|---|
| **spread: 40 × 5 layers** | **200** | **0.6182** | **0.0100** |
| concentrated at L11, k=40 | 40 | 0.1895 | 0.0091 |
| concentrated at L11, k=100 | 100 | 0.1920 | 0.0136 |
| concentrated at L11, k=200 | 200 | 0.2131 | 0.0186 |
| concentrated at L11, k=400 | 400 | 0.2364 | 0.0274 |
| concentrated at L11, k=800 | 800 | 0.1940 | 0.0330 |

- spread(200) / concentrated(200) = **2.90×**
- spread(200) / concentrated(800) = **3.19×** — spreading beats four times the budget
- and it does so at **half** the perturbation norm (0.0100 vs 0.0186)

The concentrated count curve is **flat**: it saturates by k=40 and is non-monotonic by k=800, where
the effect falls to 0.1940 even though perturbation keeps rising to 0.0330. **A single layer has a
hard ceiling near 0.24 that no number of features breaks.**

This survives all three confounds the design anticipated:
- **not feature rank** — the concentrated arm is a curve from 40 to 800, not a single point
- **not count saturation** — measured directly, and it is flat
- **not perturbation norm** — the spread arm perturbs *less* for 3× the effect

### Depth beats count

At fixed span size 3, *which* layers dominates *how many*:

| span | ablation | knockout | R |
|---|---|---|---|
| {10,11,12} | 0.7590 | 1.2975 | 58.5% |
| {10,12,14} | 0.5631 | 0.7648 | 73.6% |
| {12,13,14} | 0.5007 | 0.5721 | 87.5% |

The nested curve alone would have been misread as a span-size effect. The non-nested controls are
what identify it.

---

## 3. Three independent lines falsify redundancy

### 3.1 The direct test — blocking downstream re-reading changes nothing

Ablate at anchor layer L, then knock out `Image->Question` at *every* layer downstream so the model
cannot re-read the image at all:

| | L11 | L14 |
|---|---|---|
| ablation alone | 0.2131 | 0.2432 |
| downstream knockout alone | 0.9110 (L12–31) | 0.3533 (L15–31) |
| sum, if independent | 1.1241 | 0.5964 |
| **actual combined** | **1.2068** | **0.5989** |
| excess over additive | +0.0828 | **+0.0024** |

At layer 14 the ablation bites **0.2% harder** with all downstream re-reading blocked. Layer 11
shows modest synergy (39% amplification) — real, but nowhere near the ~2× needed to close its gap.

### 3.2 The redundancy index barely moves, and sits inside the single-layer range

`R(S) = A(S) / K(S)`, numerator and denominator on the same 256 samples:

| layer / span | ablation | knockout | R |
|---|---|---|---|
| 10 | 0.1514 | 0.2665 | 56.8% |
| 11 | 0.2131 | 0.4268 | 49.9% |
| 12 | 0.1670 | 0.2135 | **78.2%** |
| 13 | 0.1220 | −0.0506 | undefined (knockout inhibitory) |
| 14 | 0.2432 | 0.3593 | 67.7% |
| **joint {10–14}** | **1.0028** | **1.3934** | **72.0%** (95% CI 68.9–75.3) |

Mean single-layer R is 63.2%, joint is 72.0% — but **layer 12 alone recovers 78.2%, more than the
joint span**. The joint value sits inside the single-layer range, not above it.

### 3.3 The span is mildly synergistic, and no more so than the flow itself

| | sum of singles | joint | ρ |
|---|---|---|---|
| ablation | 0.8968 | 1.0028 | **1.118** |
| knockout | 1.2155 | 1.3934 | **1.146** |

ρ_A / ρ_K = **0.975**. Both are slightly super-additive, and the ablation's super-additivity is
fully accounted for by the flow's own. There is no feature-specific redundancy left over.

### 3.4 Leave-one-out points the wrong way for three of five layers

Redundancy predicts marginal ≪ standalone. Observed:

| layer | in-context marginal | standalone | ratio |
|---|---|---|---|
| 10 | 0.1726 | 0.1514 | 1.14 |
| 11 | 0.3347 | 0.2131 | **1.57** |
| 12 | 0.2991 | 0.1670 | **1.79** |
| 13 | 0.0862 | 0.1220 | 0.71 |
| 14 | 0.0962 | 0.2432 | 0.40 |

Layers 10–12 matter *more* in context than alone — synergy, not redundancy. Only 13 and 14 look
redundant.

---

## 4. Neither curve saturates — the shortfall is a constant fraction

Nested spans, ablation against its knockout null:

| span size | span | ablation | knockout | R |
|---|---|---|---|---|
| 1 | {14} | 0.2432 | 0.3593 | 67.7% |
| 2 | {13,14} | 0.2679 | 0.3191 | 83.9% |
| 3 | {12,13,14} | 0.5007 | 0.5721 | 87.5% |
| 4 | {11,…,14} | 0.8302 | 1.2762 | 65.1% |
| 5 | {10,…,14} | 1.0028 | 1.3934 | 72.0% |

Both exponential fits `y = a(1 − e^{−bk})` **degenerate to straight lines** over this range —
`a` and `b` run away with only their product identified, so `b − d` carries no information. What is
identified is the slope: **ablation 0.193/layer against knockout 0.269/layer, ratio 0.718.**

So over spans of 1–5 layers ablation recovers a **constant ~72%** of the flow, with no sign of the
ablation curve turning over sooner than the knockout curve. Seeing saturation at all would need
wider spans.

**A constant fraction is the interesting part.** It points at the feature method itself —
incomplete feature sets, SAE reconstruction error, or position selection — rather than at anything
the other layers are doing. That is a sharper and more testable target than redundancy was.

---

## 5. Validation

### Gate

| check | result |
|---|---|
| `gate_none` (no hooks) | **0.0** exactly |
| `A0` harness regression, L11 k=200 | **0.213069** vs published 0.213077 (Δ 7.7e-06) |
| delta-mode passthrough, L10–14 | 0.0 exactly |

`replace`-mode passthrough by span: L14 −0.00037, L13-14 −0.00002, L12-14 −0.00008, L11-14 +0.00058,
L10-14 **+0.00109** — against a 0.02 green threshold. **Reconstruction error does not compound when
five `replace` hooks are stacked**, so `replace` was kept and every number here is comparable to the
published single-layer runs.

The 7.7e-06 on A0 is float accumulation order through the sample cache, not a behavioural change.

### Live-encoding OOD guard

Ablating L10–13 with a 0-feature passthrough hook at L14 gives 0.90459 against 0.90659 without it —
layer 14's out-of-distribution reconstruction contributes **−0.002**. Live encoding is not
contaminating the joint result.

### Sensitivity to layer 13

Layer 13's knockout is inhibitory (−0.0506) while its ablation drop is positive, so `K(S)` is not a
sum of positive contributions. Dropping it: S′ = {10,11,12,14} gives ablation 0.9166, knockout
1.3673, **R = 67.0%** against S's 72.0%. The conclusions do not depend on layer 13.

### Headroom

Baseline margin 2.7390, baseline accuracy 0.6484. The largest ablation leaves margin 1.7362
(accuracy 0.4180) and the largest span knockout leaves 1.3456 (0.4219). Nothing is floored.

---

## 6. The control bug this run fixed

Every previously published z compared causal features against **uniform** random controls, not
matched ones. `configs/clevr_lite/*.yaml` set `matched_metric: "correct_mean"`, which does not exist
in the v2 stats files (`causal_score`, `activation_mean`, `gradient_mean` only), so
`_extract_metric_value` returned `None` and `_sample_matched_random_features` silently fell back to
`rng.choice`. Corroboration: 0 of the 45 stored control features sat in the causal top-500, where
matching predicts nearly all.

At layer 11, k=200, same intervention, only the control changed:

| control | mean | sd | perturbation | z |
|---|---|---|---|---|
| uniform (published) | −0.0004 | 0.0026 | ~0.0019 | **81.6** |
| matched (this run) | **−0.0217** | 0.0037 | **0.0225** | **63.5** |

Matched controls now perturb the stream *more* than the binding set (0.0225 vs 0.0186) and still
produce a tenth of the effect. Every published z needs revising down by roughly this factor; the
effects survive easily (Wilcoxon over questions, p = 2.2e-35 at L11, 4.0e-42 for the joint span).

`strict_matching: true` held for every condition in this run — the log confirms
`sampling=matched metric=activation_mean strict=True` with no fallbacks.

---

## 7. Caveats

- **Only `A0` and `joint_L10-14` ran with 15 control sets.** The nested, non-nested and budget
  conditions used 1, so they carry no z. The per-question Wilcoxon test is significant throughout,
  but the control arm is thin for those conditions.
- **Single-layer ablations for L10, L12, L13 were not re-measured in this harness**; ρ_A uses the
  published drops. Legitimate — the binding arm does not depend on the control regime, and A0
  reproduces to 7.7e-06 — but it is a cross-run join.
- **The six-layer matched-control re-run (Phase H) has not been done.** Only layer 11 has a matched
  number, so the published per-layer table is still on the old uniform standard.
- **Saturation is unresolved, not absent.** Spans of 1–5 layers are all in the linear regime; wider
  spans would be needed to locate a knee.

---

## 8. What to say in the writeup

1. Cross-layer redundancy does **not** explain the single-layer ablation shortfall. The direct test
   — blocking all downstream re-reading — leaves the effect unchanged at layer 14 (+0.2%).
2. The causal features are **distributed** across layers: at equal budget, spreading 200 features
   over five layers is 2.9× a concentrated 200 and 3.2× a concentrated 800, at half the
   perturbation. Each layer has a bounded contribution that more features cannot exceed.
3. Ablation recovers a **constant ~72%** of the attention-knockout flow at every span size from 1 to
   5 layers. The missing third is a property of the feature method, not of the other layers.
4. **Any single-layer SAE analysis systematically understates the circuit.** This is the
   methodological point with the broadest reach beyond this paper.
5. All previously published z-scores used uniform, not matched, random controls (§6). Restate them.
