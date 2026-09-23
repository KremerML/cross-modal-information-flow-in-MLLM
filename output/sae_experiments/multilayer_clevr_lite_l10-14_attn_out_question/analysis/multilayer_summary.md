# Multi-layer ablation summary

Span: [10, 11, 12, 13, 14]. 47 conditions.

## Redundancy index R(S) = A(S) / K(S)

**R(S) = 0.720** (95% CI 0.689 - 0.753), from A = 1.0028 and K = 1.3934 over n = 256.

Single layers, for comparison (same samples for numerator and denominator):

| layer | ablation | knockout | ratio | 95% CI |
|---|---|---|---|---|
| 10 | — | — | undefined | missing_condition |
| 11 | — | — | undefined | missing_condition |
| 12 | — | — | undefined | missing_condition |
| 13 | — | — | undefined | missing_condition |
| 14 | 0.2432 | 0.3593 | 0.677 | 0.633 - 0.722 |

## Redundancy by span

| kind | span | size | A | K | R | 95% CI |
|---|---|---|---|---|---|---|
| nested | `{14}` | 1 | 0.2432 | 0.3593 | 67.7% | 63.3% - 72.2% |
| nested | `{13,14}` | 2 | 0.2679 | 0.3191 | 83.9% | 77.9% - 90.0% |
| nested | `{12,13,14}` | 3 | 0.5007 | 0.5721 | 87.5% | 81.5% - 94.4% |
| nested | `{11-14}` | 4 | 0.8302 | 1.2762 | 65.1% | 61.9% - 68.5% |
| nested | `{10-14}` | 5 | 1.0028 | 1.3934 | 72.0% | 68.9% - 75.3% |
| single | `{11}` | 1 | 0.2131 | 0.4268 | 49.9% | 44.8% - 55.3% |
| non-nested | `{10,11,12}` | 3 | 0.7590 | 1.2975 | 58.5% | 55.2% - 62.1% |
| non-nested | `{10,12,14}` | 3 | 0.5631 | 0.7648 | 73.6% | 69.9% - 77.5% |
| sensitivity | `{10,11,12,14}` | 4 | 0.9166 | 1.3673 | 67.0% | 64.0% - 70.4% |

Pooled recovery over the nested spans is 72.6% (95% CI 69.6% - 75.8%). The per-span ratios run 65.1% to 87.5% with NO value inside every interval, so R is not constant across spans and must not be reported as such. R shows no trend with span size (-0.0103 per layer, 95% CI -0.0236 - +0.0033, containing zero). A redundancy account predicts R rises as the span covers more of the layers said to compensate; it does not rise.

Observed spread across the nested spans is 22.5% (95% CI 17.7% - 28.4%); highest lower bound 81.5% against lowest upper bound 68.5%. 10000 draws, seed 42, n = 256.

## Comparative saturation

NEITHER curve saturates over this range -- both fits collapsed to straight lines (ablation 0.193/layer, knockout 0.269/layer), so b and d are not identified and b - d carries no information. What is identified is the slope ratio, 0.718: over spans of 1-5 layers the two curves grow in a fixed proportion, with no sign of the ablation curve turning over sooner. Wider spans would be needed to see saturation at all. This is a statement about the two slopes and NOT about R span by span -- see redundancy_by_span, where the ratios disperse beyond their own intervals.

## Interaction index

Not computed: this run contains no standalone ablation for layer(s) 10, 12, 13. Summing only the layers present would understate the denominator and inflate rho_A, so it is withheld. Supply the missing single-layer conditions, or compute rho_A against the published single-layer drops by hand -- the binding arm does not depend on the control regime, so those remain comparable.

## Leave-one-out

| layer | joint | without | marginal | standalone | marginal/standalone |
|---|---|---|---|---|---|
| 10 | 1.0028 | 0.8302 | +0.1726 | — | — |
| 11 | 1.0028 | 0.6681 | +0.3347 | 0.2131 | 1.571 |
| 12 | 1.0028 | 0.7037 | +0.2991 | — | — |
| 13 | 1.0028 | 0.9166 | +0.0862 | — | — |
| 14 | 1.0028 | 0.9066 | +0.0962 | 0.2432 | 0.396 |

A marginal contribution far below the standalone effect at every layer is the strong redundancy signature: the rest of the span already carries what that layer contributes.

## Significance

| condition | binding | z | z SE | control sets | p floor | Wilcoxon p | control regime |
|---|---|---|---|---|---|---|---|
| A0_regression_L11 | 0.2131 | 64 | 12 | 15 | 0.0625 | 2.23e-35 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| budget_spread40x5 | 0.6182 | -- | -- | 1 | 0.5000 | 5.06e-35 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| joint_L10-14 | 1.0028 | 111 | 21 | 15 | 0.0625 | 3.95e-42 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L11-14 | 0.8302 | -- | -- | 1 | 0.5000 | 3.25e-42 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L12-14 | 0.5007 | -- | -- | 1 | 0.5000 | 1.87e-38 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L13-14 | 0.2679 | -- | -- | 1 | 0.5000 | 1.14e-33 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nested_L14 | 0.2432 | -- | -- | 1 | 0.5000 | 1.72e-37 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nonnested_L10,12,14 | 0.5631 | -- | -- | 1 | 0.5000 | 1.87e-39 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |
| nonnested_L10-12 | 0.7590 | -- | -- | 1 | 0.5000 | 3.26e-41 | {'10': 'matched_activation_mean', '11': 'matched_activation_mean', '12': 'matched_activation_mean', '13': 'matched_activation_mean', '14': 'matched_activation_mean'} |

z is reported to one significant figure: it is estimated from a handful of control draws and its own standard error is large. The empirical p cannot go below its floor, so the Wilcoxon test over questions is the one with power.
