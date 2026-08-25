# Research log

Chronological record of what was run, what it showed, and why the pipeline looks the way it does.

Three files divide the work between them:

- **`CLAUDE.md`** (repo root) — architecture, commands, conventions, gotchas. How the code works.
- **`output/sae_experiments/LLM_TECHNICAL_SUMMARY.md`** — every verified result number, per-layer
  tables, trust levels per run. What the numbers are.
- **this file** — how the project got there, and which reasoning is still load-bearing.

Superseded reports live in `archive/docs/`. They are kept for traceability of the record, not as
descriptions of current state; several reach conclusions this log documents as overturned.

---

## Timeline

| When | What | Outcome |
|---|---|---|
| Feb–Mar 2026 | GQA ChooseAttr, v1 statistical feature ID | 18 consecutive null ablations |
| Mar 2026 | Null-result post-mortem | Five diagnosed causes, four addressed |
| Apr–May 2026 | CLEVR-Lite built; 32-layer knockout sweep (n=7084) | Three-region flow structure |
| May 2026 | v2 causal feature ID (gradient attribution) | Null streak broken |
| 2026-08-06 | Ablation completed at all six layers | Layer 14 strongest, not layer 11 |
| 2026-08-07 | Random-control sampling audited | Controls were uniform, not matched |
| 2026-08-07 | Multi-layer program, 47 conditions | Redundancy falsified; distribution found |

---

## Phase 1 — GQA ChooseAttr and the 18 nulls (Feb–Mar 2026)

The first generation ran attention knockout on GQA ChooseAttr (forced choice: "Is the car red or
blue?"), then trained SAEs at the two knockout peaks and ablated features selected by a statistical
criterion.

**Knockout, GQA era** (`Image->Question`, n=510 and n=810 runs):

| Layer | margin_drop | Cohen's d |
|---|---|---|
| 0 | 0.54 | 0.83 |
| 11 | 0.17 | 0.60 |
| 8 | 0.076 | 0.44 |
| 12 | 0.054 | 0.34 |
| 10 | 0.054 | 0.29 |

Two peaks — layer 0 and layer 11 — which is why those became the first SAE training sites.

**v1 feature identification** (`feature_analysis/feature_identifier.py`) scored features by the
ratio of mean activation on correct versus incorrect samples. Eighteen consecutive ablation
experiments found binding features indistinguishable from random controls.

### The post-mortem (Mar 2026) and what it produced

Five causes were diagnosed. Four drove concrete changes; the fifth turned out to be the real one.

1. **Position mismatch.** Ablation targeted `attribute` text positions, where full-latent ablation
   showed only 0.3% relative perturbation — nothing to remove. At `all` positions perturbation was
   ~100% and margin drops (0.314 at L0, 0.169 at L11) tracked the knockout values. → the `image`
   position type was implemented, and later `question` became the default once it was clear that
   `Image->Question` knockout changes *question*-token activations, not image-token ones. **The
   source is the image tokens; the destination is the question tokens, and the destination is where
   the intervention has to land.**

2. **Training data far too small.** ~6–8k activation vectors for a 32,768-feature SAE, giving ~44%
   dead features. → CLEVR-Lite was built at a scale that yields ~4.9M rows.

3. **Language-side bypass.** Both options appear in the ChooseAttr question text, so the model can
   partly answer from language priors. → CLEVR-Lite uses open-ended queries.

4. **SAE architecture behind current practice.** → `b_pre` encoder bias, decoder column
   normalisation after every optimiser step, dead-feature tracking per epoch, and a cosine LR
   schedule were all added to `training/sae_trainer.py`. Still open: no auxiliary dead-feature
   prevention (AuxK/TopK/JumpReLU), and the `mean(|z|)` L1 penalty weakens as `n_features` grows.

5. **The selection metric itself** — the actual cause, and the one the post-mortem underweighted.
   See Phase 3.

Also settled here: `replace` mode (`out = decode(feats_with_selected_zeroed)`) became the standard
intervention over the error-preserving `residual` delta. Every active config uses it.

---

## Phase 2 — CLEVR-Lite and the knockout sweep (Apr–May 2026)

CLEVR-Lite is synthetic: 6 colors × 3 shapes, PIL-rendered 224×224 PNGs, open-ended queries
("What color is the triangle?"). 186,638 train / 7,790 val questions from 50K scenes. A
`false option` is synthesised by deterministically sampling a distractor from the closed attribute
set, so open-ended questions still support the forced-choice `margin` metric. Generation is
deterministic from `datasets/clevr_lite/config.json` (seed 32).

Removing the language bypass mattered: effect sizes came out **25–50% larger than GQA** across the
board, and `Image->Last` went from max d≈0.20 to d=1.15.

**The sweep** — `Image->Question`, all 32 layers, n=7084, 33 hours, all p < 1e-10:

| Layer | margin_drop | Cohen's d |
|---|---|---|
| 0 | 0.4575 | 1.042 |
| 11 | 0.4273 | 0.915 |
| 14 | 0.3983 | **1.205** |
| 10 | 0.2683 | 1.135 |
| 12 | 0.2452 | 0.764 |
| 13 | **−0.0483** | **−0.364** |
| 29 | **−0.0170** | **−1.218** |

GQA's two-peak structure resolved into three regions: an early site (layer 0), a mid cluster
(10–14), and late-layer inhibition (26, 29). Negative values are real — blocking those layers
slightly *improves* accuracy. Layer 13 is inhibitory while flanked by positive layers 12 and 14.

This sweep, not the GQA one, is the current ceiling reference for every ablation number.

---

## Phase 3 — v2 causal feature identification (May 2026)

The ratio metric is maximised by features that barely activate at all. With
`incorrect_mean = 0` for 81% of selected features, `ratio ≈ correct_mean / 2e-8` — so v1 ranked
"ghost features" with ~1e-5 magnitudes to the top, and zeroing them changed nothing. At layer 11
the v2 top-500 features have mean activation **0.117** against a median of **6.1e-08** across all
32,768 features, roughly six orders of magnitude apart.

The deeper problem, from Agrawal et al. (2025, arXiv:2505.20063): correlational scoring selects
*input* features that detect patterns, not *output* features that drive predictions, and the two
rarely coincide.

**v2** (`feature_analysis/causal_feature_identifier.py`, pipeline stage 02) inserts the SAE into
the forward pass, backprops the target (`margin` or `correct_logit`) to feature activations, and
scores `|grad| × |activation|` averaged over samples. Grounded in Marks et al. 2024 (Sparse Feature
Circuits, arXiv:2403.19647) and Agrawal et al. 2025.

This broke the null streak on first run.

> Do not compare the v1 and v2 feature sets directly. The surviving v1 catalog
> (`output/sae_experiments/sae_q_layer11/`) is a different configuration despite its name —
> `target_layer: 12`, `position_type: attribute`, GQA, and 50 features rather than 200. An overlap
> statistic between it and the layer-11 v2 set is not measuring what it appears to measure.

v1 is retained in the tree for reference. The `feature_identification` config section
(`selection_method`, `score_key`, `discrimination_threshold`, …) belongs to it, not to v2.

---

## Phase 4 — ablation at all six layers (2026-08-06)

Feature ID on the full validation set (n=7790); ablation on 256 samples; `attn_out`, `question`
positions, `replace` mode, top-200 features against 15 random control sets.

| Layer | margin_drop | z | % positive | % of knockout ceiling |
|---|---|---|---|---|
| 0 | 0.0200 | 73.1 | 56.6% | 4.4% |
| 10 | 0.1514 | 66.5 | 81.2% | 56.4% |
| 11 | 0.2131 | 81.6 | 84.8% | 49.9% |
| 12 | 0.1670 | 56.0 | 82.8% | 68.1% |
| 13 | 0.1220 | 47.7 | 64.8% | undefined |
| **14** | **0.2433** | 70.0 | **90.2%** | **61.1%** |

**Layer 14 is the strongest result, not layer 11.** Largest drop, highest fraction of positive
samples, largest knockout effect size in the sweep (d = 1.205), healthy SAE. Layer 11's higher z
reflects quieter control sets, not a bigger effect.

Two results that need care in any writeup:

- **Layer 0 is a dictionary-training failure, not a negative result.** Its SAE has
  `dead_feature_fraction = 0.742` against 0.06–0.4% elsewhere. Its 4.4%-of-ceiling number says
  nothing about whether visual information at layer 0 is sparsely encoded.
- **Layer 13 is a genuine puzzle.** Knockout is inhibitory (−0.0483) but ablating its causal
  features *hurts* (0.1220). The two interventions disagree in sign, so "% of ceiling" is undefined
  there. Worth a sentence rather than a quiet omission.

Also worth stating: mean L0 across the six SAEs is 1100–1600 active features. Reconstruction is
near-perfect (explained variance > 0.998) but the dictionaries are not very sparse, which limits
claims about individual features as clean interpretable units.

---

## Phase 5 — the random controls were uniform, not matched (2026-08-07)

Configs request `random_sampling: "matched"` with `matched_metric: "correct_mean"`. That key exists
only in v1 stats files; v2 files carry `causal_score`, `activation_mean`, and `gradient_mean`.
`_extract_metric_value` falls back through `correct_mean → ratio → diff → incorrect_mean`, finds
none, returns `None`, and `_sample_matched_random_features` silently takes its uniform branch.

Confirmed empirically: 0 of the 45 stored control features fall in the causal top-500. Matched
sampling predicts nearly all; uniform predicts ~0.7.

So at layer 11 the controls have median activation 6.1e-08 against the binding set's 0.117 — the
same ghost features the project diagnosed in v1 *selection*, resurfacing in the *control* arm.
**Every z-score in Phase 4 is inflated by an unknown amount.** Margin drops and per-sample positive
fractions are unaffected; only the separation from controls is in question.

Two further reporting problems in the same table: with 15 control sets the empirical p is floored
at 1/16 = 0.0625, and a z estimated from 15 draws has standard error ≈ z/√(2(n−1)), so `81.6`
cannot carry three significant figures.

New work should set `matched_metric: "activation_mean"` and `random_control.strict_matching: true`,
which raises instead of falling back. The permissive default is retained so existing runs reproduce,
but it now warns loudly.

**Why the test suite never caught it:** the control sampler was tested for determinism and for
producing distinct sets, never for producing sets that resemble the binding set on any metric.

---

## Phase 6 — the multi-layer program (2026-08-07)

47 conditions over layers 10–14, n=256, the same samples for every condition. Full record in
`docs/multilayer_ablation_findings.md`.

**The question.** Single-layer ablation recovers only 50–68% of the knockout ceiling. The standing
explanation was cross-layer redundancy — ablate one layer and the others re-supply the signal by
re-reading the image downstream.

**The direct test.** Ablate at L, then knock out `Image->Question` at *every* downstream layer so
the model cannot re-read the image at all. If re-reading were the mechanism the ablation should
bite far harder. It does not:

| anchor | ablation | downstream knockout | additive | actual | excess |
|---|---|---|---|---|---|
| 11 | 0.2131 | 0.9110 (L12–31) | 1.1241 | 1.2068 | +0.0828 |
| 14 | 0.2432 | 0.3533 (L15–31) | 0.5964 | 0.5989 | **+0.0024** |

Three further lines agree: the redundancy index sits *inside* the single-layer range (joint
{10–14} R = 72.0% against layer 12's 78.2% alone); the span's super-additivity is fully accounted
for by the flow's own (ρ_A/ρ_K = 0.975); and leave-one-out runs the wrong way at layers 10–12,
whose in-context marginals are 1.14×, 1.57×, and 1.79× their standalone effects.

**What is true instead — the features are distributed across layers.** At an equal budget of 200
features, spreading 40 across each of layers 10–14 gives a margin drop of **0.6182** against
**0.2131** for 200 concentrated at layer 11 (**2.90×**), and beats even 800 concentrated features
(0.1940, **3.19×**) — while perturbing the residual stream *less* (0.0100 vs 0.0186). The
concentrated count curve is flat and non-monotonic (0.190 / 0.192 / 0.213 / 0.236 / 0.194 for
k = 40/100/200/400/800): **a single layer has a hard ceiling near 0.24 that no number of features
breaks.**

Depth beats count. At fixed span size 3, which layers matters more than how many: {10,11,12} gives
R = 58.5%, {10,12,14} 73.6%, {12,13,14} 87.5%.

**Do not call R constant.** The pooled 72.6% is a slope ratio (ablation 0.193/layer against
knockout 0.269/layer). Per span R runs 65.1–87.5% with no value inside all five paired-bootstrap
intervals. What holds is the *direction*: R shows no trend with span size (−0.010/layer, 95% CI
−0.024 to +0.003), where a redundancy account predicts it should rise as the span covers more of
the compensating layers.

**Consequence for the writeup: any single-layer SAE analysis systematically understates the
circuit.**

---

## Open questions

- **Where does the residual ~28% go?** Not to compensation by other layers — that is what Phase 6
  ruled out. The remaining candidates are properties of the feature method itself: incomplete
  feature sets, SAE reconstruction error, and position selection. Sharper and more testable than
  redundancy was.
- **Every z needs re-measuring** against matched controls (Phase 5). Expect all of them to fall.
- **Layer 0 needs a working dictionary** before anything can be concluded about it.
- **Layer 13's sign disagreement** between knockout and ablation is unexplained.
- **Do the SAEs need to be sparser?** Mean L0 of 1100–1600 is high enough to weaken per-feature
  interpretability claims, and no dead-feature prevention is in place.

---

## GQA ChooseAttr dataset quality (reference)

The GQA path is no longer active, but this analysis is the reason CLEVR-Lite exists and the reason
any future GQA work should use **color** only.

`datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv`, 1000 rows, 937 unique images.
Structurally clean: no duplicate question_ids, no missing values, `answer == true option`
throughout, option order balanced 501/499.

| Category | Rows | Runnable | Contents |
|---|---|---|---|
| color | 602 | yes | color only — clean and homogeneous |
| state | 151 | with caution | weather(18) + cleanliness(11) + state(5) + opaqness(1) + **116 generic "choose" items** |
| material | 96 | no (<100) | material only — clean |
| size | 87 | no (<100) | size(41) + length(22) + height(15) + depth/thickness/weight/width(9) |
| shape | 15 | no | critically too small; 9/15 cover three attribute pairs |

Quality concerns that survive into any GQA result: ~58% of color questions involve black or white
(a "color feature" may be learning achromatic vs. chromatic); 20.6% of color questions have objects
covering <1% of image area; 25/602 color false options are multi-word or uncommon ("cream colored",
"light brown") and tokenize differently under logprob scoring. The false option is never in the
object's own attribute list (verified 0/602 for color) — sound design, but it means the task always
requires recognising the true attribute rather than filtering a co-occurring one.

49 rows (pose, activity, sportActivity, face expression) are unassigned and correctly excluded —
they are not visual attribute-binding tasks.
