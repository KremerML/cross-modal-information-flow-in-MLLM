# Project Memory: cross-modal-information-flow-in-MLLM

Last synced: 2026-08-20. In-repo copy of the working memory that otherwise lives only on the
author's machine (`~/.claude/projects/.../memory/`), so the context travels with a clone.
Architecture and commands are in `CLAUDE.md` at the repo root; the research log is in
`docs/CLAUDE.md`; verified result numbers are in
`output/sae_experiments/LLM_TECHNICAL_SUMMARY.md`.

**Project phase: writeup.** The paper draft is `overleaf/main_final.tex` ("Calibrating Sparse
Autoencoder Ablation Against Attention Knockout in a Vision-Language Model", August 2026).
Remaining work is documentation, figures, and methodology description. Two experiment programs ran
after the writeup phase began — the layer-14 ablation and the 47-condition multi-layer matrix —
and both are folded in below.

## Research goal

Study cross-modal information flow in LLaVA-v1.5-7b (32 layers, d_model=4096) by combining
attention knockout with SAE feature ablation, measuring
`margin = log P(true option) − log P(false option)`.

## Key finding 1 — CLEVR-Lite knockout sweep

33-hour sweep, n=7084, baseline margin 3.178, all p < 1e-10.
Source: `output/sae_experiments/exp_default/knockout/knockout_summary.json`.

`Image->Question`, top layers by effect size:

| Layer | margin_drop | d | Note |
|---|---|---|---|
| 14 | 0.398 | 1.205 | strongest single-layer effect |
| 10 | 0.268 | 1.135 | mid-depth binding |
| 0 | 0.457 | 1.042 | early grounding |
| 11 | 0.427 | 0.915 | much stronger than on GQA |
| 12 | 0.245 | 0.764 | extends the 10–12 cluster |

Inhibitory layers (blocking *improves* accuracy — real signal, not bugs):
29 (−0.017, d=−1.218), 26 (−0.009, d=−0.707), 15 (−0.043, d=−0.416), 13 (−0.048, d=−0.364).

**Three-region structure:** layer 0 = immediate cross-modal registration; layers 10–12 = mid-depth
semantic binding (with layer 13 inhibitory *inside* the cluster); layer 14 = final visual-semantic
consolidation.

Versus GQA: effect sizes 25–50% larger, confirming that GQA ChooseAttr's language-side bypass was
diluting effects. `Image->Last` is also much stronger here (max d=1.15 at layer 2 vs ~0.20 on GQA).

## Key finding 2 — v2 causal feature ablation

**This broke an 18-experiment null streak (May 2026).** Feature ID on the full val set (n=7790);
ablation on 256 samples; `attn_out`, `question` positions, `replace` mode, 15 random control sets.

| Layer | ablation margin_drop | z | % positive | flips | % of knockout ceiling |
|---|---|---|---|---|---|
| 0 | 0.0200 | 73.1 | 56.6% | 0 | 4.4% |
| 10 | 0.1514 | 66.5 | 81.2% | 3 | 56.4% |
| 11 | 0.2131 | 81.6 | 84.8% | 4 | 49.9% |
| 12 | 0.1670 | 56.0 | 82.8% | 3 | 68.1% |
| 13 | 0.1220 | 47.7 | 64.8% | 3 | — |
| **14** | **0.2433** | 70.0 | **90.2%** | 2 | **61.1%** |

**Layer 14 is the strongest result, not layer 11:** largest drop, highest % positive, largest
knockout d (1.205), healthy SAE. Layer 11's higher z reflects quieter control sets, not a bigger
effect.

**v2 method:** insert the SAE into the forward pass, backprop the target to feature activations,
score `|grad| × |activation|`. Marks et al. 2024 (Sparse Feature Circuits), Agrawal et al. 2025.
Implementation: `feature_analysis/causal_feature_identifier.py`.

**Root cause of the prior nulls:** v1 scored features by correct/incorrect activation ratio, which
is maximised by features that barely activate — "ghost features". At layer 11 the v2 top-500 have
mean activation 0.117 against a median of 6.1e-08 across all 32768 features.

## Key finding 3 — distribution, not redundancy

47 conditions over layers 10–14, n=256, run 2026-08-07. Full record:
`docs/multilayer_ablation_findings.md`; results under
`output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question/`.

The standing explanation for why ablation captures only ~50–68% of the knockout ceiling was that
the model re-reads the image at layers L+1…31. **Tested directly and false** — ablating at L while
knocking out `Image->Question` at every downstream layer is additive (excess +0.0024 at layer 14).

**What is true instead:** at an equal budget of 200 features, spreading 40 across each of layers
10–14 gives 0.6182 against 0.2131 concentrated at layer 11 (**2.90×**), and beats 800 concentrated
features (**3.19×**) while perturbing the residual stream *less*. A single layer has a hard ceiling
near 0.24 that no feature count breaks.

**Methodological consequence for the writeup: any single-layer SAE analysis systematically
understates the circuit.**

## Caveats that must appear in the writeup

- **The random controls were uniform, not matched** (found 2026-08-07). Configs set
  `matched_metric: "correct_mean"`, absent from every v2 stats file, so the matched sampler
  silently fell back to uniform draws. At layer 11 the controls have median activation 6.1e-08
  against the binding set's 0.117. **Every z above is inflated.** Margin drops and per-sample
  positive fractions are unaffected. Use `matched_metric: "activation_mean"` plus
  `random_control.strict_matching: true` for new work.
- **Layer 0 is a dictionary-training failure, not a negative result.** Its SAE has
  `dead_feature_fraction = 0.742` (74% of features never fire) against 0.06–0.4% at every other
  layer. Its 4.4%-of-ceiling ablation says nothing about the model.
- **Layer 13 is a genuine puzzle.** Knockout there is inhibitory (−0.048) but ablating its causal
  features hurts (+0.122). The two interventions disagree in sign; "% of ceiling" is undefined.
  Worth a sentence rather than a quiet omission.
- **Mean L0 is 1100–1600** across all six SAEs. Reconstruction is near-perfect (explained variance
  > 0.998) but the dictionaries are not very sparse, which limits claims about individual features
  being clean interpretable units.
- **Do not call the redundancy index constant.** The pooled 72.6% is a slope ratio; per span R runs
  65.1–87.5% with no value inside all five bootstrap intervals. What holds is that R shows no
  *trend* with span size (−0.010/layer, 95% CI −0.024 to +0.003), which is what refutes redundancy.
- **Do not compare the v1 and v2 feature sets.** The surviving v1 catalog
  (`output/sae_experiments/sae_q_layer11/`) is a different configuration despite its name —
  `target_layer: 12`, `position_type: attribute`, GQA, 50 features. An overlap statistic against
  the layer-11 v2 set is not measuring what it appears to measure.

## CLEVR-Lite dataset

Synthetic: 6 colors × 3 shapes, PIL-rendered 224×224 PNGs, open-ended queries ("What color is the
triangle?") so there is no language-side bypass. 186,638 train / 7,790 val questions (50K scenes ×
~3.7 q/scene). Generated deterministically — `tools/generate_clevr_lite.py` with
`datasets/clevr_lite/config.json` (seed 32) reproduces it exactly.

A `false option` is synthesised by deterministically sampling a distractor from the closed
attribute set, so open-ended questions still support the forced-choice `margin` metric.

That config also records a compositional **held-out combination split** not documented elsewhere:
11 train combos against 7 held out (blue/red/purple/yellow circle, black triangle, black/red
square) at `held_out_ratio: 0.4`.

## GQA ChooseAttr dataset quality (reference only)

- **color** (602 rows): the only clean, homogeneous category — use it for any GQA work. ~58%
  involve black/white; 20.6% have objects covering <1% of image area.
- **material** (96), **shape** (15), **size** (87): below the runnable threshold or heterogeneous
  (size bundles height/length/width/depth/thickness/weight).
- **state** (151): 77% generic "choose X|Y" items with no GQA attribute type.

Full analysis in `docs/CLAUDE.md`.

## Codebase state

Branch `multilayer-ablation`, 72 commits ahead of `main`, 0 behind (clean fast-forward).

Layout: `core/`, `hooks/`, `training/`, `data/`, `feature_analysis/`, `ablation/`, `pipeline/`
(numbered stages 00–04), `tools/` (unnumbered). `evaluation/` was folded into `ablation/`; Gen 0–2
code and superseded reports live in `archive/`.

Multi-layer work added `ablation/multilayer_ablator.py`, `ablation/multilayer_experiments.py`,
`ablation/sample_cache.py`, `tools/run_multilayer_ablation.py`, and
`tools/analyze_multilayer_ablation.py`.

Other branches: `paligemma-2` holds an unmerged PaliGemma-2 3B port using pretrained GemmaScope
SAEs (will conflict heavily — it edits paths the reorg has since moved); `codex/…-tsatoi` holds an
unmerged SAE grid sweep predating v2. `LLaVA-Instruct-150K`, `methodology-improvements`, and
`methodological-improvements-v2` are merged.

## Environment

- Python: `LLaVA-NeXT/.venv/bin/python` — the `llava` package is only importable there.
  Python 3.10.20, torch 2.10.0+cu128, transformers 4.57.6.
- `LLaVA-NeXT/` and `datasets/` are gitignored. See `CLAUDE.md` § "Artifacts not in git".

## Important paths

- Configs: `configs/clevr_lite/` (active), `configs/gqa/` (reference)
- CLEVR-Lite knockout: `output/sae_experiments/exp_default/knockout/knockout_summary.json`
- v2 results: `output/sae_experiments/sae_clevr_lite_layer{0,10,11,12,13,14}_attn_out_question_causal/`
  — cite `ablation_results.json` (n=256) at layer 11, **not** `ablation_v2_results.json` (n=50)
- Multi-layer: `output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question/`
- Paper figures: `output/paper_figures/` (fig1–fig9), mirrored in `overleaf/paper_figures/`
- Paper draft: `overleaf/main_final.tex`
- Verified result tables: `output/sae_experiments/LLM_TECHNICAL_SUMMARY.{md,json}`
