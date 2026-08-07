# Project Memory: cross-modal-information-flow-in-MLLM

Last synced: 2026-08-06. This is the in-repo copy of the working memory that otherwise lives
only on the author's machine (`~/.claude/projects/.../memory/`), so the context travels with a
clone. Full project context is in `CLAUDE.md` at the repo root; verified result numbers are in
`output/sae_experiments/LLM_TECHNICAL_SUMMARY.md`.

**Project phase: writeup.** No further experiments planned. Remaining work is documentation,
visualization, and methodology description.

## Research goal

Study cross-modal information flow in LLaVA-v1.5-7b (32 layers, d_model=4096) by combining
attention knockout with SAE feature ablation, measuring
`margin = log P(true option) − log P(false option)`.

## Key finding 1 — CLEVR-Lite knockout sweep

33-hour sweep, n=7084, baseline margin 3.178.
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

**Three-region structure:** layer 0 = immediate cross-modal registration; layers 10–12 =
mid-depth semantic binding (with layer 13 inhibitory *inside* the cluster); layer 14 = final
visual-semantic consolidation.

Versus GQA: effect sizes 25–50% larger, confirming that GQA ChooseAttr's language-side bypass
was diluting effects. `Image->Last` is also much stronger here (max d=1.15 at layer 2 vs ~0.20
on GQA). GQA's two-peak structure (0 and 11) becomes a richer multi-peak structure.

## Key finding 2 — v2 causal feature ablation

**This broke an 18-experiment null streak (May 2026).** All numbers below re-verified 2026-08-06.

Feature ID on the full val set (n=7790); ablation on 256 samples; `attn_out`, `question`
positions, `replace` mode, 15 random control sets (**uniform, not matched — see the caveats
below; every z here is inflated**).

| Layer | ablation margin_drop | z | % positive | flips | % of knockout ceiling |
|---|---|---|---|---|---|
| 0 | 0.0200 | 73.1 | 56.6% | 0 | 4.4% |
| 10 | 0.1514 | 66.5 | 81.2% | 3 | 56.4% |
| 11 | 0.2131 | 81.6 | 84.8% | 4 | 49.9% |
| 12 | 0.1670 | 56.0 | 82.8% | 3 | 68.1% |
| 13 | 0.1220 | 47.7 | 64.8% | 3 | — |
| **14** | **0.2433** | 70.0 | **90.2%** | 2 | **61.1%** |

**Layer 14 is the strongest result, not layer 11** (run 2026-08-06): largest drop, highest
% positive, largest knockout d (1.205), healthy SAE. Layer 11's higher z reflects quieter
control sets, not a bigger effect.

**Correction to older prose:** the "39% of knockout ceiling" figure compared the CLEVR-Lite
ablation against the *GQA* knockout drop (0.540). Against CLEVR-Lite's own layer-11 knockout
drop (0.4273) the correct figure is **49.9%**.

**Root cause of the prior nulls:** v1 scored features by correct/incorrect activation ratio,
which is maximised by features that barely activate — "ghost features". At layer 11 the v2
top-500 features have mean activation 0.117 against a median of 6.1e-08 across all 32768
features. Zero overlap between v1 and v2 top-200 sets.

**v2 method:** insert the SAE into the forward pass, backprop the target to feature
activations, score `|grad| × |activation|`. Marks et al. 2024 (Sparse Feature Circuits),
Agrawal et al. 2025. Implementation: `feature_analysis/causal_feature_identifier.py`.

## Caveats that must appear in the writeup

- **Layer 0 is a dictionary-training failure, not a negative result.** Its SAE has
  `dead_feature_fraction = 0.742` (74% of features never fire) against 0.06–0.4% at every
  other layer. Its 4.4%-of-ceiling ablation says nothing about the model.
- **Layer 13 is a genuine puzzle.** Knockout there is inhibitory (−0.048) but ablating its
  causal features hurts (+0.122). The two interventions disagree in sign; "% of ceiling" is
  undefined. Worth a sentence rather than a quiet omission.
- **Mean L0 is 1100–1600** across all six SAEs. Reconstruction is near-perfect
  (explained variance > 0.998) but the dictionaries are not very sparse, which limits claims
  about individual features being clean interpretable units.
- **Single-layer ablation only.** The model can re-read the image at layers L+1…31, the most
  likely reason ablation captures ~50% rather than 100% of the knockout ceiling. A multi-layer
  harness to test this is under construction on branch `multilayer-ablation`; no results yet.
- **The random controls were uniform, not matched** (found 2026-08-07). Configs set
  `matched_metric: "correct_mean"`, absent from every v2 stats file, so the matched sampler
  silently fell back to uniform draws. At layer 11 the controls have median activation 6.1e-08
  against the binding set's 0.117. **Every z above is inflated.** The margin drops and
  per-sample positive fractions are unaffected. Re-runs land in `ablation_matched_controls.json`.

## CLEVR-Lite dataset

Synthetic: 6 colors × 3 shapes, PIL-rendered 224×224 PNGs, open-ended queries ("What color is
the triangle?") so there is no language-side bypass. 186,638 train / 7,790 val questions
(50K scenes × ~3.7 q/scene). Generated deterministically — `tools/generate_clevr_lite.py
--seed 32` reproduces it exactly.

A `false option` is synthesised by deterministically sampling a distractor from the closed
attribute set, so open-ended questions still support the forced-choice `margin` metric.

## GQA ChooseAttr dataset quality (Mar 2026, reference only)

- **color** (602 rows): the only clean, homogeneous category — use it for any GQA work.
  ~58% involve black/white; 20.6% have objects covering <1% of image area.
- **material** (96), **shape** (15), **size** (87): below the runnable threshold or
  heterogeneous (size bundles height/length/width/depth/thickness/weight).
- **state** (151): 77% generic "choose X|Y" items with no GQA attribute type.

## Codebase state (branch `LLaVA-Instruct-150K`, 54 commits ahead of `main`)

Post-reorganization layout: `core/`, `hooks/`, `training/`, `data/`, `feature_analysis/`,
`ablation/`, `pipeline/` (numbered stages 00–04), `tools/` (unnumbered). `evaluation/` was
folded into `ablation/`; Gen 0–2 code lives in `archive/`.

Other branches: `paligemma-2` holds an unmerged PaliGemma-2 3B port using pretrained
GemmaScope SAEs (will conflict heavily — it edits paths the reorg has since moved);
`codex/…-tsatoi` holds an unmerged SAE grid sweep. `methodology-improvements`,
`methodological-improvements-v2`, and `codex/…-causal-impact` are fully merged.

## Environment

- Python: `LLaVA-NeXT/.venv/bin/python` — the `llava` package is only importable there.
- `LLaVA-NeXT/` and `datasets/` are gitignored. See `CLAUDE.md` § "Artifacts not in git".

## Important paths

- Configs: `configs/clevr_lite/` (active), `configs/gqa/` (reference)
- CLEVR-Lite knockout: `output/sae_experiments/exp_default/knockout/knockout_summary.json`
- v2 results: `output/sae_experiments/sae_clevr_lite_layer{0,10,11,12,13}_attn_out_question_causal/`
  — cite `ablation_results.json` (n=256) at layer 11, **not** `ablation_v2_results.json` (n=50)
- Writeup figures: `output/sae_experiments/report_assets/`
- Verified result tables: `output/sae_experiments/LLM_TECHNICAL_SUMMARY.{md,json}`
