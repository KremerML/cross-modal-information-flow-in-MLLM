# Related work and positioning

Prepared 2026-09-14/15 for *Calibrating Sparse Autoencoder Ablation Against Attention Knockout in a
Vision-Language Model* (`overleaf/main_final.tex`), with `gemma3_addition.tex` and
`llava16_addition.tex`.

Every arXiv/ACL identifier below was fetched during preparation. Where a claim rests only on a
search snippet rather than a fetched abstract or full text, it is marked **[unverified]**. Numbers
from our own runs are read from the committed summaries and the `.tex`, not recalled.

---

## 1. The delta is seven months, not nine

`refs.bib` holds 31 entries; the newest are `rudman2026pih` (2601.05201) and `yang2026monet`
(2602.20330). The draft is current to roughly **February 2026**. What follows concentrates on
**March–September 2026**, with earlier work included only where it is load-bearing and missing.

Four things changed in that window.

**SAEs came under serious attack.** Korznikov et al. (2602.14111, Feb 2026) show that baselines
constraining SAE feature directions or activation patterns to random values match trained SAEs on
interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and **causal editing (0.73 vs
0.72 — the random baseline wins)**. On synthetic data with known features, SAEs recovered **9% of
true features at 71% explained variance**. Their conclusion: "SAEs in their current state do not
reliably decompose models' internal mechanisms." Alongside this, 2607.12166 names **causal
inertness** — features passing a cosine ≥ 0.90 recovery bar with no causal effect (9% in a
well-trained SAE, up to 77% in a degraded one) — and 2607.20596 finds that **SAE family determines
causal behaviour more than scale does**: across three families, six models and 3.9M features,
GemmaScope and BatchTopK features are causally anchored while LlamaScope features are locally
redundant (the token returns to within 2× its pre-ablation rank 96–98% of the time).

This is the most consequential shift for us, and it cuts in our favour. The paper's design — an
external, dictionary-independent yardstick plus activation-matched random controls — is precisely
the discipline this literature says the field lacks.

**The layer-0 dead-dictionary problem was solved by someone else.** Wang et al. (2508.16929, ICML
2026) show attention outputs occupy a subspace of **~60% effective dimensionality** against ~90%
for MLP outputs and the residual stream, identify this low-rank geometry as a key cause of dead
features, and propose **Active Subspace Initialization**, cutting dead features from **87% to below
1%** in attention-output SAEs. Our dictionaries sit on `attn_out`. Layer 0's 74.2% dead rate is a
named, explained, fixable phenomenon.

**The information-flow lineage grew a direct competitor.** Salazar et al. (2607.03358, July 2026)
run attention knockouts and corrupted-input patching on controlled synthetic and natural data and
find two routes — a **direct** route (image tokens read by the final token at later layers) and a
**text-mediated** route (image → query tokens → final token) — with models falling back on one when
the other is ablated. Those are our `Image->Last` and `Image->Question`.

**Native multimodal models were shown to work differently.** 2412.06646 finds that natively
multimodal models funnel image-text communication through **a single post-image token**, where
LLaVA-style adapted models use a distributed pattern across many image tokens.

---

## 2. The map

### C1 — Information flow and knockout

| Work | ID | What it means for us |
|---|---|---|
| Zhang, Yadav, Han, Shutova, *Cross-modal Information Flow in MLLMs* | 2411.18620, CVPR 2025 | Cited. Lower layers move general visual features into question tokens, middle layers object-specific ones, upper layers propagate to the last position. |
| Neo, Ong, Torr, Geva, Krueger, Barez, *Towards Interpreting Visual Information Processing in VLMs* | 2410.07149, ICLR 2025 | Cited. Visual tokens become text-like across layers; object-token removal costs >70% accuracy. |
| Basu et al., *Understanding Information Storage and Transfer in MLLMs* | 2406.04236, NeurIPS 2024 | A *small consistent subset* of visual tokens carries the transfer — our sparsity premise, stated before we started. |
| **Kaduri, Bagon, Dekel, *What's in the Image?*** | **2411.17491, CVPR 2025** | **Missing from `refs.bib`, and more cited (72) than Zhang et al. (66).** Query tokens store *global* image information — the model describes an image from them alone. **Middle layers handle ~25% of cross-modal interactions**; early and late contribute minimally. Independent corroboration of both our knockout claims. |
| **Salazar, Frank, Oneata, Elliott, Fierro, *Pathways of Visual Information Flow*** | **2607.03358, July 2026** | **The nearest competitor.** See §3. |
| Wu, Zhang, Zhou, *How Vision Becomes Language* | 2602.15580, Feb 2026 | Partial Information Decomposition on **LLaVA-1.5-7B and LLaVA-1.6-7B**, six GQA tasks, with Image→Question knockouts as a causality check. Language-unique information ≈82% of final predictions; cross-modal synergy <2%; visual-unique peaks early and decays. **Layer-wise correlations >0.96 across model variants** — independent corroboration of our 1.5↔1.6 layer agreement. |
| Nikankin, Arad, Gandelsman, Belinkov, *Same Task, Different Circuits* | 2506.09047, June 2025 | The co-citation hub of this whole neighbourhood. Circuits are **largely disjoint between modalities but implement similar functionality**; differences lie in processing modality-specific positions. |
| *Visual Representations inside the Language Model* | 2510.04819 | LLaVA-OneVision, Qwen2.5-VL, Llama-3-LLaVA-NeXT. Image **value** tokens carry enough for zero-shot perception; image **key** tokens in later layers carry artifacts. The LM holds *less* visual information than SigLIP. |
| *LLMs Can Compensate for Deficiencies in Visual Representations* | 2506.05439, EMNLP 2025 Findings | Controlled self-attention ablations; the decoder **largely compensates** when visual contextualisation is reduced. |
| *Where Does Vision Meet Language?* | 2601.08151, Jan 2026 | Fusion at specific layers plus a late **"review"** re-attention. The reviewer asked for this cluster (O10). |
| *Understanding Multimodal LLMs: Mechanistic Interpretability of LLaVA in VQA* | 2411.10950 | Value-output matrices extract **colour** from visual embeddings; query-key match question features to visual positions. Closest published work to CLEVR-Lite's colour queries. |

### C2 — SAEs and transcoders for VLMs

| Work | ID | What it means for us |
|---|---|---|
| **Korznikov et al., *Sanity Checks for SAEs*** | **2602.14111** | See §1. The stakes-raiser. |
| *Sparse Visual Thought Circuits in VLMs* | 2603.25075, Mar 2026 | **Closest design neighbour.** Qwen3-VL-8B, SAEs at the mid-decoder layer, controlled synthetic benchmark, norm-matched perturbation controls — and **no model-centric baseline**. Modularity largely fails: the union of two task-selective feature sets induces output drift. |
| *Sparse Autoencoders Learn Monosemantic Features in VLMs* | 2504.02821, NeurIPS 2025 | SAE interventions on CLIP's vision encoder steer LLaVA outputs without touching the LM. |
| *How Visual Representations Map to Language Feature Space in MLLMs* | 2506.11976 | Pre-trained **language-model** SAEs as probes on a frozen-LM/frozen-ViT/trainable-adapter VLM. Visual representations **align with language features progressively, converging in middle-to-later layers**; ViT outputs are badly misaligned with early LLM layers. A second, independent explanation for why layer 0's dictionary fails and the middle band works. |
| *Cascaded Sparse Autoencoders* | 2606.16193, June 2026 | Second-level SAE trained on the first level's *decoder weights*. **Qwen3-VL, Gemma-3 and LLaVA** — our exact model set. |
| *Decomposing Multimodal Embedding Spaces with Group-Sparse Autoencoders* | 2601.20028 | **Split dictionaries**: standard SAEs on multimodal embeddings learn features that are mostly *unimodal*. (CLIP/CLAP embeddings, not decoder `attn_out` — transfer is plausible, not established.) |
| *Do Sparse Autoencoders Generalize? A Case Study of Answerability* | 2502.19964 | Standard SAE feature search **fails across domains** — the general form of our Gemma Scope out-of-domain failure. |
| *Transcoders Trace Visual Grounding and Hallucinations in VLMs* | 2605.22902 | **WITHDRAWN 2026-09-13.** See §3. |
| Kissane et al., *Interpreting Attention Layer Outputs with SAEs* | 2406.17759 | Ancestor of our `attn_z` site. |
| Also: 2607.08605 (S2AE), 2606.25657, 2605.24946 (VISTA), 2605.18229, 2510.03659, 2406.04093, 2506.17673. |

### C3 — Circuits and heads

2510.21518 (Head Pursuit — heads specialise by semantic/visual attribute, editable with minimal
change); 2512.10300 (functional heads universally sparse; vision vs language vs cross-modal heads);
2601.05201 (cited — small head set ablated cuts hallucination ≥40%); 2602.20330 (cited — circuit
tracing in VLMs); 2507.00898 (**ONLY** — *one*-layer intervention suffices to mitigate
hallucination, a counterpoint worth acknowledging against "allocation beats size"); 2507.19110
(LISA, reviewer-named, O12 — *"shallow layers provide visual grounding, middle layers encode semantics,
and deep layers tend to amplify spurious signals"*);
2412.18108; 2505.15865 (OCR heads).

### C4 — Visual token redundancy and pruning-as-interpretability

2503.03321 (**See What You Are Told: Visual Attention Sink** — 147 citations, the most-cited work
in the forward chain); 2602.00462 (**LatentLens** — LogitLens *substantially underestimates* visual
token interpretability, which qualifies the Neo et al. picture); 2510.02912 (challenges
attention-magnitude token selection — the analogue of our activation-dominance worry); 2608.04483
(token roles via EmbedLens; preserving non-alive tokens can maintain or improve performance — the
alive/sink/dead proportions are **[unverified]**); 2603.00510; 2503.20540; 2510.17205; 2606.03569.

### C5 — Modality collapse and text dominance

2508.10552 (*When Language Overrules*); Sim et al., Findings of ACL 2025 (the survey the brief
started from); 2606.28273 (*Vision-Default, Prior-Override*); 2604.02486 (*VLMs Need Words*);
2507.01790; 2606.06890; 2606.24335; 2405.07987 (Platonic Representation Hypothesis) and its 2026
challenge **2604.18572** (*Back into Plato's Cave* — convergence is fragile, degrades at scale,
alignment local and partial).

### C6 — Architectural variation

**2412.06646 (*The Narrow Gate*)** is the key entry; see §8. Also 2511.21631 (Qwen3-VL technical
report — DeepStack injects ViT features at multiple LLM depths, which documents why our
single-contiguous-span assumption breaks there); 2405.09818 (Chameleon); 2605.25343.

### C7 — Narrow counterfactuals and head-level localisation

2406.16320 (**NOTICE**, cited — Semantic Image Pairs as the visual counterpart to symmetric token
replacement, motivated by Gaussian-noise corruption producing *illusory* patching results);
2509.14837 (**V-SEAM** — visual semantic editing plus attention modulation); 2509.17588
(*Interpreting Attention Heads for Image-to-Text Information Flow*); 2511.05923 (**FCCT**, cited as
`li2026fcct` — last-token MHSA in layers 12–18 carries the highest causal effect, plus the **IRI**
intervention the reviewer named); 2505.17127; 2507.13868; 2601.04398; 2508.21258 (RelP);
2511.05442 (APP); 2603.25035.

### C8 — Atomic and diagnostic probes

| Work | ID | What it means for us |
|---|---|---|
| **Savietto et al., *The Geometry of Representational Failures in VLMs*** | **2602.07025** | **36 composites = 6 colours × 6 shapes** (ours: 6 × 3), on Qwen/InternVL/Gemma. Steering success: probe **0.0 / 2.0 / 0.0%**, PCA-probe 44.2 / 46.8 / 17.9%, centroid **78.1 / 35.9 / 75.3%**. Unconstrained probes learn **discriminative shortcuts** — directions that separate classes but lack causal relevance. Binding concluded to be additive. |
| *Counting Circuits* | 2603.18523 | Qwen2.5-VL, Visual Activation Patching + HeadLens. **Counting**, which is *not* one of our task families — a method neighbour, not a task neighbour. |
| Others | 2605.25427, 2602.15183, 2508.16652, 2510.13232 (negation), 2506.15871, 2503.17349, 2407.14494 (InterpBench), 1612.06890 (cited) | |

### C9 — Redundancy vs distribution, cross-layer feature identity

| Work | ID | What it means for us |
|---|---|---|
| **Balagansky, Maksimov, Gavrilov, *Mechanistic Permutability / SAE Match*** | **2410.07656** | The method thread 3 needs. Data-free cross-layer alignment by minimising MSE between **folded** parameters (activation thresholds folded into encoder/decoder weights to absorb scale differences). **Features persist over several layers**; some are pass-through via the residual stream, occupying dictionary entries without adding information. Tested on Gemma 2. |
| **Droste, Adriano, Korte, Giese, *Where Decoder Cosine Similarity Fails*** | **2609.12591, 2026-09-11** | Pythia-160M layers 7→8, 20M tokens, 38,125 strong ablation-effect transitions: **88.0% have both state-target and update-target decoder cosine below 0.7**. Gemma-3-4B layers 21→22: 53.6%. Recommend **ablation-based validation** over cosine thresholds. |
| **Position: Prioritize Feature Consistency in SAEs** | **2505.20254** | Proposes **PW-MCC** across independent training runs; **0.80 achievable for TopK SAEs**. High PW-MCC correlates with ground-truth recovery and with semantic similarity of explanations. |
| *Are Single-Token SAE Features Causally Necessary?* | 2607.20596 | See §1. Family > scale; depth changes the *kind* of causal effect. |
| *From Geometric Recovery to Causal Validation* | 2607.12166 | **Causal inertness**; structural (antipodal geometry, present in good SAEs) vs competitive (a TopK pathology); read- vs write-inertness — features steerable through a decoder atom yet producing zero ablation effect. *Single-author, toy setups plus one production SAE — medium confidence.* |
| *Evolution of SAE Features Across Layers in LLMs* | 2410.08869 | The same-or-different question directly. |
| Others | 2504.02922 (crosscoders), 2506.20040 (cross-layer discrete concepts), 2608.05732 (CircuitSteer), 2511.10840 (multilingual CLT), 2604.01604 (CRaFT), 2501.08319 (output-centric descriptions), 2502.18147 (Jacobian SAEs), EleutherAI seed-similarity blog | |

---

## 3. Nearest neighbours and threat assessment

### The real competitor: Salazar et al., 2607.03358

Same method family (attention knockout plus patching), same controlled-synthetic setting, and it
names both of our flows: their **direct** route is our `Image->Last`, their **text-mediated** route
is our `Image->Question`. Their finding that **models fall back on the text-mediated pathway when
the usual one is ablated** is a live alternative account of our results, and our four re-reading
tests do not cover it. We tested compensation *within* a pathway — downstream re-reading along
`Image->Question`. We did not test *switching between* pathways.

This matters most for LLaVA-1.6, where A exceeds K at short spans (R = 194.5% at {13,14}). If
feature ablation suppresses the fallback route while an `Image->Question` knockout leaves it
available, A can exceed K with no dictionary pathology at all. That is a cleaner explanation than
anything currently in `llava16_addition.tex`, and it is testable with the existing harness (§12).

### The withdrawn paper

**arXiv 2605.22902 was withdrawn on 2026-09-13** — the day before this review. Verified from the
arXiv submission history: v1 2026-05-21 (20,602 KB), v2 2026-09-13 (1 KB, withdrawn), comment
*"Focused analysis on attribution maps revealed different behaviors than the ones reported"*. The
retracted part is exactly the SAE-vs-transcoder comparison on Gemma 3-4B-IT. It cannot be used to
corroborate the Gemma boundary-conditions result. Its **question** — are transcoders better than
SAEs for cross-modal attribution — is now demonstrably open, which is an opportunity.

### Citation gaps that will be noticed

1. **2411.17491 (Kaduri et al., CVPR 2025)** — 72 citations, more than Zhang et al., corroborates
   both our knockout claims, and is absent. This is the most conspicuous omission.
2. **2506.09047 (Nikankin et al.)** — cites Zhang, Neo, Basu *and* NOTICE. Anyone reading our
   related work will have read it.
3. **2503.03321** — 147 citations, the most-cited work in the forward chain.
4. **2602.15580** — runs both our LLaVA models and independently corroborates our layer agreement.

### What is still unclaimed

Nothing found calibrates a sparse-dictionary ablation against a model-centric causal ceiling **at a
matched locus, on the same samples, with the ratio reported**. 2603.25075 comes closest in design
and explicitly lacks the baseline. The gap is real; §6 states how to argue it.

### What is at risk

- **The middle-layer localisation is no longer novel** and was not ours to begin with (Zhang,
  Kaduri, Basu, Jiang). It must be presented as replication, not discovery.
- **The conduit claim** is independently established by Kaduri et al.
- **Single-model scope** is answered by our own additions, not by the literature.

---

## 4. Where the LLaVA-1.6 result lands

R moves 72.6% → 93.6% with the language model, prompt, data, recipe and span all fixed and every
precondition holding. Three anchors now exist for it.

**It is the one comparison with SAE family held fixed.** 2607.20596 finds cross-family causal
differences exceed within-family scale effects. LLaVA (in-house ReLU/L1) versus Gemma (Gemma Scope
JumpReLU) confounds model with dictionary family. LLaVA-1.5 versus 1.6 does not. **This makes 1.6,
not Gemma, the load-bearing replication**, and the draft should say so.

**A pathway-substitution reading is available.** Salazar et al. supply the mechanism; our own
`Image->Last` peaks *later* in 1.6 (layer 19: +0.639) than in 1.5, which is consistent with a
fallback route that is more available in 1.6.

**Visual information is more diffuse in 1.6 by construction.** 1176 image tokens against 576. If
dictionary features at post-image text positions aggregate a wider visual context while an
`Image->Question` knockout severs only the edges into those positions, the two interventions are no
longer measuring the same quantity — which is a sharper statement of "R is not a property of the
language model alone."

---

## 5. Terminology

**Do not adopt "absorption."** Chanin et al. (2409.14507) fixed it as a sparsity-driven SAE
pathology where a latent is swallowed by its children. The project does not currently use the word
as a claim, and should not start. The existing phrasing — *"Question tokens rather than the
prediction position are therefore the conduit"* — is unambiguous.

**"Knockout" is safe and correctly attributed.** The field uses it as we do, descending from Geva
et al. (2304.14767). Kaduri et al. and Salazar et al. use it for VLMs.

**"Redundancy index" is unclaimed.** No competing use found. But the docs are right that R is not
constant, and §3's competitor makes the point sharper: R mixes pathway mediation with pathway
substitution.

**Adopt "causal inertness" (2607.12166) with attribution** if we want a name for part of the
shortfall, and **PW-MCC (2505.20254)** for dictionary consistency.

**Fix "question positions."** The repo already records that it means all post-image text positions.
Salazar et al. call these *query tokens*, Kaduri et al. the same. Using their term costs nothing
and removes the misnomer.

---

## 6. Candidate open questions, ranked

### Q1 (recommended) — When can a sparse-dictionary ablation be read as measuring a pathway?

*The field's version:* SAEs are under active attack for failing to beat random baselines on causal
editing (2602.14111), for causal inertness (2607.12166), and for family-dependent causal behaviour
(2607.20596). Nobody has proposed an external standard for when a dictionary ablation licenses a
mechanistic claim.

*Who owns it:* unclaimed. 2603.25075 is closest and explicitly has no model-centric baseline.

*What we answer:* all of it. R is the calibration; the three Gemma preconditions are the validity
criteria; LLaVA-1.6 shows the calibration moves even when the preconditions hold. The Gemma
negative result stops being an embarrassment and becomes the paper's second contribution.

*What a reviewer will demand that we lack:* a cross-pathway control (§3); a seed replicate; and
layer 0, which ASI (2508.16929) now makes addressable.

**Recommended.** It is the only framing where all three runs — including the negative one — are
load-bearing, and it sits on the live controversy rather than beside it.

### Q2 — Is cross-modal transfer mediated by sparse features, or only localised?

*Who owns it:* contested. Basu et al. established the sparse-subset claim for tokens; the SAE-for-VLM
cluster is crowded (2504.02821, 2603.25075, 2606.16193, 2506.11976).

*Risk:* our answer is partly negative (72.6% at best, layer 0 unaddressable), and the crowd is large.

### Q3 — Where does visual information go, and what carries it?

*Who owns it:* Zhang, Neo, Kaduri, Basu, Salazar, and the pruning literature. **Crowded and largely
settled.** Our knockout arm would read as replication.

---

## 7. Result-by-result mapping

| Result | Q1 (recommended) | Q2 | Q3 |
|---|---|---|---|
| LLaVA-1.5: R = 72.6% [69.6, 75.8], trend −0.010/layer (n.s.) | The calibration itself | Partial mediation, quantified | Replicates known localisation |
| Single-layer ceiling ≈0.24 against 20× budget | Budget is not the binding constraint — a validity fact | Evidence against pure sparsity | — |
| Allocation beats size, 2.90× [2.51, 3.37] at 54% perturbation | Where a budget is spent is a design rule for the field | Distribution over redundancy | Cross-layer structure |
| Downstream re-reading rejected on four tests | Rules out one confound; §3 names two it does not | Supports distribution | — |
| Gemma: preconditions fail, R = 121.5% | **The boundary conditions — the second contribution** | Negative result | Not interpretable |
| LLaVA-1.6: preconditions hold, R = 93.6% | **The load-bearing replication** (family fixed) | Mediation is not a constant | Layer set replicates exactly |

---

## 8. Going finer than a layer

Our unit is a layer-and-flow. The field has moved to heads and to narrow counterfactuals.

**Corruption design.** NOTICE's core methodological point is that Gaussian-noise corruption yields
*illusory* patching results, and that semantic image pairs are needed. **This does not touch our
knockout**: we do not corrupt inputs at all — we mask attention edges in the materialised 4D
additive mask. Worth saying explicitly, because a reviewer who knows NOTICE will wonder.

**Head-level localisation.** 2509.17588 does image-to-text flow at head level; 2510.21518 and
2512.10300 find functional heads are universally sparse; 2601.05201 ablates a small head set for a
≥40% hallucination reduction. Our `knockout/block_config.py` grammar is defined over
`(target, source)` position pairs at a layer; a head axis would be a real extension, not a
reframing, and `mask_hooks` rewrites a mask that is already per-head-broadcastable.

**Our own coarseness, stated honestly.** The `Image->Question` span is all post-image text
positions, including the answer-format suffix and `ASSISTANT:`. Salazar et al. and Kaduri et al.
both call these query tokens and both find they act as *global* image descriptors — which makes our
span defensible as a unit, but it is a unit, not a localisation.

**The early-fusion objection is correct and now citable.** 2412.06646 finds native multimodal models
funnel image-text communication through a **single post-image token** whose ablation badly degrades
image understanding, against a distributed pattern in adapted models, with greater image/text
separation in the residual stream. So the distributed span we measure is a property of the
late-fusion adapter architecture. Nikankin et al. (2506.09047) add that circuits are largely
disjoint across modalities while implementing similar functionality — so "LLaVA repurposes text
machinery" is too strong even for LLaVA. Both belong in Limitations, and together they name the
experiment that would settle it: run the same span/flow design on a native model and predict the
span collapses to one position.

---

## 9. CLEVR-Lite among the diagnostic probes

Three templates — `query_color_unambiguous` (2,637), `query_shape_unambiguous` (4,039),
`query_color_negation` (1,114) of 7,790 validation questions. Colour, shape and negation. **No
counting, no spatial relations.**

**It is an asset, and there is now a paper to argue it with.** Savietto et al. (2602.07025) build
36 colour×shape composites and find naive probes achieve **0–2% steering success** while centroid
methods reach 78.1% — because unconstrained probes learn *discriminative shortcuts*, directions
that separate classes without causal relevance. That is our "ghost features" result (fig4) arrived
at independently, in the concept-vector setting, on three other models. It is the strongest
available justification for our switch to gradient-weighted attribution selection, and it should be
cited exactly there.

**Where it is a limitation.** The reviewer's "one synthetic dataset" objection stands on
generalisation, not on atomicity. Neighbours all use several tasks: 2603.25075 uses 7 task types ×
3 difficulties, 2603.18523 spans multiple visual tasks.

**The unused held-out compositional split (11 train vs 7 held-out colour-shape combinations) is now
worth running.** 2602.15183 (*Seeing to Generalize: How Visual Data Corrects Binding Shortcuts*) and
2508.16652 make held-out combinations a live question, and we already have the split. Running the
existing multilayer design on held-out combinations would answer "one task" with almost no new
machinery, and would say whether R differs for compositions the model never saw.

---

## 10. Telling redundancy from distribution by reading the features

The proposal: interpret the top-k features ablated at each layer and ask whether the same features
recur (redundancy) or different ones appear (distribution). It would convert an inference from
effect sizes into an observation about content.

### What is on disk (verified 2026-09-14, `vlm-flow-probe`)

- **`causal_feature_catalog.json` exists for all 34 Gemma layers and all 32 LLaVA-1.6 layers; 66 are
  tracked in git.** Each maps feature id → `{name, type, causal_score, activation_mean,
  gradient_mean}`.
- **The catalogs hold only the selected top-200**, not the full scored dictionary.
- **LLaVA-1.5 has no catalogs in this repo.**
- Condition `summary.json` records `features_per_layer` counts but **not** which features were
  ablated; `control_feature_sets` is empty. The top-k sets are nonetheless reconstructible from the
  catalogs, since selection is deterministic in `causal_score`.
- **No SAE checkpoints are present locally** — the SAE directories hold `config.yaml`,
  `provenance.json` and `reconstruction_eval.json` only. Local `output/` is 17 MB. **Decoder
  matrices are on the cluster.**

So: the score-structure half runs locally right now with no GPU. The feature-identity half needs the
cluster.

### What the catalogs already show

Computed from the committed catalogs:

| | activation range within top-200 | gradient range | raw top-200 index overlap, adjacent layers |
|---|---|---|---|
| LLaVA-1.6, L10–14 | 2.25–3.58× | 1.67–3.09× | 0–4 of 200 (0.0–2.0%) |
| Gemma 3, L11–17 | 85.6–231× | 3.82–7.10× | 6–12 of 200 (3.0–6.0%) |

Two things follow. **Within the selected set**, LLaVA-1.6's activation and gradient terms have
comparable spread, while Gemma's activation spread is ~20–50× its gradient spread — so the
activation-dominance concern (O4) is far more acute for Gemma than for LLaVA-1.6. And **raw index
overlap is at or near chance** (expected ≈0.6% for LLaVA-1.6's 32,768-wide dictionaries, ≈1.2% for
Gemma's 16,384), confirming that index comparison across separately-trained dictionaries carries no
identity information. That is the null the real experiment has to beat.

### The three hazards

1. **Index incomparability** — handled above; identity must come from decoder directions, activation
   patterns on shared samples, or max-activating examples.
2. **Decoder cosine is a weak functional proxy.** 2609.12591: 88.0% of strong ablation-effect
   transitions fall below 0.7 cosine (Pythia-160M 7→8); 53.6% on Gemma-3-4B 21→22. A
   low-similarity result would **not** license "different features, therefore distribution" — that
   is precisely the inference they undercut. Their recommendation is ablation-based validation,
   which is what we already do.
3. **Seed variance.** SAEs on the same data with different seeds learn different features, and the
   draft trains one seed per layer. Without a same-layer baseline, cross-layer dissimilarity is
   uninterpretable.

### The design

**Step 0 (control, runs first).** Train a second SAE at one span layer with a different seed.
Compute **PW-MCC** (2505.20254) between the two same-layer dictionaries. This is the consistency
floor. Reference value 0.80 for TopK SAEs on LLM activations — but ours are ReLU/L1, so measure,
do not assume.

**Step 1 (matching).** Apply **SAE Match** (2410.07656) — parameter folding, then minimise MSE
between folded parameters — to adjacent span layers. Report matched-pair decoder cosine *and* the
folded-MSE objective, not cosine alone.

**Step 2 (grounding).** For each matched pair, measure the **ablation effect** of each member on the
shared 256-item set. Per 2609.12591, ablation effect is the trustworthy signal; similarity is the
descriptor.

**Step 3 (the reading).** Cross-layer similarity is evidence of *redundancy* only if it clearly
exceeds the Step 0 floor **and** matched pairs have correlated ablation effects. Similarity below
the floor is evidence of *distribution*. Similarity above the floor with uncorrelated ablation
effects is the pass-through case 2410.07656 describes — features occupying dictionary entries
without adding information — and is evidence for neither.

### What it can and cannot settle

It can distinguish "the same directions recur" from "different directions appear", grounded in
ablation effect rather than geometry alone. It **cannot** settle redundancy versus distribution on
its own, because 2609.12591 shows functional connection and geometric similarity come apart, and
because pathway substitution (§3) produces a distribution-like signature for a different reason.
Best read as a third, independent line of evidence alongside the four existing tests — narrowing the
question, not closing it.

---

## 11. The standing objections, answered

From the Stanford Agentic Reviewer (ICLR submission 2026-08-19, 6.1/10, weak accept).

| # | Objection | Status |
|---|---|---|
| O1 | One model, one dataset | **Answered by our own work** — Gemma 3 and LLaVA-1.6. Cite 2607.20596 for why family-fixed 1.6 is the stronger of the two. |
| O2 | 256 non-random eval items | **Open.** No literature relief. Cheap to fix. |
| O3 | One SAE seed | **Open, and now named** — 2505.20254 gives the metric (PW-MCC). Also Step 0 of §10. |
| O4 | Selection dominated by activation magnitude | **Sharpened.** 2602.07025 shows correlational selection fails causally; §10 shows the concern is acute for Gemma, mild for LLaVA-1.6. Successors: 2508.21258 (RelP), 2502.18147. |
| O5 | Replace-mode reconstruction error; no delta comparison | **Partly answered** — Gemma ran delta mode. |
| O6 | Layer 0 dictionary 74% dead | **Answered by 2508.16929** — dimensional collapse in attention outputs; ASI takes dead features 87% → <1%. 2506.11976 adds a second reason (early-layer visual reps are least language-aligned). Converts a dead end into a next experiment. |
| O7 | Single random spread control | **Open, and now urgent** — 2602.14111 makes controls the whole ballgame. |
| O8 | "Near-constant 72%" needs regression | **Answered** — the analyzer reports slopes (A 0.193/layer, K 0.269/layer, ratio 0.718) and R-vs-span trend. |
| O9 | Activity density; features not very sparse | **Open.** |
| O10 | Missing layer-wise fusion work | **Answered** — 2601.08151, plus 2411.17491 and 2506.09047. (2504.21447, *MLMs See Better When They Look Shallower*, is about **vision-encoder** layers, not LM layers — not relevant here.) |
| O11 | Missing low-rank attention-output geometry | **Answered** — 2508.16929. |
| O12 | Missing LISA, FCCT+IRI | **Answered** — 2507.19110, 2511.05923 (already cited), 2507.00898. |
| O13 | Why is `Image->Last` weak here? | **Sharpened into a real tension.** FCCT finds last-token MHSA in layers 12–18 has the *highest* causal effect; Salazar et al. make the direct route real but conditional. Needs a paragraph, not a footnote. |
| O14 | Question-token set underspecified | **Known** — "question positions" = all post-image text positions. Adopt the field's "query tokens". |

---

## 12. What to read next, and what is cheap to run

**Read first (full text):** 2607.03358 (the competitor), 2411.17491 (the citation gap), 2506.09047
(the hub), 2508.16929 (the layer-0 fix), 2602.14111 (the stakes).

**Cheap with the existing harness:**

1. **The cross-pathway control.** Measure `Image->Last` knockout *under* `Image->Question` ablation
   on the shared 256-item set. If the fallback route carries more when features are ablated than
   when edges are cut, that explains LLaVA-1.6's R > 1 without any dictionary pathology. This is the
   single highest-value experiment the review turned up: it addresses the nearest competitor, the
   most exposed claim, and O13 at once.
2. **The seed replicate** (§10 Step 0). Answers O3 and unlocks thread 3.
3. **The held-out compositional split** (§9). Already generated; answers "one task" cheaply.
4. **A randomised 256-item draw.** Answers O2 directly; the reviewer asked for it by name.
5. **Layer 0 with ASI** (2508.16929). Medium cost, but it addresses the strongest transfer site,
   which the draft currently concedes.

**Not cheap, and probably not now:** head-level localisation (§8); a native-multimodal replication.

---

## Provenance

- Candidate ledger: 138 entries across nine clusters.
- Close reads: ~27, notes per paper alongside the ledger.
- Forward-citation chain via Semantic Scholar from 2411.18620 (66 citing), 2410.07149 (100),
  2406.04236 (53), 2406.16320 (22), 2502.03032 (12), 2410.08869 (15), 2410.07656 (22),
  2605.22902 (0).
- Search surfaces: `WebSearch` and direct arXiv/ACL fetches. OpenAlex has no citation edges for this
  subfield (2411.18620 is W4404997774 with `cited_by_count: 0`); the arXiv Atom API was unreliable
  from this host.
- **Falsification pass on the §3 "nothing found calibrates a dictionary ablation against a
  model-centric ceiling" claim:** queried for "SAE ablation vs attention ablation upper bound",
  "how much of circuit recovered sparse features ceiling", "attention knockout vs SAE calibration",
  and checked the forward citations of 2406.17759 and 2403.19647. Nearest hit remains 2603.25075,
  which states it has no such baseline. The claim survives, at medium-high confidence.
