# Related work — round 2

Prepared 2026-09-21 for *Calibrating Sparse Autoencoder Ablation Against Attention Knockout in a
Vision-Language Model* (`overleaf/main_final.tex`, with `gemma3_addition.tex` and
`llava16_addition.tex`). Gates and process rules from `docs/literature-review-guidelines.md`.
Round 1 is `docs/related-work_deprecated.md`; it is a failure case, not a source.

## 1. What changed

Round 1 searched by topic — VLM plus SAE — and found the VLM interpretability literature, which is
young and largely unreviewed; one of its 55 entries survived the gates. Our contribution is not a
VLM claim. It is the claim in the abstract that *a flat response to added features is
indistinguishable from having captured everything available* without an independent estimate of the
total effect, which is a claim about intervention validity. Round 2 searched by that claim and
found the neighbours in the text-only LLM evaluation literature: five peer-reviewed papers
(COLM 2024, ICLR 2025, ICML 2025 ×3) and three 2026 preprints from labs with a track record.

Gate 2 was amended once, and only once: text-only work is in scope when its claim is about the
**intervention** — validity, faithfulness, completeness, calibration — and out of scope when it is
about a model's mechanism. The video rejection (2609.05149) is unaffected.

Nine entries, all verified by fetch on 2026-09-21, are in `docs/related-work.bib` with their gate
verdicts. Eight are reported below; `kantamneni2025saeprobing` is covered inside entry 6.

## 2. The list

### 1. Miller, Chughtai & Saunders — *Transformer Circuit Faithfulness Metrics are not Robust*
COLM 2024 · arXiv 2407.08734 · **peer-reviewed** · `miller2024faithfulness`

They vary six axes of ablation methodology — node vs edge vs branch, zero/Gaussian/resample/mean
value, all tokens vs specified tokens, ablate-clean vs restore-clean, circuit vs complement,
granularity — and measure how much circuit faithfulness moves. Edge-level ablation scores
substantially higher than node-level; resample scores systematically lower than mean (p = 1e-5);
ablating all tokens scores lower than token-specific ablation. On IOI, the same circuit scores 87%
under Wang et al.'s node-level mean ablation at specific positions and a median well over 100%
under edge + specific tokens + mean. Their conclusion: faithfulness scores reflect the researcher's
methodological choices as much as the circuit.

**Consequence.** R = A/K divides a numerator and a denominator that sit on opposite sides of their
strongest axis. A is a node-level zero-ablation (`replace` mode discards the feature's contribution
at a site); K is an edge-level mask on Image→Question attention. They show that pairing is not
neutral and is directionally biased, which puts a named methodological term inside our headline
number. It also supplies a cleaner explanation of R > 100% than dictionary pathology: Gemma's
121.5% and LLaVA-1.6's 194.5% at {13,14} are the regime where their edge-level configuration
exceeds 100% on a circuit known to be correct.

### 2. Gerasimov, Rusalev, Balagansky, Laptev, Kurochkin & Gavrilov — *Unstable Features, Reproducible Subspaces*
arXiv 2606.12138, 2026-06-10 · preprint, T-Tech (the SAE Match group) · `gerasimov2026seeddependence`

Feature stability across seeds, at scale over models, layers, dictionary sizes and SAE variants.
"Stable features carry most of the reconstruction- and prediction-relevant signal, while unstable
features have weak marginal impact." Unstable features are individually non-reproducible but
"concentrate in reproducible lower-rank subspaces."

**Consequence.** This is the first named mechanism for our budget ceiling, and it already matches
our data. Our rank-band controls put the whole effect in ranks 1–40 at layer 11 (+0.1895), with
ranks 41–400 at ≈0 — exactly the split between a small stable set carrying the signal and an
unstable tail with weak marginal impact. It also reframes the flat curve: if the tail re-spans a
subspace the head already covers, adding features is not supposed to help, and the ceiling is a
property of the dictionary's stable rank rather than of the model.

### 3. Bayat Makou, Niu, Dutta & Gurevych — *Many Circuits, One Mechanism*
arXiv 2606.06267, v2 2026-08-26 · preprint, UKP Lab TU Darmstadt · `bayatmakou2026manycircuits`

75 circuits from five Pythia models across four token-frequency bands. Structurally distinct
circuits implement the same computation: band-specific edges transfer across bands, a shared core
recovers ≥99% of circuit performance above 70M, and interchange interventions confirm internal
representations are interchangeable. They name the mismatch **phantom specialization**, and find
that repeated extraction within one band samples from an equivalence class of valid subgraphs
rather than recovering a unique mechanism. Source-level evaluation inflates faithfulness; edge-level
evaluation exposes the many-to-one mapping.

**Consequence.** Our distribution reading rests partly on the observation that top-200 feature sets
at adjacent layers overlap at chance (0–4 of 200). That is a structural comparison, and they show
structural difference is not sufficient evidence of distinct mechanisms. Their prescribed test —
cross-condition transfer plus a shared core — is the one our design has not run. Together with
entry 2, two independent groups now say a selected set is one draw from an equivalence class.

### 4. Leask, Bussmann, Pearce, Bloom, Tigges, Al Moubayed, Sharkey & Nanda — *Sparse Autoencoders Do Not Find Canonical Units of Analysis*
ICLR 2025 · arXiv 2502.04878 · **peer-reviewed** · `leask2025canonical`

SAE stitching shows dictionaries are incomplete; meta-SAEs, trained on a decoder matrix, show
latents are not atomic (an "Einstein" latent decomposes into "scientist", "Germany", "famous
person"). They recommend choosing SAE size for the task rather than assuming a canonical set.

**Consequence.** Our budget curve varies k from 40 to 800 at a fixed width of 32,768. Completeness
is a property of width, not of k, so the flat curve cannot bear on whether the dictionary contains
the rest of the mechanism — which is the first of the three explanations the Discussion leaves
open. The project already has this experiment queued for a different reason (whether the ~1,178
live features at layer 11 reflect the model or an undertrained dictionary); this makes it load-bearing.

### 5. Hoang, Chatterjee, Chakraborty, Gurevych & Dutta — *Sparse Autoencoders Encode Both Concepts and Functions*
arXiv 2607.24645, 2026-07-27 · preprint, UKP Lab and IIT Delhi · `hoang2026fega`

FEGA removes the same active feature across contexts and analyses the resulting cloud of logit
changes. "Across SAE variants, consistent one-dimensional effects are rare: few features behave
like reusable directions." Value-like features (static attributes) give structured but
multi-directional effects; pointer-like features (context-dependent operations) give diffuse ones.
A feature can be interpretable and causally relevant without providing a stable direction.

**Consequence.** A competing account of the same ceiling as entry 2, and the two are separable. If
added features have diffuse, mutually non-aligned effects, a budget does not accumulate however
stable its members are. It also bears on selection: our ranking is `|grad| × |activation|`, a score
defined at a single point, and they show the effect of a feature is not a fixed direction across
contexts. Our own layer-12 contradiction — top-200-by-activation beating the causal top-200 — is
the kind of result their framework predicts.

### 6. Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning & Potts — *AxBench*
ICML 2025 (PMLR 267) · arXiv 2501.17148 · **peer-reviewed** · `wu2025axbench`

A benchmark for steering and concept detection on Gemma-2-2B and 9B. Prompting beats every method
at steering; difference-in-means beats every method at concept detection; SAEs are not competitive
at either. Kantamneni, Engels, Rajamanoharan, Tegmark & Nanda (ICML 2025, PMLR 267:29018–29049)
reach the same verdict on a third task, sparse probing, where SAEs fail to beat baselines under
data scarcity, class imbalance, label noise and covariate shift.

**Consequence.** Our control arm is random feature sets, and the rank-band rework already showed
that arm was not identified. These two set the field's bar higher than random: a simple supervised
direction at the same locus. That is also a sharper novelty argument than round 1's. SAEs now have
an external standard on probing, steering and detection, and lose on all three; ablation — the
modality carrying the strongest mechanistic claims — is the one with no external standard, which is
what R supplies.

### 7. Karvonen, Rager, Lin, Tigges, Bloom, Chanin, Lau, Farrell, McDougall, Ayonrinde, Till, Wearden, Conmy, Marks & Nanda — *SAEBench*
ICML 2025 · arXiv 2503.09532 · **peer-reviewed** · `karvonen2025saebench`

Eight metrics spanning interpretability, feature disentanglement and applications: sparse probing,
unlearning (WMDP-bio), spurious correlation removal, targeted probe perturbation, and others. The
causal evaluations use zero ablation and score against probes or task accuracy.

**Consequence.** This is the evidence under the novelty claim rather than a threat to it. Every
SAEBench metric is dictionary-internal or probe-referenced; none compares a feature ablation to a
model-centric intervention that severs an identified pathway at the same locus. Argue novelty
against this list by name, which is checkable, instead of against the absence of search hits.

### 8. Salazar, Frank, Oneata, Elliott & Fierro — *Pathways of Visual Information Flow in Vision-Language Models*
arXiv 2607.03358, 2026-07-03 · preprint, University of Copenhagen and DTU · **carried from round 1** · `salazar2026pathways`

Two routes: a direct route where the final token reads image tokens at later layers, and a
text-mediated route through query tokens. Pathway choice is task-, data- and prompt-dependent, and
models fall back on the text-mediated route when the usual one is ablated. Their own conclusion —
"ablation-based interventions can reveal what models could do rather than what they normally do" —
is the same warning entries 1 and 3 give from the text side.

**Consequence.** Unchanged from round 1 and still the only in-modality competitor. Their direct and
text-mediated routes are our `Image->Last` and `Image->Question`. Our four re-reading tests cover
compensation within a pathway, not substitution between pathways, so pathway substitution remains a
live alternative account of both the ceiling and LLaVA-1.6's R > 100%.

## 3. Consequences, consolidated

| Our claim at risk | What the paper implies | What we run |
|---|---|---|
| R = 72.6% [69.6, 75.8] measures how much of the pathway the features carry | Node-level vs edge-level is the largest axis in their sweep, and our numerator and denominator sit on opposite sides of it (1) | Mean-ablate `attn_out` at post-image text positions as a node-level ceiling; recompute R against it and report both |
| Gemma R = 121.5%, LLaVA-1.6 {13,14} R = 194.5% need a dictionary explanation | A correct circuit scores >100% under one defensible configuration (1); pathway substitution gives a second mechanism (8) | `Image->Last` knockout *under* `Image->Question` ablation on the shared 256 items |
| The single-layer ceiling (≈0.24 across a 20× budget) bounds what the dictionary holds | Completeness is a width property, not a k property (4) | Retrain at a second width at layer 11 or 14; re-run the k-sweep there |
| The flat tail means the features captured what is available | The tail is the unstable, weak-marginal-impact set (2), or its effects are diffuse and non-additive (5) | Second seed at layer 11; split the top 800 by cross-seed stability and re-run the k-sweep within each half |
| Features are distributed across layers, not redundant | Chance-level structural overlap is not sufficient evidence of distinct mechanisms (3) | Cross-layer transfer: ablate layer L's selected set at layer L'; test for a shared core recovering the span effect |
| Binding features beat random controls | Random is below the field's bar; difference-in-means is the bar (6) | Difference-in-means direction at the same locus and budget as a third arm |
| Nothing calibrates a dictionary ablation against a model-centric ceiling | SAEBench's eight metrics are the checkable version of that claim (7) | No experiment; a citation and a narrowed claim — see §5 |

Four of the seven rows run on the existing 256-item harness. The width and seed rows need GPU time.

## 4. The open question to attach to

**When does a feature-ablation effect license a mechanistic claim?**

The field has built an external standard for every use of SAEs except the one that carries the
strongest claims. Probing has one and SAEs lose (Kantamneni et al.); steering and concept detection
have one and SAEs lose (AxBench); reconstruction and disentanglement have one (SAEBench). Ablation
has none, and two 2026 results say a selected feature set is one draw from an equivalence class —
across seeds (2) and across circuit extractions (3) — which makes an uncalibrated ablation effect
uninterpretable rather than merely imprecise. R is a candidate standard; Miller et al. (1) supplies
its validity condition.

*Evidence class:* five peer-reviewed papers (COLM 2024, ICLR 2025, ICML 2025 ×3) and three
preprints from labs with an established record (UKP ×2, T-Tech). Round 1's recommended framing
rested on three unreviewed preprints, one of which is now rejected at gate 1.

*What a reviewer will demand that we lack:* the ablation-axis match (row 1), a seed replicate, and
a non-dictionary baseline arm.

## 5. The novelty claim, adjudicated

Round 1's claim: *nothing calibrates a sparse-dictionary ablation against a model-centric causal
ceiling at a matched locus, on the same samples, with the ratio reported.* Round 1 tested it only
against the VLM literature.

**It narrows, and survives narrowed.** Reporting a recovered fraction is routine in circuit work —
faithfulness is defined as (m(C) − m(∅)) / (m(M) − m(∅)) and normalises against the empty circuit
and the full model. So "we report a ratio" is not the contribution. What is unclaimed is the
*denominator*: an intervention that severs an identified pathway at the same locus, rather than
deleting everything or nothing. And Miller et al. adds a condition round 1 did not state: the ratio
is interpretable only when numerator and denominator share an ablation axis, which ours currently
do not.

Recommended restatement: **R is a pathway-calibrated faithfulness measure, and the ablation-axis
match is its validity condition.** That is narrower than round 1's claim, defensible against the
circuit literature, and it makes row 1 of §3 a contribution rather than a correction.

Queries that tested it, run 2026-09-21:

- `compare sparse autoencoder feature ablation against attention ablation same locus recovered fraction ceiling language model`
- `normalize sparse autoencoder feature ablation effect by attention head ablation effect ratio same layer "how much of" causal effect explained by features`
- `circuit completeness how much of model behavior recovered fraction of effect sparse feature circuit faithfulness upper bound`
- `SAEBench causal evaluation unlearning metric compare feature ablation to component ablation upper bound sparse probing ceiling`
- `interpretability intervention needs external upper bound baseline "what fraction" mechanism sparse features versus severing pathway 2026`
- `attention knockout ceiling compare sparse autoencoder ablation vision language model calibrate intervention 2026`

Nearest hits: SAEBench (probe-referenced, no model-centric ceiling) and Marks et al. 2024
(faithfulness against the empty circuit, already cited). No hit pairs a dictionary ablation with a
pathway-severing denominator.

One further fact bears on novelty. The only peer-reviewed VLM-plus-SAE work found in either round —
Pach, Karthik, Bouniot, Belongie & Akata, NeurIPS 2025 — operates on CLIP's **vision encoder**. The
decoder's `attn_out` at post-image text positions has no peer-reviewed dictionary work at all.

## Appendix — screened and rejected

A rejection record, not review coverage. 34 candidates entered stage 1. Nine are in
`related-work.bib` — the eight reported above plus `kantamneni2025saeprobing`, covered inside
entry 6. The 25 kills by gate: gate 1 — 5, gate 2 — 1, gate 3 — 2, gate 4 — 17. Gate-4 kills fall
across stages 2 and 3; the eleven candidates close-read in full at stage 3 are entries 1–8 above
plus 2606.18322, 2607.20596 and 2510.08510.

| Candidate | Gate | Why |
|---|---|---|
| Korznikov et al., *Sanity Checks for SAEs* (2602.14111) | 1 | No institutional affiliation. Round 1's anchor. Its claim is now available peer-reviewed from AxBench and Kantamneni et al. |
| *Sparse Visual Thought Circuits in VLMs* (2603.25075) | 1 | Single author (Yunpeng Zhou), no affiliation, unreviewed. Round 1 called this "the closest design neighbour" and rested the novelty argument on its admission of having no model-centric baseline. |
| *SAE Interventions are Unreliable* (2606.18322) | 1 | No affiliations on the listing. Its distinction — causal usefulness vs intervention completeness — is ours, and 95.8% post-intervention recovery would be a strong result. Unusable until reviewed; worth re-checking. |
| *Explanation Multiplicity* (2608.13754) | 1 | Single author, Hochschule Trier, unreviewed. 73.2% of claim pairs flip across 15,840 pre-registered specifications — the same class of claim as entry 1, without review. |
| *Hierarchical Sparse Circuit Extraction* (2601.12879) | 1 | Authors not identifiable to an institution; no venue. |
| *Are Single-Token SAE Features Causally Necessary?* (2607.20596) | 4 | Cite-only. Family-over-scale supports "LLaVA-1.6 is the family-fixed replication", but names no experiment. Round 1 made it load-bearing. |
| Testa, Wong, Lenci, Magnini & Gatt (2609.05149) | 2 | Video. Carried from round 1. |
| Takishita et al. (2506.05439) | 3 | Superseded; CLIP-style models, Sept 2025. Carried from round 1. |
| Makelov, Lange & Nanda (2311.17030) | 3, 4 | NeurIPS 2023 *workshop*, Dec 2023. The subspace-patching illusion is now standard practice rather than a live threat. |
| Hanna, Pezzelle & Belinkov (2403.17806), COLM 2024 | 4 | Recommends behavioural faithfulness over circuit overlap. We already measure behaviourally. Corroboration. |
| Heap, Lawson, Farnik & Aitchison (2501.17727) | 4 | Automated interpretability metrics do not distinguish trained from random transformers. Our metric is causal, not automated-interpretability. A published mechanism that could explain our symptom is not an objection answered. |
| Luo et al., *To Sink or Not to Sink* (2510.08510), ICLR 2026 | 4 | Peer-reviewed and in-modality, but studies ViT-side attention sinks and proposes accuracy improvements. No claim of ours at risk. |
| Pach, Karthik, Bouniot, Belongie & Akata (2504.02821), NeurIPS 2025 | 4 | CLIP vision encoder, not the decoder. Cited in §5 for the novelty argument. |
| *Cascaded Sparse Autoencoders* (2606.16193) | 4 | Dictionary architecture on our model set. Improves a tool we do not claim to improve. |
| Kaduri, Bagon & Dekel (2411.17491), CVPR 2025 | 4 | Corroborates knockout claims we already hold. Carried from round 1. |
| Nikankin, Arad, Gandelsman & Belinkov (2506.09047) | 4 | Sharpens the yardstick, not the contribution. Carried from round 1. |
| Wang et al. (2508.16929), ICML 2026 | 4 | Dimensional collapse as a cause of dead attention-output features is a hypothesis to test against layer 0's 74.2%, not an objection answered. Carried from round 1. |
| *Where Does Vision Meet Language?* (2601.08151); *Visual Representations inside the LM* (2510.04819); *How Vision Becomes Language* (2602.15580); Basu et al. (2406.04236); *LLaVA in VQA* (2411.10950) | 4 | In-modality flow work. Localisation results we replicate rather than contest; none names an experiment. |
| *Beyond the Hard Budget* (2606.27321); *SoftSAE* (2605.06610); *Size Doesn't Matter* (2606.15054) | 4 | SAE sparsity-budget architecture. The k in their budget is the dictionary's L0; the k in ours is the size of an ablated set. Different quantity. |
