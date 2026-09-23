# Literature review guidelines

Written 2026-09-21, after reviewing the 2026-09-14/15 pass (`docs/related-work.md`). That pass
triaged 138 candidates by topical adjacency and citation count. Both are the wrong sort. This file
records the bar a candidate has to clear before it earns space in the review, and the process rules
that stop the failure modes that pass exhibited.

## The four gates

A candidate must pass **all four**. Failing any one is a rejection, not a demotion.

| Gate | The question | Rejects |
|---|---|---|
| 1. Reputable | Peer-reviewed venue, or a preprint from an identifiable lab with institutional affiliations and a track record? | Unreviewed preprints with no stated affiliation |
| 2. In scope | Same modality, model family and intervention target as ours? | Video, encoder-only, adjacent-but-different settings |
| 3. Current | Still the state of the art on its own question, or superseded? | Work overtaken by newer results, however sound at the time |
| 4. Informative | Does it change the framing, a claim's risk, or an experiment we would run? | Corroboration of what we already know |

Gates 1 and 3 are independent: peer review is necessary, not sufficient. Gate 4 is judged against
**our contribution**, not our method family — see below.

---

## Gate 1 — Reputable

Record venue, author affiliations and peer-review status **at collection time**, in the bib entry.
A candidate with no author field is not a candidate.

**Worked rejection: Korznikov et al., *Sanity Checks for Sparse Autoencoders* (2602.14111).**
The previous pass made this "the stakes-raiser" — the anchor of §1 and the evidentiary base of the
recommended Q1 framing. It is an unreviewed February 2026 preprint whose authors list `@gmail.com`
addresses and no institutional affiliation. Its headline claim (random baselines beat trained SAEs
on causal editing) is exactly the kind of result that needs review before it can carry a framing.

**The compounding error to avoid.** Q1 in §6 rested on 2602.14111, 2607.12166 (single author,
self-declared medium confidence) and 2607.20596 — all three unreviewed. A recommendation is only as
strong as the weakest paper it stands on, and a stack of three weak papers is not a strong one.
**State, at each recommendation, what class of evidence it rests on.**

## Gate 2 — In scope

Modality, model family and intervention target must match. Topic-level adjacency is not scope.

**Worked rejection: Testa, Wong, Lenci, Magnini & Gatt (2609.05149).** EMNLP 2026, established
authors, and a title squarely on our subject — causal information flow in multimodal
decision-making. It studies **video**. Out of scope, and the strong venue and authorship make it
more dangerous rather than less, because it survives a gate-1 check unexamined.

Note how it nearly got through: it sat in `related-work.bib` with no cluster, no edges and no
mention anywhere in the prose. **Orphans are the route by which out-of-scope work reaches the
citation stage.** See process rule 2.

## Gate 3 — Current

A paper can be peer-reviewed, well-authored, methodologically sound and still not evidence about
the systems we study.

**Worked rejection: Takishita, Gala, Mohamed, Inui & Kementchedjhieva (2506.05439).** Findings of
EMNLP 2025, Inui's group, controlled self-attention ablations, directly on the compensation question
that threatens our no-downstream-re-reading claim. Clears gate 1 comfortably. But it is September
2025 and studies CLIP-style models, and has been superseded on its own question. Cite it as
lineage if at all; do not present it as a live threat.

## Gate 4 — Informative *for our positioning*

The test is whether the paper changes something: the framing, the risk attached to a specific claim,
an experiment we would run, or a citation a reviewer would fault us for missing. Three ways a good
paper fails this gate:

**It is about our yardstick, not our contribution.** *Nikankin, Arad, Gandelsman & Belinkov, Same
Task, Different Circuits* (2506.09047) is a very good paper — reputable lab, relevant, the
highest-degree node in the citation graph. It is also an attention-knockout and circuits paper, and
**we are not an attention-knockout study**. The knockout is our external yardstick; the contribution
is the dictionary ablation calibrated against it. A paper that sharpens knockout methodology adds
nothing to where we sit. Cite it, do not report it.

**It corroborates what we already know.** *Kaduri, Bagon & Dekel, What's in the Image?*
(2411.17491) is CVPR 2025, well cited, and independently supports both knockout claims. It is not
news. Our knockout methodology follows Zhang et al., whose work we know intimately, and
corroboration of an established result is a bib entry, not a finding. The previous pass ranked it
"the most conspicuous omission" on citation count alone — which is a measure of a paper's standing,
never of its value to us.

**Its mechanism is not established as ours.** *Wang et al., Dimensional Collapse in Transformer
Attention Outputs* (2508.16929) is ICML 2026 and reports dead features in attention-output SAEs
dropping from 87% to below 1% under Active Subspace Initialization. The previous pass wrote this up
as answering objection O6 and converting our layer-0 dead dictionary "from a dead end into a next
experiment." We have no evidence that the cause they identify is the cause of our 74.2% dead rate.
**A published mechanism that could explain our symptom is a hypothesis to test, not an objection
answered.** Write it up that way or not at all.

---

## The positive example: Salazar, Frank, Oneata, Elliott & Fierro (2607.03358)

*Pathways of Visual Information Flow in Vision-Language Models*, July 2026, Copenhagen.

| Gate | Why it passes |
|---|---|
| Reputable | Identifiable group, institutional affiliations, methodology that holds up on inspection |
| In scope | Image-text VLMs, attention knockout plus patching, controlled synthetic and natural data — our setting |
| Current | July 2026, nothing between it and now on its question |
| Informative | Names a mechanism that threatens a specific claim of ours and implies a specific experiment |

What a gate-4 pass looks like as output — not a summary, but a consequence:

| Our claim at risk | What the paper implies | What we run |
|---|---|---|
| Downstream re-reading rejected on four tests | Their direct/text-mediated routes are our `Image->Last`/`Image->Question`, and models fall back on one when the other is ablated. Our four tests cover compensation *within* a pathway, not switching *between* them. | `Image->Last` knockout *under* `Image->Question` ablation, shared 256 items |
| LLaVA-1.6 R = 194.5% at {13,14} | Pathway substitution explains A exceeding K with no dictionary pathology | Same experiment |

One paper, one threatened claim, one experiment, cheap on the existing harness. That is the unit of
output the review should produce. A paper that cannot be written into that table did not pass gate 4.

---

## Process rules for the next round

1. **Metadata at collection time.** Venue, affiliations, peer-review status and date in the bib
   entry when the candidate is entered. Nine of the twenty-two SAE-side entries last round had no
   author field, so gate 1 could not be applied to them at all.
2. **No orphans.** In the bib implies written up in the prose implies at least one edge in the
   graph. Last round had three bib entries with no mention in the prose.
3. **No mention-only tails.** Eleven IDs appeared only inside comma-separated "Also:" lists in the
   C2 and C9 sections and were never close-read. Close-read it or drop it; a list of identifiers is
   not review coverage.
4. **Cluster by our contribution, not by topic.** C1 (information flow and knockout) is our
   method's ancestry. C2, C8 and C9 (dictionaries, probes, cross-layer feature identity) are where
   we actually live. Effort should follow the second group.
5. **Citation count decides nothing.** It was the stated reason for elevating 2411.17491 and
   2503.03321, and it is a measure of a paper's standing in its own field, not of its bearing on
   ours.
6. **Grade the evidence under every recommendation.** If a proposed framing rests only on
   unreviewed preprints, that belongs in the recommendation, not in a footnote.

## The finding to carry into the next round

Applying these gates to the existing review leaves **one paper standing**: Salazar et al.

That is a structural result, not an accident of filtering. Every reputable paper in the review sits
in C1, the knockout cluster, which is not our contribution. Of the twenty-two entries in C2, C8 and
C9 — where the contribution actually lives — exactly one is peer-reviewed (2504.02821, NeurIPS
2025, April 2025, and on the CLIP vision encoder rather than the decoder). The only other reputable
dictionary work in the list is 2024 foundational method work we already cite as ancestry
(2406.17759, 2406.04093).

So the §3 claim survives: nothing calibrates a sparse-dictionary ablation against a model-centric
causal ceiling at a matched locus on the same samples. But the reason the gap is open is that the
sub-area has not reached peer review. Good for novelty, awkward for framing, and the next round's
first job is to establish which of the two it is — by finding the reputable neighbours, or by
documenting that they do not exist.
