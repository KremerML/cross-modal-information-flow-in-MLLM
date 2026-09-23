# Research Update: Cross-Modal Information Flow in MLLMs

**Date:** May 3, 2026
**Period:** April 7 – May 3, 2026
**Branch:** `LLaVA-Instruct-150K`
**Model:** LLaVA-v1.5-7b

---

## 1. CLEVR-Lite Dataset Scale-Up

- Scaled from 1,000 train + 100 val images to **50,000 scenes** (PIL-rendered 224×224 PNGs, 6 colors × 3 shapes)
- Final dataset: **186,638 train / 7,790 val questions** (~3.7 questions per scene)
- Yields ~1.1M question-token activation vectors for SAE training
- Open-ended queries ("What color is the triangle?") — eliminates the language-side bypass that diluted GQA ChooseAttr effects by 25–50%

## 2. CLEVR-Lite Knockout Sweep (Complete)

- **Full 32-layer Image→Question knockout sweep**, n=7,084, ~33 hours runtime
- Checkpoint-resumable execution via JSONL write per sample + `os.fsync()` + `try/finally`
- Key results (Cohen's d / margin drop):

| Layer | d | Margin Drop | Role |
|-------|-------|-------------|------|
| 0 | 1.04 | 0.46 | Early embedding transfer |
| 10 | 1.14 | 0.27 | Mid-layer cluster onset |
| 11 | 0.91 | 0.43 | Confirmed from GQA (much stronger here) |
| 12 | 0.76 | 0.25 | Cluster member |
| 13 | −0.36 | — | **Inhibitory** (flanked by positives) |
| 14 | 1.21 | 0.40 | **Strongest overall** |
| 29 | −1.22 | — | Strong late-layer inhibition |

- **Three-region structure** identified: early (layer 0), mid-cluster (10–14 with inhibitory 13), late inhibitory (29)
- Image→Last flow much stronger than GQA (max d=1.15 at layer 2)
- Effect sizes 25–50% larger than GQA across the board

## 3. SAE Multi-Layer Training

- Created **6 CLEVR-Lite SAE configs** (`configs/clevr_lite/sae_layer{0,10,11,12,13,14}_attn_out_question.yaml`): 32,768 features, `position_type=question`
- **3 SAEs fully trained** (layers 0, 10, 11) on ~4.9M activation rows each:

| Layer | Explained Var. | Mean L0 | Dead Features |
|-------|---------------|---------|---------------|
| 0 | 99.98% | 1,416 | 74.2% |
| 10 | 99.89% | 1,128 | 0.2% |
| 11 | 99.85% | 1,180 | 0.4% |

- Layer 0's high dead-feature fraction (74%) suggests early-layer representations are low-rank
- **Layer 12 activation collection complete** (186,638 samples, 38 chunks on disk); SAE training in progress but pipeline interrupted
- Layers 13, 14 not yet started

## 4. Feature Identification v2: Gradient-Based Causal Scoring (Breakthrough)

### Motivation
- **18 consecutive null experiments** with v1 ratio-based features across multiple layers, position types, ablation modes, and both GQA and CLEVR-Lite datasets
- v1 selected "ghost features" — epsilon-inflated artifacts with 1e-6 magnitude activations (88% had `incorrect_mean = 0`)

### v2 Method
- Insert SAE into forward pass as a differentiable hook
- `detach().requires_grad_(True)` on feature activations to isolate gradient signal
- Compute `d(logit_margin)/d(feature_activation)` via backpropagation
- Causal score = `|grad| × |activation|` averaged over all samples and question positions
- Based on Sparse Feature Circuits (Marks et al., ICLR 2025) and Input vs Output Features (Agrawal et al., 2025)

### v1 vs v2 Comparison (Layer 11)

| Property | v1 (ratio) | v2 (causal) |
|----------|-----------|-------------|
| Feature overlap | — | **0 / 200 (0%)** |
| Activation range | 6e-7 to 1e-5 | 0.087 to 0.274 |
| Mean activation | 1.16e-6 | 0.135 |
| Ratio | — | **116,000× larger** |

### v2 Ablation Result (Layer 11, n=256)

| Metric | Binding (v2) | Random (15 sets) | z-score |
|--------|-------------|------------------|---------|
| Margin drop | **+0.213** | −0.0004 | **81.6** |
| Accuracy drop | +0.016 | −0.0008 | — |
| Relative perturbation | 0.019 | 0.002 | — |

- 217/256 samples (84.8%) showed positive margin drops
- 4 predictions flipped correct→wrong (1.6%)
- **Captures ~50% of the knockout ceiling** at layer 11 (0.213 vs 0.43)
- 200 features out of 32,768 (0.6%) carry half the cross-modal signal at this layer

### New Files
- `sae_experiments/feature_analysis/causal_feature_identifier.py` — core v2 implementation
- `sae_experiments/scripts/02b_identify_features_causal.py` — runner script
- `sae_experiments/scripts/analyze_causal_features.py` — v1/v2 post-hoc comparison
- `sae_experiments/scripts/feature_id_v2_analysis.md` — methodology writeup
- `sae_experiments/scripts/feature_id_v2_breakthrough_report.md` — full results report

## 5. Infrastructure and Pipeline Improvements

- **Activation collection with checkpointing**: chunked writes to disk (float16), manifest-based resume, checkpoint interval configurable — enables collection of 4.9M+ rows without holding all in RAM
- **Multi-layer activation collector** (`collect_activations.py`): collects activations for multiple layers in a single forward pass using numpy memory-mapped files; peak RSS ≈ model weights + one batch
- **SAE training script improvements**: activation caching to disk, model release before training (`del model; torch.cuda.empty_cache()`), chunk reassembly after model unload
- **Full pipeline script** (`run_full_pipeline.sh`): two-phase (train SAEs → causal feature ID) with per-layer skip logic for resumability
- Shell scripts for CLEVR-Lite activation collection and SAE training (`scripts/`)
- `.gitignore` cleanup: untracked `.pyc` files, untracked large datasets and output files
- Docs relocated from repo root to `docs/`

## 6. Uncommitted Work in Progress

- `activation_collector.py`: checkpoint/resume support with chunked disk writes and float16 storage
- `feature_identifier.py`: refactoring (minor)
- `sae_trainer.py`: integration with chunked activation cache
- `01_train_sae.py`: activation caching, model release before training, chunk reassembly

---

## 7. Next Steps

1. **Complete multi-layer SAE training** — finish layer 12 (activation collection done, training interrupted); train layers 13, 14
2. **Run v2 causal feature ID on all knockout-identified layers** (0, 10, 12, 14) — currently only layer 11 is done
3. **Full-dataset ablation** — scale from n=256 subsample to full n=7,790 validation set
4. **Multi-layer simultaneous ablation** — ablate v2 features at multiple layers jointly to close the gap toward the knockout ceiling
5. **Feature interpretability** — characterize what the top causal features encode (color detectors? shape detectors? binding operators?)
6. **Vary top-k** — test ablation with top-50, top-100, top-500 features to map the marginal contribution curve

## 8. Answers to Questions from Previous Sessions

### Why collect activations at question-token positions, not image-token positions?

**Short answer:** We are studying cross-modal *transfer*, not visual *encoding*. The transferred signal only exists at question positions.

**Architectural constraint (causal mask).** LLaVA is decoder-only with image tokens prepended to text. The causal attention mask means image tokens can only attend to other image tokens — they never receive information from the question. Cross-modal information flow is therefore unidirectional (image → question), and the only place the transferred signal exists is at question (or later) positions.

**`attn_out` at question positions IS the cross-modal signal.** The attention output at question position q is:

```
attn_out[q] = Σ_k attention_weight[q, k] · V[k]
```

When q attends to image tokens, its attention output contains the visual information that was *selected and routed* to answer this specific question. SAE features at question positions decompose this transferred signal — visual information in the format the language head can use.

**Image-position features answer a different (weaker) question.** Ablating features at image positions asks "which visual representations, if corrupted at the source, hurt performance?" This is confounded because:

1. A corrupted image feature affects all 32 downstream layers that attend to it — layer-specific transfer cannot be isolated
2. It mixes source encoding with transfer (is the feature important because it encodes something, or because it gets read?)
3. It tells us nothing about binding — just about which visual patterns exist in the image

**The v2 gradient method reinforces the choice.** We compute d(logit_margin)/d(feature_activation). Features at question positions sit downstream of image→question attention and upstream of the answer logits — exactly on the causal path we are isolating. Features at image positions sit upstream of the attention operation itself, so their gradients would capture "importance of this visual feature to the entire rest of the model" rather than "importance of this specific cross-modal transfer at layer L."

**Analogy.** To study how a radio signal reaches a receiver, you analyze the signal at the antenna (destination), not at the transmitter (source). The transmitter broadcasts everything; what matters is what gets through and gets decoded.

**Note:** This is distinct from the earlier `attribute` vs `question` position-type decision. The `attribute` position type (tokens spanning the attribute word in the question) was abandoned because the full-latent ceiling showed only 0.3% relative perturbation there. The `question` position type (all question tokens) captures the full set of positions where cross-modal information arrives via attention.

---

## 9. Current Limitations

- **Single-layer ablation ceiling**: the model can re-read image tokens at layers L+1 through 31, explaining the ~50% gap between SAE ablation and full knockout
- **Layer 11 only**: v2 causal features validated at one layer; the pattern may differ at layers 0 (early embedding) and 14 (strongest knockout site)
- **n=256 subsample**: ablation effect is strong (z=81.6) but statistical confidence on accuracy drop and flip rate is limited at this sample size
- **SAE coverage**: only `attn_out` is modeled; the MLP pathway and residual bypass at each layer are not captured
- **Layer 0 SAE quality**: 74% dead features suggest the 32,768-feature architecture may be overparameterized for early-layer representations
- **Pipeline fragility**: layer 12 training interrupted twice (May 2–3); multi-hour GPU runs need more robust scheduling
- **No semantic labeling**: top causal features have not been mapped to interpretable concepts yet
