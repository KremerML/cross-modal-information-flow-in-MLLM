# Breaking the Null Result: From Correlational to Causal SAE Feature Identification

**Date:** May 2026
**Layer:** 11, attn_out, question positions
**Dataset:** CLEVR-Lite (7,790 val samples for feature ID; 256-sample ablation)
**SAE:** 32,768 features, trained on ~1.1M question-token activation vectors

---

## 1. The Problem: 18 Consecutive Null Experiments

Every ablation experiment using v1 ratio-based feature identification produced the same result: ablating the top-200 "binding features" had no measurable effect on model predictions. The binding features performed identically to random feature sets. This held across:

- Multiple layers (0, 11)
- Multiple position types (attribute, question, image, all)
- Multiple ablation modes (zero, mean, replace)
- Both GQA ChooseAttr and CLEVR-Lite datasets

The conclusion was that the feature identification method was selecting the wrong features — features that were statistically associated with correctness but had no causal role in the model's computation.

## 2. The Old Method (v1): Activation Ratio

### Algorithm

The v1 method (`feature_identifier.py`, `find_discriminative_features()`) works as follows:

1. **Collect activations**: Run all validation samples through the model, hook into layer 11's attention output, extract hidden states at question token positions.

2. **Encode through SAE**: Pass activations through the sparse autoencoder encoder: `z = ReLU(W_enc @ (h - b_pre))`. Mean-pool feature activations across token positions per sample, yielding a `(n_samples, 32768)` matrix.

3. **Label correctness**: For each sample, compare the model's predicted answer to the ground truth. Split samples into correct (90.9%) and incorrect (9.1%) groups.

4. **Compute discrimination ratio**: For each feature i:
   ```
   correct_mean[i]   = mean(z[correct_samples, i])
   incorrect_mean[i] = mean(z[incorrect_samples, i])
   ratio[i] = (correct_mean[i] + 1e-8) / (incorrect_mean[i] + 1e-8)
   ```

5. **Select top-200 by ratio**: Features with the highest ratio are designated "binding features" and passed to the ablation pipeline.

### Why It Fails

#### Failure 1: Epsilon-Inflated Ghost Features

The ratio metric is catastrophically unstable when the denominator is near zero. In practice, 88% of the selected features (176 out of 200) had `incorrect_mean = 0`. For these features, the ratio becomes:

```
ratio = (correct_mean + 1e-8) / (0 + 1e-8) ≈ correct_mean / 1e-8
```

This means any feature with even a trace activation on correct samples — no matter how tiny — gets an enormous ratio. The top-ranked feature (feature 6588, ratio = 353.6) had:

```
correct_mean   = 1.22e-05
incorrect_mean = 2.46e-08
```

A `correct_mean` of 1.22e-05 is five orders of magnitude below the residual stream norm (~5.5). Ablating this feature removes a perturbation of magnitude ~0.01% of the signal. The model cannot detect the difference.

**Top-10 v1 features and their activation magnitudes:**

| Rank | Feature | Ratio | correct_mean | incorrect_mean |
|------|---------|-------|--------------|----------------|
| 1 | 6588 | 353.6 | 1.22e-05 | 2.46e-08 |
| 2 | 17575 | 317.0 | 6.33e-06 | 6.44e-09 |
| 3 | 25593 | 292.6 | 5.84e-06 | 8.62e-09 |
| 4 | 12640 | 267.9 | 5.35e-06 | 0 |
| 5 | 2257 | 264.2 | 5.27e-06 | 0 |
| 6 | 21211 | 173.6 | 3.46e-06 | 0 |
| 7 | 30836 | 165.2 | 3.29e-06 | 0 |
| 8 | 17259 | 153.7 | 3.06e-06 | 0 |
| 9 | 10702 | 146.9 | 2.93e-06 | 0 |
| 10 | 4277 | 145.5 | 2.90e-06 | 0 |

These features are essentially dead — they fire on fewer than 1% of samples with negligible magnitude. The ratio metric treats them as highly discriminative because a near-zero numerator divided by an exactly-zero denominator yields a large number.

**Summary statistics across all 200 v1 features:**
- Activation range: 6.12e-07 to 1.22e-05
- Mean activation: 1.16e-06
- Features with `incorrect_mean = 0`: 176/200 (88.0%)

#### Failure 2: Correlational, Not Causal

Even if the ratio metric worked correctly (i.e., selected features with substantial activation magnitudes), it would still select the wrong features. The ratio measures statistical association between feature activation and prediction correctness. Following the framework of Agrawal et al. (2025, arXiv:2505.20063), this selects "input features" — features that detect input patterns correlated with easy/hard samples — rather than "output features" that causally drive the model's predictions.

A feature that fires on blue objects (which the model happens to classify correctly) would get a high ratio, but ablating it doesn't change the output because the model uses other pathways to determine "blue."

#### Failure 3: Noisy Denominator from Class Imbalance

With 90.9% accuracy (7,084 correct vs 706 incorrect), the "incorrect" group is 10x smaller. For sparse features that fire on <1% of samples, the mean over 706 noisy negatives is unreliable. This makes the denominator of the ratio essentially random for rare features, injecting noise into the rankings.

## 3. The New Method (v2): Gradient-Based Causal Scoring

### Algorithm

The v2 method (`causal_feature_identifier.py`, `compute_causal_scores()`) replaces correlational selection with causal attribution via backpropagation:

1. **Insert SAE into the computation graph**: Register a forward hook at layer 11's attention output. The hook:
   - Encodes the activation: `z_raw = SAE.encode(h)`
   - Detaches z from the encoder's computation graph and creates a new leaf tensor: `z = z_raw.detach().requires_grad_(True)`
   - Decodes back: `h' = SAE.decode(z)`
   - Replaces the layer output with h'

   The critical step is `detach().requires_grad_(True)`. This makes z a leaf variable in the computation graph — gradients flow from the output through the decoder, through z, but stop there. This isolates the gradient signal to "how much does each feature affect the output through the downstream layers."

2. **Forward pass**: Run the model with `torch.enable_grad()` (not the default `torch.no_grad()` used for inference). The model's parameters remain frozen (`requires_grad_(False)`), but z is differentiable.

3. **Compute objective**: At the last token position, compute:
   ```
   objective = logit(true_answer) - logit(false_answer)
   ```
   This is the logit margin — how much the model prefers the correct answer over the distractor.

4. **Backpropagate**: `objective.backward()` computes `d(objective)/d(z)` — the gradient of the output margin with respect to each feature activation at each position.

5. **Score features**: For each sample, compute:
   ```
   causal_score[i] = mean_over_positions(|grad[i]| * |z[i]|)
   ```
   Then average across all samples. The product `|grad| * |activation|` is a first-order Taylor approximation of each feature's contribution to the output.

6. **Select top-200 by causal score**.

### Why It Works

#### Reason 1: Selects Features That Actually Affect the Output

The gradient `d(margin)/d(z_i)` directly measures how sensitive the model's prediction is to changes in feature i. If ablating feature i would decrease the margin, its gradient will be large. Features with large `|grad| * |activation|` are those where (a) the model is sensitive to the feature, and (b) the feature is actually active. Both conditions must hold.

This is the key difference from v1: the ratio metric has no concept of "does changing this feature change the output." It only asks "is this feature more active on correct samples?" — a fundamentally different (and weaker) question.

#### Reason 2: Naturally Filters Ghost Features

Features with near-zero activation automatically get near-zero causal scores, regardless of their gradient. The product `|grad| * |activation|` ensures that a feature must have meaningful magnitude to be selected. This eliminates the entire class of epsilon-inflated ghost features that dominated v1.

#### Reason 3: No Correct/Incorrect Split Required

The v2 method does not partition samples into correct and incorrect groups. It computes gradients on every sample independently. This avoids the class imbalance problem (90/10 split) and makes the method robust to task difficulty.

### Theoretical Grounding

- **Sparse Feature Circuits** (Marks et al., ICLR 2025 Oral, arXiv:2403.19647): Introduced integrated gradients through SAE features to discover sparse circuits. Our method uses a single-step gradient (not integrated), which is a first-order approximation sufficient for feature ranking.

- **Input vs Output Features** (Agrawal et al., arXiv:2505.20063): Demonstrated that SAE features decompose into "input features" (detect input patterns) and "output features" (causally drive predictions), with little overlap between them. Ratio-based selection captures input features; gradient-based selection captures output features.

## 4. Results: v1 vs v2 Head-to-Head

### Feature Set Comparison

| Property | v1 (ratio) | v2 (causal) |
|----------|-----------|-------------|
| Selection metric | correct_mean / incorrect_mean | \|grad\| * \|activation\| |
| Number of features | 200 | 200 |
| **Overlap** | — | **0 features (0%)** |
| Activation range | 6.1e-07 to 1.2e-05 | 0.086 to 0.274 |
| Mean activation | 1.16e-06 | 0.135 |
| **Activation ratio** | — | **115,564x larger** |
| Features with zero incorrect_mean | 176/200 (88%) | N/A |

Zero overlap between the two feature sets. The methods select entirely disjoint subsets of the 32,768 features. This confirms that v1 was selecting a fundamentally different (and wrong) class of features.

The activation magnitude difference is the most telling diagnostic: v1 features have activations at 1e-6, five orders of magnitude below the residual stream norm (~5.5). Ablating them perturbs the residual stream by ~0.0001%. v2 features have activations at 0.1–0.3, constituting a measurable fraction of the signal.

### Top-10 v2 Features

| Rank | Feature | Causal Score | Activation | Gradient |
|------|---------|-------------|------------|----------|
| 1 | 17706 | 0.00519 | 0.256 | 0.00765 |
| 2 | 6011 | 0.00456 | 0.274 | 0.00691 |
| 3 | 11282 | 0.00328 | 0.194 | 0.00719 |
| 4 | 5490 | 0.00251 | 0.163 | 0.00664 |
| 5 | 79 | 0.00249 | 0.187 | 0.00614 |
| 6 | 28068 | 0.00245 | 0.161 | 0.00637 |
| 7 | 31197 | 0.00237 | 0.157 | 0.00671 |
| 8 | 31516 | 0.00237 | 0.165 | 0.00639 |
| 9 | 21514 | 0.00232 | 0.170 | 0.00623 |
| 10 | 3632 | 0.00220 | 0.168 | 0.00588 |

Note: the gradient magnitudes are relatively uniform (~0.006–0.008) across top features, while activation magnitudes vary more (0.16–0.27). This suggests that in this setting, the causal score is primarily driven by which features are active (have large |z|) rather than which features the model is most sensitive to (large |grad|). The gradient acts as a filter — confirming that the feature participates in the output computation — while the activation magnitude determines the ranking.

### Causal Score Distribution (All 32,768 Features)

| Statistic | Causal Score | Activation Mean | Gradient Mean |
|-----------|-------------|-----------------|---------------|
| p50 | 1.66e-10 | 6.07e-08 | 0.00286 |
| p90 | 2.73e-09 | 6.14e-07 | 0.00322 |
| p95 | 2.65e-08 | 1.93e-05 | 0.00336 |
| p99 | 6.34e-04 | 0.101 | 0.00385 |
| max | 5.19e-03 | 0.399 | 0.00765 |

The causal score distribution is extremely sparse: the median is 1.66e-10, while the max is 0.00519 — a span of ~7 orders of magnitude. This confirms that only a small number of features carry meaningful causal signal, consistent with the sparse feature circuits framework.

The gradient distribution is notably flat (p50=0.00286, max=0.00765, a ~2.7x range), confirming that gradient magnitude alone does not discriminate well between features. It is the product with activation that creates the sharp separation.

### Ablation Results

**v2 causal features (n=256 samples, layer 11, attn_out, question positions, replace mode):**

| Metric | Binding (v2 causal, 200 features) | Random (mean of 15 sets, 200 features each) |
|--------|-----------------------------------|---------------------------------------------|
| Accuracy drop | +0.0156 (1.6%) | −0.0008 |
| **Margin drop** | **+0.2131** | −0.0004 |
| Relative perturbation | 0.0186 | 0.0019 |

- **z-score: 81.6** (binding margin_drop vs random distribution)
- 217 / 256 samples (84.8%) showed positive margin drops
- 4 predictions flipped correct → incorrect (1.6%)
- Empirical p-value: 0/15 random sets exceeded binding effect (p < 0.067)

For comparison, knockout at layer 11 (zeroing all Image→Question attention) produces a margin drop of 0.43. The v2 SAE ablation captures **49.6% of the knockout ceiling** at this layer.

**v1 ratio-based features** produced null results across all prior experiments (exact match with random baseline, margin drops indistinguishable from zero).

## 5. Interpretation

### What the result means

200 out of 32,768 SAE features (0.6%) at layer 11 carry approximately half the cross-modal information that knockout removes at this layer. This demonstrates that:

1. **Cross-modal information flow is sparse**: The visual information used for attribute binding is not diffusely spread across thousands of features. A small, identifiable set does the computational work.

2. **The SAE decomposition is meaningful**: The sparse autoencoder successfully decomposes the residual stream into causally active units. The features it learns correspond to functional components of the model's computation, not just statistical patterns.

3. **Gradient-based attribution works for feature selection**: The first-order Taylor approximation (|grad| * |activation|) is sufficient to identify causally important features. This validates the Sparse Feature Circuits framework (Marks et al. 2024) in the multimodal setting.

### What the 50% gap means

The remaining ~50% gap between SAE ablation (0.213) and knockout (0.43) is expected and informative:

- **Knockout is a stronger intervention**: It zeros ALL attention from image to question at layer 11, across the full residual stream. SAE ablation only modifies 200 specific features in the attention output at question positions.
- **More features beyond top-200**: The causal score distribution has a long tail. Features 201–500 likely carry additional signal.
- **MLP pathway**: The SAE covers attn_out only, not the MLP or residual bypass at this layer.
- **Reconstruction error**: The SAE explains 99.85% of variance, but 0.15% unexplained variance introduces noise in the ablation.

### What remains unknown

- Whether the same features are important at other knockout-identified layers (0, 10, 12, 14)
- Whether multi-layer ablation can close the gap toward the knockout ceiling
- What these features semantically represent (color detectors? shape detectors? binding operators?)
- Whether the effect scales to the full dataset (n=7,790 vs current n=256)

## 6. Code Changes Summary

### New files

| File | Purpose |
|------|---------|
| `sae_experiments/feature_analysis/causal_feature_identifier.py` | Core v2 implementation: CausalFeatureIdentifier class |
| `sae_experiments/scripts/02b_identify_features_causal.py` | Runner script (same config format as v1's 02_identify_features.py) |
| `sae_experiments/scripts/analyze_causal_features.py` | Post-hoc comparison of v1 vs v2 feature sets |
| `sae_experiments/scripts/feature_id_v2_analysis.md` | Methodology writeup and adversarial review |

### Key implementation details

**The SAE hook** (the most critical piece of new code):

```python
def sae_hook(module, inp, output):
    acts = output[0] if isinstance(output, (tuple, list)) else output
    z_raw = sae.encode(acts)
    z = z_raw.detach().requires_grad_(True)  # detach from encoder graph
    z.retain_grad()                          # keep grad after backward()
    features_buffer["z"] = z                 # save for later retrieval
    recon = sae.decode(z)                    # decode back to residual stream
    # replace the layer output with the reconstruction
    return recon
```

The `detach().requires_grad_(True)` pattern is the key insight. By detaching z from the encoder's computation graph and making it a new leaf tensor, we ensure that:
- Gradients flow from the output through the decoder, through z
- Gradients do NOT flow through the encoder (we don't want to know how h maps to z — we want to know how z maps to the output)
- z.grad gives us exactly d(output)/d(feature_activation)

**Forward pass with gradient tracking:**

```python
with torch.enable_grad():  # override model's default no_grad inference
    outputs = model(input_ids=input_ids, images=image_tensor, ...)
    logits = outputs.logits[0]
    objective = logits[-1, true_id] - logits[-1, false_id]  # margin
    objective.backward()  # MUST be inside enable_grad block
```

### Compatibility

The v2 pipeline outputs a `causal_feature_catalog.json` in the same format as v1's `feature_catalog.json`, so the existing ablation pipeline (`03_run_ablation.py`) works without modification — just pass `--features path/to/causal_feature_catalog.json`.
