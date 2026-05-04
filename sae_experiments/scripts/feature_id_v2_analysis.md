# Feature Identification v2: Analysis and Redesign

## 1. Current Approach (v1) — Theoretical Summary

Our current feature identification pipeline (`02_identify_features.py`) works as follows:

1. **Collect activations**: Hook into layer L's attention output at question token positions.
   Run all val samples through the model, extract `(n_samples, seq_len, d_model)` activations.

2. **Encode through SAE**: `features = SAE.encode(activations)` → `(n_samples, n_features)`.
   Mean-pool across token positions per sample.

3. **Score correctness**: Use `option_logprob` — compute log P(true_option) vs log P(false_option).
   Label each sample as correct/incorrect based on which logprob is higher.

4. **Compute discrimination ratio**: For each feature i:
   - `correct_mean[i] = mean(features[correct_samples, i])`
   - `incorrect_mean[i] = mean(features[incorrect_samples, i])`
   - `ratio[i] = (correct_mean[i] + 1e-8) / (incorrect_mean[i] + 1e-8)`

5. **Select top-k by ratio**: Features with highest ratio → "binding features".

6. **Ablate**: Zero these features in a forward-pass hook, decode back to residual stream,
   measure whether predictions change.

## 2. Why This Fails — Diagnosis

### Problem 1: Correlational, not causal (fundamental)

The ratio metric measures **statistical association** between feature activation and prediction
correctness. This selects what Agrawal et al. (2025, arXiv:2505.20063) call "input features" —
features that detect input patterns but do not causally drive outputs. Their key finding:
*high input scores and high output scores rarely co-occur in the same feature.* Our top-200
features by ratio are virtually guaranteed to be input features.

### Problem 2: Epsilon-inflated ghost features (metric artifact)

With `incorrect_mean = 0` for 81% of selected features (176/200), the ratio becomes:
`(correct_mean + 1e-8) / (0 + 1e-8 + 1e-8) ≈ correct_mean / 2e-8`

The top feature (ratio=353.6) has `correct_mean = 1.22e-5` — five orders of magnitude below
residual stream norms. Ablating them changes nothing because there is nothing to change.

### Problem 3: Correct/incorrect split is noisy for high-accuracy tasks

With 90.9% accuracy (7084/706 split), the "incorrect" group is 10x smaller and likely
dominated by ambiguous or atypical samples. Mean activation over 706 noisy negatives is
unreliable for sparse features that fire on <1% of samples.

## 3. New Approach (v2) — Gradient-Based Causal Feature Identification

### Method

For each sample, insert the SAE into the model's forward pass and compute the gradient of
the output logit margin w.r.t. each SAE feature activation:

1. Register hook on layer L's attention output module
2. Hook: encode h → features z (detach, requires_grad), decode z → recon h', replace h with h'
3. Forward pass through layers L+1 to end → output logits
4. Compute objective = logit(correct_answer) - logit(distractor)
5. Backpropagate: `grad = d(objective) / d(z)`
6. Causal score = mean over samples of `|grad| * |z|`, averaged over question positions
7. Select top-k by causal score

### Theoretical grounding

- **Sparse Feature Circuits** (Marks et al., ICLR 2025 Oral, arXiv:2403.19647)
- **Input vs Output Features** (Agrawal et al., EMNLP 2025, arXiv:2505.20063)
- **SpARE** (Zhao et al., NAACL 2025 Oral, arXiv:2410.15999)

### Results

**v1 vs v2 comparison (layer 11, attn_out, question positions):**

| Metric | v1 (ratio) | v2 (causal) |
|--------|-----------|-------------|
| Feature overlap | — | **0 / 200 (0%)** |
| Activation range | 6.1e-7 to 1.2e-5 | 0.087 to 0.260 |
| Mean activation | 1.2e-6 | 0.135 |
| Ratio of means | 1x | **116,357x larger** |

The v2 features have meaningful activation magnitudes — they actually contribute to the
residual stream. v1 selected ghost features with negligible signal.

**Stability**: Top-10 features identical between n=50 and n=200 runs.

## 4. Adversarial Review — Key Issues and Resolutions

### Addressed before full run:

1. **b_pre in decode** (Issue 11): b_pre.norm()=2.28, decoder.bias partially compensates
   (cosine=0.75). The SAE was trained with this architecture; 99.85% explained variance is
   measured under these conditions. Not a blocking issue — the gradient computation uses
   the same architecture.

2. **Answer tokenization** (Issue 3): All CLEVR-Lite answers are single-token except "purple"
   (2 tokens, 8.1% of samples). First token "pur" is still discriminative.

3. **Sample stability** (Issue 7): Top-10 features are identical between n=50 and n=200
   independent runs.

### Acknowledged but deferred:

4. **Single-layer ablation limitation** (Issue 10): The gradient identifies features important
   at this layer, but the model can re-read image tokens at layers 12-14. This affects the
   ablation experiment, not the feature identification. Multi-layer ablation should be the
   next step if single-layer ablation still produces null results with v2 features.

5. **Position-selective hook** (Issue 12): Current hook replaces all positions; ablation only
   modifies question positions. The mismatch adds noise but doesn't change feature rankings
   meaningfully (reconstruction is 99.85% accurate).

6. **Logit scale bias** (Issue 4): Raw logits favor confident predictions. Can compare with
   log-softmax target in post-hoc analysis.

7. **Metric components** (Issue 9): Compare |grad|-only, |activation|-only, and product
   rankings post-hoc to verify the product adds information.

## 5. Implementation

- **New module**: `sae_experiments/feature_analysis/causal_feature_identifier.py`
- **New script**: `sae_experiments/scripts/02b_identify_features_causal.py`
- **Config**: Uses same YAML configs as v1 (`02_identify_features.py`)
- **Output**: `{experiment_dir}_causal/causal_feature_catalog.json`

Compatible with existing ablation pipeline (`03_ablate_features.py`).
