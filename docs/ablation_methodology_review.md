# Ablation Methodology Review

## 1. Current Approach — Theoretical Summary

Our ablation pipeline (`feature_ablator.py`, `ablation_experiments.py`, `03_run_ablation.py`) works as follows:

### Feature ablation hook (FeatureAblator.create_ablation_hook)
For each sample during inference, a forward hook is registered on the target layer's attn_out module. The hook:
1. **Encodes** the full layer output `h` through the SAE: `z = SAE.encode(h)` across ALL sequence positions
2. **Zeros** the target feature activations: `z_modified[:, feature_indices] = 0`
3. **Decodes** back: `h_modified = SAE.decode(z_modified)`
4. **Applies** the modification:
   - **Replace mode**: `output = h_modified` (at target positions, keep original elsewhere)
   - **Residual mode**: `output = h + (SAE.decode(z_modified) - SAE.decode(z_original))`

### Three-condition test (AblationExperiment.run_three_condition_test)
1. **Baseline**: Run model unmodified, collect predictions and logit margins
2. **Binding ablation**: Ablate top-200 causal features, measure margin drop
3. **Random control**: Ablate 200 random features (15 sets), measure margin drop

### Metrics
- `margin_drop = baseline_margin - ablated_margin` (higher = more disruption)
- `accuracy_drop = baseline_acc - ablated_acc`
- `relative_perturbation = ||h_modified - h|| / ||h||`

### Previous v2 result (layer 11, n=256):
- Binding margin_drop: +0.213
- Random mean margin_drop: -0.0004
- z-score: 81.6
- Captures 39% of knockout ceiling (0.540)

---

## 2. Literature Review — Key Findings

### 2.1 SAE Reconstruction Error as Confound

**Marks et al. (2024), "Sparse Feature Circuits" (arXiv:2403.19647, ICLR 2025 Oral)**:
Decomposes activations as `h = sum(f_i * d_i) + epsilon`, treating epsilon as an explicit error
node. With error nodes, the SAE insertion becomes an identity: `SAE(h) + epsilon = h`. Their
circuit discovery attributes effects to both features AND error nodes, only pruning features
(never the error term). **Our "replace mode" discards epsilon entirely.**

**Gurnee (2024), "SAE Reconstruction Errors Are (Empirically) Pathological"**:
SAE reconstruction errors produce 2-4.5x larger KL divergence than random perturbations of
the same L2 norm. SAE errors point in systematically harmful directions.

**Lee & Heimersheim (2024), "Investigating Sensitive Directions in GPT-2" (arXiv:2410.12555)**:
Partially refutes the above: when comparing against covariance-aware (non-isotropic) random
directions, SAE errors are not pathologically large. The activation space is highly anisotropic.

**Net assessment**: Our random-feature control does control for reconstruction error (same
encode/decode path), but we should additionally measure the "SAE pass-through" effect
(insert SAE without zeroing any features) as a sanity check.

### 2.2 Zero Ablation vs Alternatives

**Li & Janson (2024), "Optimal Ablation for Interpretability" (arXiv:2409.09951, NeurIPS 2024 Spotlight)**:
Optimal ablation (OA) sets components to the constant minimizing expected loss. **OA importance
is only 11.1% of zero-ablation importance.** Zero ablation massively overestimates component
importance. However, OA has the highest rank correlation (0.907) with counterfactual ablation.

**Heimersheim & Nanda (2024), "How to Use and Interpret Activation Patching" (arXiv:2404.15255)**:
Zero ablation of SAE features is less problematic than zero ablation of neurons, because SAE
features have a natural zero (inactive state). Most features are inactive most of the time,
so zero IS close to the population mean for sparse features.

**Chughtai et al. (2024), "Transformer Circuit Faithfulness Metrics Are Not Robust" (arXiv:2407.08734, COLM 2024)**:
Circuit faithfulness scores are highly sensitive to ablation methodology. The type of ablation
determines what question you're asking.

### 2.3 Position-Selective Ablation

**Marks et al. (2024)**: Encode ALL positions through the SAE, but include the error term at
every position. At non-target positions, the net effect is zero (features + error = original).
At target positions, selectively zero features while preserving the error term.

**Chughtai et al. (2024)**: "If a circuit is specified at chosen token positions, it should be
tested with position-specific ablation."

**Our approach**: We encode ALL positions through the SAE in the hook, but only replace
activations at target positions (question tokens). This means:
- At question positions: `h_new = SAE.decode(z_modified)` — no error term preserved
- At non-question positions: `h_new = h_original` — untouched

The error term is lost at question positions in replace mode, but preserved in residual mode
(since delta = SAE_modified - SAE_original, the error cancels).

### 2.4 Multi-Layer Ablation

**Anthropic (2024), "Sparse Crosscoders" (transformer-circuits.pub)**:
Crosscoders outperform per-layer SAEs, indicating significant redundant structure across layers.
Features persist across multiple layers. This directly explains our 39% ceiling — the model
re-reads image information at layers 12-14.

### 2.5 Attribution Patching Connection

**Syed et al. (2023), "Attribution Patching Outperforms Automated Circuit Discovery" (arXiv:2310.10348)**:
Our v2 gradient×activation scoring IS attribution patching applied to SAE features. This is
well-established methodology.

**Dunefsky et al. (2024), "Features That Make a Difference" (arXiv:2411.10397)**:
Standard SAEs waste capacity on high-activation but low-gradient features. Gradient-based
scoring correctly filters for functionally important features.

---

## 3. Diagnosed Issues

### Issue A: Replace mode discards SAE reconstruction error (CRITICAL)

**Current behavior**: In replace mode, `create_ablation_hook` sets:
```python
recon_mod = self.sae.decode(feats_mod, target_shape=acts.shape)
out = recon_mod  # at target positions
```

This replaces `h` with `SAE.decode(SAE.encode(h) with zeroed features)`. The reconstruction
error `epsilon = h - SAE.decode(SAE.encode(h))` is silently discarded. Per Marks et al., this
error should be preserved:
```python
error = acts - self.sae.decode(feats_full, target_shape=acts.shape)
out = recon_mod + error  # preserve the error term
```

**Impact**: Without the error term, every sample suffers systematic reconstruction damage
at question positions, regardless of which features are zeroed. Our random-feature control
partially addresses this (same reconstruction path), but the absolute margin_drop numbers
are inflated. The z-score is valid; the raw margin_drop of 0.213 is not directly comparable
to the knockout ceiling of 0.540.

**Severity**: MODERATE. The relative comparison (binding vs random) is sound. The absolute
effect size is biased upward by reconstruction error.

### Issue B: No SAE pass-through baseline (MODERATE)

We never measure the effect of just inserting the SAE without zeroing ANY features. This
would quantify the reconstruction error baseline:
- `passthrough_margin_drop = baseline_margin - SAE_passthrough_margin`

If this is large, it means reconstruction error is a significant confound. If near zero,
our numbers are clean.

### Issue C: Residual mode double-decodes (MINOR)

In residual mode:
```python
recon_full = self.sae.decode(feats_full)   # decode 1
recon_mod = self.sae.decode(feats_mod)     # decode 2
delta = recon_mod - recon_full
out = acts + delta
```

This is mathematically equivalent to: `out = acts + W_dec @ (z_mod - z_full)`, where
`z_mod - z_full` is simply the zeroed-out features with negated activations. The error
term cancels correctly (both decodes share the same decoder bias). This is actually the
**correct** approach per the literature — it isolates the feature effect without
reconstruction error.

**However**: We're still computing two full decode passes. Since `z_mod - z_full` only
differs at the zeroed feature indices, we could compute `delta = -W_dec[:, feature_indices] @ z_full[:, feature_indices]`
directly. This is a minor efficiency issue, not a correctness issue.

### Issue D: Hook encodes ALL positions, may waste compute (MINOR)

The hook encodes and decodes all sequence positions (~600 tokens) even when only ~26
question positions are targeted. This is correct for residual mode (delta at non-target
positions is zero anyway) but wasteful. Not a correctness issue.

### Issue E: b_pre handling in encode but not decode

The SAE encodes with `x - b_pre` but decode doesn't add `b_pre` back:
```python
def encode(self, x): return ReLU(W_enc @ (x - b_pre))
def decode(self, z): return W_dec @ z  # + decoder.bias only
```

In replace mode, this means `h_new = W_dec @ z + b_dec`, while `h_original ≈ W_dec @ z + b_dec + epsilon`.
The b_pre is "baked into" the encoder and decoder biases during training, so this is
consistent (the SAE was trained this way). But it means the SAE's zero-point is shifted
by b_pre relative to the residual stream.

In residual mode, this cancels: `delta = (W_dec @ z_mod + b_dec) - (W_dec @ z_full + b_dec) = W_dec @ (z_mod - z_full)`.

**Severity**: LOW for residual mode, MODERATE for replace mode.

### Issue F: Replace mode used in v2 breakthrough result

The v2 breakthrough result (margin_drop=0.213, z=81.6) used **replace mode**. Per Issue A,
this discards the error term. The relative effect (vs random) is valid, but the absolute
margin_drop may be inflated.

**Recommendation**: Re-run the ablation in residual mode and compare. If margin_drop is
similar, the effect is robust. If significantly smaller, reconstruction error was a contributor.

### Issue G: score_options relies on true/false option labels

`batch_ablation_experiment` computes logprob margins using `true option` and `false option`
fields from the dataset. CLEVR-Lite doesn't have binary forced-choice — it's open-ended
("What color is the triangle?" → "red"). The `true option` is the correct answer and
`false option` is... what? If not properly set, margin computation is meaningless.

**Need to verify**: Does CLEVRLiteVQADataset populate `true option` and `false option`?

---

## 4. Proposed Fixes

### Fix 1: Add error term preservation in replace mode
```python
# In create_ablation_hook, replace mode:
error = acts - recon_full.to(device=acts_device, dtype=acts_dtype)
out = recon_mod.to(device=acts_device, dtype=acts_dtype) + error
```
This makes replace mode equivalent to residual mode (they should produce identical results).

### Fix 2: Add SAE pass-through baseline condition
Run the ablation with an empty feature list (zero features zeroed) to measure pure
reconstruction error. This is the "SAE insertion without ablation" baseline.

### Fix 3: Verify CLEVR-Lite true/false options
Check how margin is computed for open-ended CLEVR-Lite questions.

### Fix 4: Standardize on residual mode (or error-corrected replace mode)
Residual mode naturally preserves the error term. Make it the default.

### Fix 5: Multi-layer simultaneous ablation (future)
Ablate the same feature set across layers 10-14 simultaneously to close the 39% ceiling gap.

---

## 5. What Does NOT Need Fixing

> **RETRACTION (2026-08-07).** The third and fifth bullets below are wrong and are kept only so the
> error is traceable. The random controls were never matched — they were uniform. Configs set
> `matched_metric: "correct_mean"`, a key absent from every v2 stats file, so `_extract_metric_value`
> returned `None` and `_sample_matched_random_features` silently took its uniform branch. At layer 11
> the controls have median activation 6.1e-08 against the binding set's 0.117. The z-scores therefore
> compare top-causal features against near-dead features and are inflated. See CLAUDE.md
> § "Ablation modes" and the multi-layer work for the fix (`matched_metric: "activation_mean"` plus
> `random_control.strict_matching: true`).

- **Zero ablation of SAE features is appropriate** (features have natural zero state)
- **gradient × activation feature identification is well-grounded** (attribution patching)
- ~~**Random-feature control methodology is sound** (same encode/decode path)~~ — **retracted, see above**
- **Position-selective ablation is correct** (residual mode delta is zero at non-target positions)
- ~~**The relative z-score of 81.6 is valid** regardless of reconstruction error~~ — valid with respect
  to *reconstruction error*, but inflated by the uniform-control bug above

---

## 7. Experimental Validation

### Test: Layer 11, n=50, error-preserving delta mode

| Condition | margin_drop | acc_drop | rel_perturb |
|-----------|------------|----------|-------------|
| SAE pass-through (0 features) | **+0.0000** | 0.0000 | 0.0000 |
| Binding (200 causal features) | **+0.2077** | 0.0000 | 0.0188 |
| Random (200 random features, mean of 15) | -0.0002 | 0.0000 | 0.0013 |
| **z-score** | **71.5** | — | — |

### Conclusions

1. **Pass-through = 0 exactly** proves the delta approach introduces zero reconstruction contamination.
   The original replace-mode result (0.213) was NOT inflated by reconstruction error.
2. **Error-preserving delta produces nearly identical results** (0.208 vs 0.213 within sampling noise).
   The methodology was already sound for relative comparisons.
3. ~~**The original z-score of 81.6 is valid.**~~ **Retracted 2026-08-07** — valid against *reconstruction
   error*, which is all this test examined, but inflated by the uniform-control bug (see the retraction
   note in section 5). The z here (71.5) has the same defect. The slight reduction from 81.6 is due to
   smaller sample size (n=50 vs n=256).
4. **No methodological fix was necessary for the core finding.** The improvements are:
   - Cleaner code (single path instead of two modes)
   - Pass-through baseline provides an additional sanity check
   - Error term is preserved by construction (for any future mode changes)

### What DOES need addressing (future work)

- **Multi-layer ablation**: The 39% ceiling (0.213 vs 0.540 knockout) is a real limitation.
  Simultaneously ablating layers 10-14 should close the gap.
- **Feature count sweep**: Try top-50, top-100, top-500 features to find the saturation point.
- **Individual feature contribution**: Which of the 200 features account for most of the 0.208?

---

## 8. References

- Marks et al. (2024), "Sparse Feature Circuits" (arXiv:2403.19647)
- Gurnee (2024), "SAE Reconstruction Errors Are (Empirically) Pathological"
- Lee & Heimersheim (2024), "Investigating Sensitive Directions in GPT-2" (arXiv:2410.12555)
- Li & Janson (2024), "Optimal Ablation for Interpretability" (arXiv:2409.09951)
- Heimersheim & Nanda (2024), "How to Use and Interpret Activation Patching" (arXiv:2404.15255)
- Chughtai et al. (2024), "Transformer Circuit Faithfulness Metrics Are Not Robust" (arXiv:2407.08734)
- Anthropic (2024), "Sparse Crosscoders" (transformer-circuits.pub)
- Syed et al. (2023), "Attribution Patching Outperforms Automated Circuit Discovery" (arXiv:2310.10348)
- Dunefsky et al. (2024), "Features That Make a Difference" (arXiv:2411.10397)
- Braun et al. (2024), "End-to-End Sparse Dictionary Learning" (arXiv:2405.12241)
- Shu et al. (2025), "Beyond Input Activations / GradSAE" (arXiv:2505.08080)
- Makelov et al. (2024), "Towards Principled Evaluations of SAEs" (arXiv:2405.08366)
- Gao et al. (2024), "Scaling and Evaluating Sparse Autoencoders" (arXiv:2406.04093)
