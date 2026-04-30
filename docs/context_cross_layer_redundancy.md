# Cross-Layer Redundancy: Problem Context

## Research Setup

The project studies cross-modal information flow in **LLaVA-v1.5-7b**, a 32-layer multimodal LLM (d_model=4096). Images are encoded by a vision encoder and inserted as 576 patch tokens into the token sequence. The task is **ChooseAttr** forced-choice VQA: given an image and a question like "Is the car red or blue?", the model picks the true attribute.

Two main tools:

**Attention knockout**: At inference time, zero out attention weights from image token positions → question token positions at a specific layer. This blocks the model from reading visual information at that layer. The causal effect is measured as `margin_drop = baseline log P(true/false) - knockout log P(true/false)`.

**Sparse Autoencoders (SAE)**: A 1-hidden-layer autoencoder (`encode: d_model → n_features`, `decode: n_features → d_model`) trained on activations at a specific layer. Used to decompose the layer's output into interpretable sparse features. After training, individual feature subsets are ablated and the behavioral effect is measured.

---

## Knockout Results (Ground Truth)

Layer-0 and layer-11 attn_out show the two dominant causal Image→Question peaks:

| Layer | margin_drop | Cohen's d | Interpretation |
|-------|-------------|-----------|----------------|
| **0** | **0.54** | **0.83** | Dominant — early cross-modal registration |
| **11** | **0.17** | **0.60** | Second peak — semantic binding |

The knockout mechanism: `methods.set_block_attn_hooks_llava(model, block_config)` replaces each targeted layer's `self_attn.forward` with a wrapper that injects a zeroed attention mask for the specified (source, target) token pairs. `remove_wrapper_llava` restores the originals. This is **method replacement** on the `forward` function, not a PyTorch hook — it persists for the entire duration it is registered.

---

## SAE Experiments: 18 Consecutive Null Results

SAEs were trained at layer-0 and layer-11 attn_out, then features were identified by correct/incorrect activation ratio and ablated. All 18 experiments showed binding features ≈ random features in their effect on model behavior.

**Full-latent ceiling experiment** (`10_full_latent_ablation.py`): zeroed ALL 4096 SAE features in `replace` mode to find the theoretical maximum effect. Results (n=128, color category):

| Layer | Position | Accuracy drop | Margin drop | Relative perturbation | Knockout margin_drop |
|-------|----------|--------------|-------------|----------------------|----------------------|
| 11 | attribute | +3.1% | 0.046 | 0.003 (0.3%) | 0.17 |
| 11 | all | +3.1% | 0.169 | 0.999 (99.9%) | 0.17 |
| 0 | attribute | −0.8% | 0.022 | 0.003 (0.3%) | 0.54 |
| 0 | all | −0.8% | 0.314 | 1.016 (101.6%) | 0.54 |

**Key finding from the ceiling**: At `all` positions, zeroing every SAE feature gives margin_drops of 0.314 (L0) and 0.169 (L11) — closely matching the knockout values (0.54 and 0.17). The SAE captures ~100% of activation norm. So the SAE is not the problem. The problem is something else.

---

## Diagnosis: Cross-Layer Redundancy

### What SAE ablation does

`replace` mode: `out = sae.decode(feats_without_selected)` — replaces the full activation at that layer with the SAE reconstruction minus the ablated features. This modifies the **residual stream output of one layer**.

### Why it doesn't replicate the knockout

The attention knockout places a **persistent information barrier**: at every forward pass through the targeted layer, the image tokens cannot attend to question tokens and vice versa. The barrier is maintained for the entire computation.

The SAE ablation is a **one-layer output modification**. After layer L is modified, the model continues computing normally at layers L+1 through 31. At each of those layers, the `self_attn` mechanism can directly attend to the **unchanged image token residual streams** — the image patch token positions still contain their original representations from layers 0 through L-1. The model reads those unchanged image tokens at layer L+1, L+2, ..., 31 and recovers the ablated visual signal.

### Concrete example

Suppose we ablate at layer 11. After ablation:
- Layer 11's output for question tokens is modified (missing some visual features)
- But image token residual streams at positions 0–10 are **unchanged** (these are the accumulated residual stream outputs up to layer 10, unaffected by our intervention at layer 11's attn_out)
- Layer 12's attention mechanism can directly read those image token residual streams
- The visual information "flows back in" at layer 12

This is why the full-latent ceiling at `all` positions gives margin_drop=0.169 but not 0.17 exactly, and why sparse feature ablation gives ~0: with all features ablated the perturbation is large enough to matter; with a few features the model trivially compensates.

### Why the knockout doesn't have this problem

The knockout blocks `self_attn.forward` itself — not just its output. Every layer where the knockout is applied cannot let image tokens influence question tokens, regardless of what the residual stream contains. The block is architectural and persistent, not a post-hoc output modification.

---

## What Has Been Tried

### Position fix (implemented, partially helped)

Prior experiments ablated at `attribute` text token positions (e.g., "red", "blue" in the question). The ceiling confirmed these positions carry only 0.3% relative perturbation — wrong positions entirely. Switching to `image` token positions (the 576 visual patch tokens) increased relative perturbation to 3.85%, but margin_drop remained ~0 due to cross-layer redundancy.

### Knockout-guided feature identification (script 07, implemented and run — null)

Standard feature selection used correct/incorrect ratio — selecting features correlated with task success, not with the causal pathway. Script `07_knockout_guided_features.py` runs paired forward passes (normal + layer-0 knockout) at layer 11, and ranks features by `mean(abs(feats_normal - feats_ko))` at **image token positions**.

**Result (601 samples, layer-11 SAE, layer-0 knockout):**
```
Knockout effect scores: max=0.00000, mean=0.00000, p95=0.00000
```
Every feature scored exactly zero. The layer-0 knockout produces **no measurable change** in layer-11 activations at image token positions.

**Why**: The `Image→Question` knockout blocks question tokens from *reading* image tokens at layer 0. It does not modify the image tokens' own residual streams. Image token activations propagate through all 32 layers completely unchanged by this knockout — the knockout only affects what question tokens receive, not what image tokens contain. Capturing attn_out at image positions therefore shows zero knockout effect by construction.

**Implication**: The knockout effect is only visible at **question token positions** — where cross-modal information arrives. All prior work capturing at image positions was measuring the wrong side of the information transfer.

### Activation patching (script 08, implemented and run — null by construction)

`08_activation_patching.py`: per sample, computes `acts_normal` and `acts_ko` at layer-11 image positions, encodes both through the SAE, patches binding features' values with their knockout values, and injects the reconstruction via a replacement hook.

**Result (601 samples):**
```
Knockout effect (mean|acts_normal - acts_ko| at image pos): 0.00000
Binding margin_drop = -0.0001  (Cohen's d = -0.026)
```
The patching experiment is null because `acts_normal ≡ acts_ko` at image positions (from the finding above). Patching zeros into zeros changes nothing.

**Root cause of both failures**: All experiments in scripts 07 and 08 captured/patched at image token positions. The `Image→Question` knockout exclusively modifies question token representations, not image token representations. The entire position selection was wrong.

---

## Current Codebase State

**Key files:**

| File | Purpose |
|------|---------|
| `sae_experiments/scripts/01_train_sae.py` | Train SAE at a layer |
| `sae_experiments/scripts/02_identify_features.py` | Ratio-based feature selection |
| `sae_experiments/scripts/03_run_ablation.py` | Standard SAE feature zeroing ablation |
| `sae_experiments/scripts/07_knockout_guided_features.py` | Knockout-supervised feature selection |
| `sae_experiments/scripts/08_activation_patching.py` | Per-feature activation patching |
| `sae_experiments/ablation/feature_ablator.py` | Core ablation hook; has `attn_block_config` + `attn_block_resolver` params |
| `sae_experiments/ablation/ablation_experiments.py` | 3-condition test orchestration; `n_random_sets=0` supported |
| `sae_experiments/data/activation_collector.py` | Collects layer activations; supports `image` position type |
| `sae_experiments/utils/knockout_utils.py` | `sequence_logprob`, `estimate_inputs_embeds_shape`, `get_image_token_range`, `resolve_flow_ranges`, `build_block_config` |
| `sae_experiments/utils/hook_utils.py` | `HookManager`, `create_activation_capture_hook`, `get_target_module` |
| `methods.py` | `set_block_attn_hooks_llava`, `remove_wrapper_llava` |

**Configs ready to run:**
- `configs/sae_categories/sae_layer11_attn_out_v2/color.yaml` — layer 11, color only, V2 SAE (4096 features, l1_coeff=5e-4, training on all positions, feature ID + ablation on image positions)
- `configs/sae_categories/sae_layer0_attn_out_v2/color.yaml` — same for layer 0

**V2 SAE architecture** (applied): `b_pre` encoder bias, `normalize_decoder()` after every optimizer step, cosine LR scheduler, dead feature fraction tracked per epoch.

**Completed runs:**
- Layer-0 SAE (V2): trained, 37 discriminative features found at image positions, ablation margin_drop ≈ 0
- Layer-11 SAE (V2): trained on 383,910 vectors (20 epochs, 2:41), checkpoint at `output/sae_experiments/sae_v2_layer11_attn_out_color/sae_checkpoint.pt`
- Script 07 (knockout-guided feature ID at image positions): run, all knockout effect scores = 0 — wrong positions
- Script 08 (activation patching at image positions): run, margin_drop = -0.0001 — null by construction

---

## Updated Understanding of the Problem

The original diagnosis of "cross-layer redundancy" was partially correct but missed a more fundamental issue: **position selection**. All SAE work to date has targeted image token positions, but the `Image→Question` knockout exclusively affects question token representations. The information transfer goes:

```
image tokens → (attention) → question tokens
```

The *source* is the image tokens; the *destination* is the question tokens. When the knockout blocks this flow, question token activations change. Image token activations do not.

This means:
- Ablating SAE features at **image positions** cannot replicate the knockout — not because of cross-layer redundancy, but because nothing is happening at image positions when the knockout fires
- The causal pathway through the SAE should be measured at **question token positions** — this is where the cross-modal information arrives and where the knockout's effect is visible
- The "cross-layer redundancy" framing was correct at the level of subsequent layers re-reading image tokens, but this is secondary to the primary position error

**What is known to work as a ceiling**: Zeroing all SAE features at `all` positions (including question positions) gives margin_drop = 0.169 at layer 11, matching the knockout value. This confirms the SAE does capture the relevant signal when question positions are included.

## The Core Question

Given that:
1. The SAE captures ~100% of activation norm when applied at all positions (including question tokens)
2. The `Image→Question` knockout changes question token activations, not image token activations
3. All prior SAE experiments targeted image token positions, which are unaffected by the knockout

**What experimental design would produce a valid test of whether specific SAE features at question token positions causally mediate the Image→Question cross-modal pathway?**

Key constraints to respect:
- The intervention must target question token positions (where the knockout effect lands)
- Cross-layer redundancy still applies: modifying question token activations at layer L allows layers L+1 to L+31 to re-read image tokens and partially restore the signal
- The SAE was trained on all-position activations; its features span both image and question token representations
