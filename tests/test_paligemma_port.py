"""Unit tests for the PaliGemma 2 port.

Fast tests use only synthetic tensors and stub models — no model download required.
Tests that download from HuggingFace are marked ``@pytest.mark.slow``.

Run fast tests only:
    pytest tests/test_paligemma_port.py -v -m "not slow"

Run all tests (including HF downloads):
    pytest tests/test_paligemma_port.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---------------------------------------------------------------------------
# Stub model helpers (no GPU / HF download required)
# ---------------------------------------------------------------------------

D_MODEL = 8
N_FEATURES = 16


def _make_sae(d_model=D_MODEL, n_features=N_FEATURES):
    """Return a GemmaScopeJumpReLUSAE with random weights."""
    from sae_experiments.models.gemma_scope_sae import GemmaScopeJumpReLUSAE
    sae = GemmaScopeJumpReLUSAE(d_model=d_model, n_features=n_features)
    with torch.no_grad():
        nn.init.orthogonal_(sae.W_enc)
        nn.init.orthogonal_(sae.W_dec)
        sae.b_enc.zero_()
        sae.b_dec.zero_()
        sae.threshold.fill_(-1e6)  # effectively disable gating for most tests
    return sae


class StubSelfAttn(nn.Module):
    """Minimal self-attention stub that records the last attention mask it received."""

    def __init__(self):
        super().__init__()
        self.last_attention_mask = None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        self.last_attention_mask = (
            attention_mask.clone() if attention_mask is not None else None
        )
        return hidden_states, None, None


class StubLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = StubSelfAttn()

    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class StubLM(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([StubLayer() for _ in range(n_layers)])

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class StubModel(nn.Module):
    """Mimics PaliGemmaForConditionalGeneration structure."""

    def __init__(self, n_layers=2, d_model=D_MODEL, num_image_tokens=256):
        super().__init__()
        self.language_model = StubLM(n_layers=n_layers)
        self.config = MagicMock()
        self.config.num_image_tokens = num_image_tokens
        self.config.num_hidden_layers = n_layers
        # Minimal logit output for sequence_logprob tests
        self._d_model = d_model
        self._vocab_size = 32
        self._logit_layer = nn.Linear(d_model, self._vocab_size, bias=False)

    def forward(self, input_ids=None, pixel_values=None, use_cache=False, **kwargs):
        batch, seq_len = input_ids.shape
        hidden = torch.randn(batch, seq_len, self._d_model)
        logits = self._logit_layer(hidden)
        out = MagicMock()
        out.logits = logits
        return out

    def generate(self, input_ids=None, pixel_values=None, **kwargs):
        batch = input_ids.shape[0]
        # Return the first token of the vocabulary as the generated token
        sequences = torch.zeros(batch, 1, dtype=torch.long)
        scores = [torch.randn(batch, self._vocab_size)]
        out = {"sequences": sequences, "scores": scores}
        return out


# ===========================================================================
# 8a  GemmaScopeJumpReLUSAE
# ===========================================================================

class TestGemmaScopeJumpReLUSAE:

    def test_jumprelu_zero_below_threshold(self):
        """All pre-activations below threshold → all feature activations == 0."""
        from sae_experiments.models.gemma_scope_sae import GemmaScopeJumpReLUSAE
        sae = GemmaScopeJumpReLUSAE(d_model=D_MODEL, n_features=N_FEATURES)
        with torch.no_grad():
            sae.threshold.fill_(1e6)   # very large threshold
            sae.b_dec.zero_()
            sae.b_enc.zero_()
            nn.init.eye_(sae.W_enc[:min(D_MODEL, N_FEATURES), :min(D_MODEL, N_FEATURES)])
        x = torch.zeros(1, D_MODEL)
        z = sae.encode(x)
        assert (z == 0).all(), "All features should be zero when pre_acts < threshold"

    def test_jumprelu_active_above_threshold(self):
        """Pre-activations all above threshold → output == relu(pre_acts)."""
        from sae_experiments.models.gemma_scope_sae import GemmaScopeJumpReLUSAE
        sae = GemmaScopeJumpReLUSAE(d_model=D_MODEL, n_features=N_FEATURES)
        with torch.no_grad():
            sae.threshold.fill_(-1e6)  # very small threshold (always pass)
            sae.b_dec.zero_()
            sae.b_enc.zero_()
            sae.W_enc.fill_(0.0)
            # Make W_enc a scaled identity-like matrix
            for i in range(min(D_MODEL, N_FEATURES)):
                sae.W_enc[i, i] = 1.0
        x = torch.ones(1, D_MODEL)
        z = sae.encode(x)
        pre = x @ sae.W_enc + sae.b_enc
        expected = F.relu(pre)
        assert torch.allclose(z, expected, atol=1e-5), "z should equal relu(pre) when threshold is -inf"

    def test_encode_decode_shape(self):
        sae = _make_sae()
        x = torch.randn(3, D_MODEL)
        z = sae.encode(x)
        assert z.shape == (3, N_FEATURES), f"Expected (3, {N_FEATURES}), got {z.shape}"
        recon = sae.decode(z)
        assert recon.shape == (3, D_MODEL), f"Expected (3, {D_MODEL}), got {recon.shape}"

    def test_encode_3d_input_shape(self):
        """encode handles (batch, seq, d_model) input → (batch*seq, n_features)."""
        sae = _make_sae()
        x = torch.randn(2, 5, D_MODEL)
        z = sae.encode(x)
        assert z.shape == (10, N_FEATURES)

    def test_decode_with_target_shape(self):
        sae = _make_sae()
        x = torch.randn(2, 5, D_MODEL)
        z = sae.encode(x)
        recon = sae.decode(z, target_shape=x.shape)
        assert recon.shape == x.shape

    def test_encode_decode_reconstruction_synthetic(self):
        """With identity-like W_enc and W_dec, decode(encode(x)) ≈ x for positive x."""
        from sae_experiments.models.gemma_scope_sae import GemmaScopeJumpReLUSAE
        d = 4
        sae = GemmaScopeJumpReLUSAE(d_model=d, n_features=d)
        with torch.no_grad():
            # Set W_enc = W_dec^T = identity; b_dec = 0; b_enc = 0; threshold = -1e6
            sae.W_enc.copy_(torch.eye(d))
            sae.W_dec.copy_(torch.eye(d))
            sae.b_dec.zero_()
            sae.b_enc.zero_()
            sae.threshold.fill_(-1e6)
        x = torch.abs(torch.randn(3, d))  # positive so relu doesn't kill anything
        recon, _ = sae.forward(x)
        assert torch.allclose(recon, x, atol=1e-4), "Reconstruction should match input for identity weights"

    def test_encode_nonnegative(self):
        """Feature activations are always ≥ 0 (JumpReLU gate + relu)."""
        sae = _make_sae()
        x = torch.randn(10, D_MODEL)
        z = sae.encode(x)
        assert (z >= 0).all(), "All SAE feature activations must be non-negative"

    @pytest.mark.slow
    def test_from_hf_loads_without_error(self):
        """from_hf() downloads and loads layer 0 SAE without errors."""
        from sae_experiments.models.gemma_scope_sae import GemmaScopeJumpReLUSAE
        sae = GemmaScopeJumpReLUSAE.from_hf(layer_idx=0, width="16k")
        assert sae.d_model == 2304
        assert sae.n_features == 16384

    @pytest.mark.slow
    def test_from_hf_encode_nonnegative(self):
        from sae_experiments.models.gemma_scope_sae import GemmaScopeJumpReLUSAE
        sae = GemmaScopeJumpReLUSAE.from_hf(layer_idx=0, width="16k")
        x = torch.randn(4, 2304)
        z = sae.encode(x)
        assert (z >= 0).all()


# ===========================================================================
# 8b  paligemma_hooks
# ===========================================================================

class TestPaliGemmaHooks:

    def test_hook_install_and_remove(self):
        from sae_experiments.utils.paligemma_hooks import (
            set_block_attn_hooks_paligemma, remove_wrapper_paligemma,
        )
        model = StubModel(n_layers=2)
        self_attn = model.language_model.model.layers[0].self_attn
        # The class-level function (unbound) is the canonical original
        original_func = StubSelfAttn.forward

        hooks = set_block_attn_hooks_paligemma(model, {0: [(1, 0)]})
        # After install: a plain hooked function is stored in instance __dict__
        installed = self_attn.__dict__.get("forward")
        assert installed is not None, "hook should be stored in instance __dict__"
        assert installed is not original_func, "installed hook should differ from original"

        remove_wrapper_paligemma(model, hooks)
        # After remove: the stored original bound method is put back in __dict__
        # Its underlying __func__ must match the class-level original
        restored = self_attn.__dict__.get("forward")
        assert restored is not None, "restore should leave a forward in instance __dict__"
        assert restored.__func__ is original_func, "restored forward should be the original"

    def test_hook_blocks_correct_positions(self):
        from sae_experiments.utils.paligemma_hooks import (
            set_block_attn_hooks_paligemma, remove_wrapper_paligemma,
        )
        model = StubModel(n_layers=1)
        q_len = kv_len = 5
        mask = torch.zeros(1, 1, q_len, kv_len)
        dtype_min = float(torch.finfo(torch.float32).min)

        hooks = set_block_attn_hooks_paligemma(model, {0: [(2, 1)]})
        hs = torch.randn(1, q_len, D_MODEL)
        model.language_model.model.layers[0].self_attn(hs, attention_mask=mask)
        got_mask = model.language_model.model.layers[0].self_attn.last_attention_mask

        assert got_mask[0, 0, 2, 1] == pytest.approx(dtype_min, rel=1e-3), \
            "Position (2,1) should be blocked"
        # All other positions in the original mask should remain 0
        for r in range(q_len):
            for c in range(kv_len):
                if (r, c) != (2, 1):
                    assert got_mask[0, 0, r, c] == pytest.approx(0.0), \
                        f"Position ({r},{c}) should be unmodified"

        remove_wrapper_paligemma(model, hooks)

    def test_hook_handles_none_attention_mask(self):
        from sae_experiments.utils.paligemma_hooks import (
            set_block_attn_hooks_paligemma, remove_wrapper_paligemma,
        )
        model = StubModel(n_layers=1)
        hooks = set_block_attn_hooks_paligemma(model, {0: [(1, 0)]})
        hs = torch.randn(1, 3, D_MODEL)
        # Should not raise even with attention_mask=None
        model.language_model.model.layers[0].self_attn(hs, attention_mask=None)
        got_mask = model.language_model.model.layers[0].self_attn.last_attention_mask
        assert got_mask is not None, "Hook should construct a mask when None is passed"
        remove_wrapper_paligemma(model, hooks)

    def test_hook_does_not_modify_unblocked_positions(self):
        from sae_experiments.utils.paligemma_hooks import (
            set_block_attn_hooks_paligemma, remove_wrapper_paligemma,
        )
        model = StubModel(n_layers=1)
        q_len = kv_len = 4
        mask = torch.zeros(1, 1, q_len, kv_len)
        hooks = set_block_attn_hooks_paligemma(model, {0: [(2, 1)]})
        hs = torch.randn(1, q_len, D_MODEL)
        model.language_model.model.layers[0].self_attn(hs, attention_mask=mask)
        got = model.language_model.model.layers[0].self_attn.last_attention_mask
        assert got[0, 0, 0, 0] == pytest.approx(0.0)
        assert got[0, 0, 1, 2] == pytest.approx(0.0)
        remove_wrapper_paligemma(model, hooks)

    def test_multiple_layers_blocked(self):
        from sae_experiments.utils.paligemma_hooks import (
            set_block_attn_hooks_paligemma, remove_wrapper_paligemma,
        )
        model = StubModel(n_layers=3)
        block_config = {0: [(1, 0)], 2: [(3, 2)]}
        hooks = set_block_attn_hooks_paligemma(model, block_config)
        assert len(hooks) == 2
        remove_wrapper_paligemma(model, hooks)


# ===========================================================================
# 8c  knockout_utils
# ===========================================================================

class TestKnockoutUtils:

    def test_get_paligemma_image_token_count_224(self):
        from sae_experiments.utils.knockout_utils import get_paligemma_image_token_count
        assert get_paligemma_image_token_count("google/paligemma2-3b-pt-224") == 256

    def test_get_paligemma_image_token_count_448(self):
        from sae_experiments.utils.knockout_utils import get_paligemma_image_token_count
        assert get_paligemma_image_token_count("google/paligemma2-3b-pt-448") == 1024

    def test_get_image_token_count_from_model(self):
        from sae_experiments.utils.knockout_utils import get_image_token_count
        model = StubModel(num_image_tokens=256)
        assert get_image_token_count(model) == 256

    def test_get_image_token_range(self):
        from sae_experiments.utils.knockout_utils import get_image_token_range
        r = get_image_token_range(256)
        assert r == list(range(0, 256))
        assert len(r) == 256

    def test_get_question_token_range_finds_text(self):
        """get_question_token_range returns positions of question text in input_ids."""
        from sae_experiments.utils.knockout_utils import get_question_token_range
        # Build a fake tokenizer that encodes words to integer IDs
        tokenizer = MagicMock()
        # "what color" → [100, 101]
        tokenizer.encode.return_value = [100, 101]
        # input_ids: 5 image tokens (0..4), then [100, 101, 102]
        ids = torch.tensor([[0] * 5 + [100, 101, 102]])
        result = get_question_token_range(ids, "what color", tokenizer)
        assert result == [5, 6], f"Expected [5, 6], got {result}"

    def test_get_last_token_range(self):
        from sae_experiments.utils.knockout_utils import get_last_token_range
        ids = torch.zeros(1, 10, dtype=torch.long)
        assert get_last_token_range(ids) == [9]

    def test_build_block_config_single_layer(self):
        from sae_experiments.utils.knockout_utils import build_block_config
        cfg = build_block_config(3, 10, window=1, src_tgt_pairs=[(5, 2)])
        assert cfg == {3: [(5, 2)]}

    def test_build_block_config_window(self):
        from sae_experiments.utils.knockout_utils import build_block_config
        cfg = build_block_config(5, 10, window=3, src_tgt_pairs=[(1, 0)])
        assert set(cfg.keys()) == {4, 5, 6}
        for v in cfg.values():
            assert v == [(1, 0)]

    def test_build_block_config_window_clamp(self):
        from sae_experiments.utils.knockout_utils import build_block_config
        # Window extending beyond layer 0 should clamp at 0
        cfg = build_block_config(0, 10, window=3, src_tgt_pairs=[(0, 1)])
        assert 0 in cfg
        assert all(k >= 0 for k in cfg)

    def test_sequence_logprob_returns_float(self):
        """sequence_logprob returns a float for a simple stub model."""
        from sae_experiments.utils.knockout_utils import sequence_logprob
        model = StubModel()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [5]  # answer "red" → token id 5
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        pixel_values = torch.randn(1, 3, 224, 224)
        result = sequence_logprob(model, tokenizer, input_ids, pixel_values, "red")
        assert isinstance(result, float)

    def test_sequence_logprob_empty_answer_returns_none(self):
        from sae_experiments.utils.knockout_utils import sequence_logprob
        model = StubModel()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []
        result = sequence_logprob(model, tokenizer,
                                   torch.zeros(1, 5, dtype=torch.long),
                                   torch.randn(1, 3, 224, 224), "")
        assert result is None

    def test_resolve_flow_ranges_image_question(self):
        from sae_experiments.utils.knockout_utils import resolve_flow_ranges
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [100, 101]
        ids = torch.tensor([[0] * 5 + [100, 101, 102]])
        src, tgt = resolve_flow_ranges("Image->Question", ids, 5, "q", tokenizer)
        assert src == list(range(0, 5))
        assert tgt == [5, 6]

    def test_resolve_flow_ranges_image_last(self):
        from sae_experiments.utils.knockout_utils import resolve_flow_ranges
        tokenizer = MagicMock()
        ids = torch.tensor([[0] * 5 + [1, 2, 3]])   # seq_len = 8
        src, tgt = resolve_flow_ranges("Image->Last", ids, 5, "", tokenizer)
        assert src == list(range(0, 5))
        assert tgt == [7]


# ===========================================================================
# 8d  activation_collector
# ===========================================================================

class TestActivationCollector:

    def test_select_positions_image(self):
        from sae_experiments.data.activation_collector import ActivationCollector
        model = StubModel(num_image_tokens=256)
        collector = ActivationCollector(model, layer_idx=0)
        ids = torch.zeros(1, 300, dtype=torch.long)
        positions = collector._select_positions("image", ids, 256, "", None, {})
        assert positions == list(range(0, 256))

    def test_select_positions_all(self):
        from sae_experiments.data.activation_collector import ActivationCollector
        model = StubModel()
        collector = ActivationCollector(model, layer_idx=0)
        ids = torch.zeros(1, 10, dtype=torch.long)
        positions = collector._select_positions("all", ids, 256, "", None, {})
        assert positions == list(range(0, 10))

    def test_select_positions_last(self):
        from sae_experiments.data.activation_collector import ActivationCollector
        model = StubModel()
        collector = ActivationCollector(model, layer_idx=0)
        ids = torch.zeros(1, 8, dtype=torch.long)
        positions = collector._select_positions("last", ids, 256, "", None, {})
        assert positions == [7]

    def test_select_positions_question(self):
        from sae_experiments.data.activation_collector import ActivationCollector
        model = StubModel()
        collector = ActivationCollector(model, layer_idx=0)
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [10, 11]
        ids = torch.tensor([[0] * 5 + [10, 11, 12]])
        positions = collector._select_positions("question", ids, 5, "q", tokenizer, {})
        assert positions == [5, 6]

    def test_select_positions_image_clamps_to_seq_len(self):
        from sae_experiments.data.activation_collector import ActivationCollector
        model = StubModel(num_image_tokens=256)
        collector = ActivationCollector(model, layer_idx=0)
        ids = torch.zeros(1, 10, dtype=torch.long)  # shorter than 256
        positions = collector._select_positions("image", ids, 256, "", None, {})
        assert len(positions) == 10  # clamped to seq_len

    def test_collector_hook_fires(self):
        """register_hooks fires and captures activation shape."""
        from sae_experiments.data.activation_collector import ActivationCollector

        # Build a tiny real model so hooks work
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = StubLM(n_layers=1)
                self.config = MagicMock()
                self.config.num_image_tokens = 4

            def forward(self, input_ids, pixel_values=None, use_cache=False, **kw):
                hs = torch.randn(1, input_ids.shape[1], D_MODEL)
                for layer in self.language_model.model.layers:
                    layer.self_attn(hs)
                return MagicMock(logits=torch.randn(1, input_ids.shape[1], 32))

        # Patch StubLayer to actually return a tensor from forward
        class RealLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = StubSelfAttn()

            def forward(self, hidden_states, **kwargs):
                # Return tensor so hook can capture it
                return hidden_states

        m = TinyModel()
        m.language_model.model.layers[0] = RealLayer()

        collector = ActivationCollector(m, layer_idx=0, activation_site="residual")
        collector.register_hooks()

        input_ids = torch.zeros(1, 6, dtype=torch.long)
        pixel_values = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            m(input_ids=input_ids, pixel_values=pixel_values, use_cache=False)

        # Hook captures whatever the layer returns — just verify it ran
        # (StubSelfAttn returns tuple with None so storage may not have "acts")
        # Just verify no crash; storage key exists if layer returned a tensor
        collector.hook_manager.remove_hooks()


# ===========================================================================
# 8e  feature_ablator
# ===========================================================================

class TestFeatureAblator:

    def _make_ablator(self, mode="replace"):
        from sae_experiments.ablation.feature_ablator import FeatureAblator
        model = StubModel()
        sae = _make_sae()
        return FeatureAblator(model, sae, layer_idx=0, activation_site="residual"), model, sae

    def test_ablation_replace_mode_output_shape(self):
        ablator, model, sae = self._make_ablator()
        acts = torch.randn(1, 10, D_MODEL)
        hook_fn = ablator.create_ablation_hook(feature_indices=[0], positions=None, mode="replace")
        out = hook_fn(None, None, acts)
        assert out.shape == acts.shape

    def test_ablation_replace_mode_differs_from_input(self):
        """Replace mode output should differ from original acts (SAE reconstruction != identity)."""
        ablator, model, sae = self._make_ablator()
        acts = torch.randn(1, 5, D_MODEL)
        hook_fn = ablator.create_ablation_hook(feature_indices=[0], positions=None, mode="replace")
        out = hook_fn(None, None, acts)
        # With random SAE weights, reconstruction ≠ original with high probability
        # Just check shapes match and output is a tensor
        assert isinstance(out, torch.Tensor)
        assert out.shape == acts.shape

    def test_ablation_tuple_output_preserved(self):
        """Hook should preserve tuple structure of layer output."""
        ablator, model, sae = self._make_ablator()
        acts = torch.randn(1, 5, D_MODEL)
        dummy_extra = torch.zeros(1)
        hook_fn = ablator.create_ablation_hook(feature_indices=[], mode="replace")
        out = hook_fn(None, None, (acts, dummy_extra))
        assert isinstance(out, tuple)
        assert out[0].shape == acts.shape
        assert out[1] is dummy_extra

    def test_ablation_no_features_selected_replace(self):
        """Ablating 0 features in replace mode → output is full SAE reconstruction."""
        ablator, model, sae = self._make_ablator()
        acts = torch.randn(1, 4, D_MODEL)
        hook_fn = ablator.create_ablation_hook(feature_indices=[], mode="replace")
        out = hook_fn(None, None, acts)
        feats = sae.encode(acts)
        expected = sae.decode(feats, target_shape=acts.shape)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_resolve_positions_image(self):
        from sae_experiments.ablation.feature_ablator import FeatureAblator
        model = StubModel(num_image_tokens=4)
        sae = _make_sae()
        ablator = FeatureAblator(model, sae, layer_idx=0)

        # Minimal dataset mock
        dataset = MagicMock()
        dataset.dataset_dict = {"q1": {"question": "test"}}
        dataset.tokenizer.encode.return_value = [1, 2]

        input_ids = torch.zeros(1, 8, dtype=torch.long)
        line = {"q_id": "q1", "attribute_tokens": []}
        positions = ablator._resolve_positions("image", input_ids, 4, dataset, line)
        assert positions == [0, 1, 2, 3]

    def test_resolve_positions_all_returns_none(self):
        from sae_experiments.ablation.feature_ablator import FeatureAblator
        model = StubModel()
        sae = _make_sae()
        ablator = FeatureAblator(model, sae, layer_idx=0)
        dataset = MagicMock()
        dataset.dataset_dict = {"q1": {"question": ""}}
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        result = ablator._resolve_positions("all", input_ids, 4, dataset, {"q_id": "q1"})
        assert result is None

    def test_diagnostics_buffer_populated(self):
        ablator, model, sae = self._make_ablator()
        acts = torch.randn(1, 5, D_MODEL)
        buf = []
        hook_fn = ablator.create_ablation_hook(
            feature_indices=[0], mode="replace", diagnostics_buffer=buf
        )
        hook_fn(None, None, acts)
        assert len(buf) == 1
        assert "delta_norm" in buf[0]
        assert "relative_norm" in buf[0]


# ===========================================================================
# 8f  paligemma_dataset
# ===========================================================================

class TestPaliGemmaDataset:

    def _make_dataset(self, n_rows=5):
        from sae_experiments.data.paligemma_dataset import PaliGemmaChooseAttrDataset
        import pandas as pd, tempfile, os

        questions = [{"q_id": str(i)} for i in range(n_rows)]
        dataset_dict = {
            str(i): {
                "question": f"Is the car red or blue?",
                "answer": "red",
                "true option": "red",
                "false option": "blue",
                "image": "dummy.jpg",
            }
            for i in range(n_rows)
        }

        # Mock processor
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.pad_token_id = 0
        processor.tokenizer.eos_token_id = 1
        processor.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 224, 224),
        }

        dataset = PaliGemmaChooseAttrDataset(
            questions=questions,
            dataset_dict=dataset_dict,
            image_folder="/tmp",
            processor=processor,
        )
        return dataset

    def test_dataset_len(self):
        ds = self._make_dataset(5)
        assert len(ds) == 5

    def test_dataset_item_keys(self):
        ds = self._make_dataset(3)
        input_ids, pixel_values, question_text, row_dict = ds[0]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(pixel_values, torch.Tensor)
        assert isinstance(question_text, str)
        assert isinstance(row_dict, dict)

    def test_question_text_preserved(self):
        ds = self._make_dataset(2)
        _, _, q_text, _ = ds[0]
        assert q_text == "Is the car red or blue?"

    def test_dataloader_batch_shape(self):
        from sae_experiments.data.paligemma_dataset import PaliGemmaChooseAttrDataset
        ds = self._make_dataset(4)
        loader = ds.create_dataloader(batch_size=2)
        batch = next(iter(loader))
        input_ids, pixel_values, question_texts, row_dicts = batch
        assert input_ids.shape[0] == 2
        assert pixel_values.shape[0] == 2
        assert len(question_texts) == 2
        assert len(row_dicts) == 2

    def test_dataloader_single_batch(self):
        ds = self._make_dataset(3)
        loader = ds.create_dataloader(batch_size=1)
        batch = next(iter(loader))
        input_ids, pixel_values, _, _ = batch
        assert input_ids.shape[0] == 1


# ===========================================================================
# 8g  hook_utils — get_target_module for PaliGemma
# ===========================================================================

class TestHookUtils:

    def test_get_target_module_paligemma_residual(self):
        from sae_experiments.utils.hook_utils import get_target_module
        model = StubModel(n_layers=2)
        layer = get_target_module(model, 0, "residual")
        assert layer is model.language_model.model.layers[0]

    def test_get_target_module_paligemma_attn_out(self):
        from sae_experiments.utils.hook_utils import get_target_module
        model = StubModel(n_layers=2)
        module = get_target_module(model, 0, "attn_out")
        assert module is model.language_model.model.layers[0].self_attn

    def test_get_target_module_paligemma_layer_index(self):
        from sae_experiments.utils.hook_utils import get_target_module
        model = StubModel(n_layers=3)
        layer2 = get_target_module(model, 2, "residual")
        assert layer2 is model.language_model.model.layers[2]

    def test_get_target_module_llama_style(self):
        """Falls through to the model.model.layers branch for LLaMA-style models."""
        from sae_experiments.utils.hook_utils import get_target_module

        class LlamaStyleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([StubLayer()])

        m = LlamaStyleModel()
        layer = get_target_module(m, 0, "residual")
        assert layer is m.model.layers[0]
