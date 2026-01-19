import unittest
import torch
from torch import nn

from sae_experiments.ablation.feature_ablator import FeatureAblator
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder


class DummyModel(nn.Module):
    def __init__(self, d_model=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model)])

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        x = torch.randn(1, 2, self.layers[0].in_features)
        return self.layers[0](x)


class TestFeatureAblator(unittest.TestCase):
    def test_ablation_hook_shape(self):
        model = DummyModel(d_model=4)
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(model, sae, layer_idx=0)

        acts = torch.randn(1, 2, 4)
        hook = ablator.create_ablation_hook([0, 1, 2])
        output = hook(None, None, acts)
        self.assertEqual(output.shape, acts.shape)


if __name__ == "__main__":
    unittest.main()
