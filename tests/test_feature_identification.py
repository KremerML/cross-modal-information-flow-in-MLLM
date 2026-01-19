import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sae_experiments.feature_analysis.feature_identifier import FeatureIdentifier
from sae_experiments.models.sparse_autoencoder import SparseAutoencoder


class DummyDataset(Dataset):
    def __init__(self):
        self.questions = [
            {"q_id": "1", "question": "red cube", "answer": "red"},
            {"q_id": "2", "question": "blue cube", "answer": "blue"},
        ]
        self.dataset_dict = {q["q_id"]: q for q in self.questions}
        self.tokenizer = None

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        image_tensor = [torch.zeros(1, 3, 4, 4)]
        image_sizes = [(4, 4)]
        prompt = ""
        mask_tensor = torch.zeros(1, 3, 4, 4)
        return input_ids, image_tensor, image_sizes, prompt, mask_tensor

    def create_dataloader(self):
        return DataLoader(self, batch_size=1, shuffle=False)


class DummyModel(nn.Module):
    def __init__(self, d_model=8, vocab_size=10):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model)])

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        x = self.embed(input_ids)
        return self.layers[0](x)


class TestFeatureIdentifier(unittest.TestCase):
    def test_feature_activation_shapes(self):
        dataset = DummyDataset()
        model = DummyModel(d_model=8)
        sae = SparseAutoencoder(d_model=8, n_features=16, l1_coeff=1e-3)
        identifier = FeatureIdentifier(sae, model, dataset, layer_idx=0)

        acts, meta = identifier.compute_feature_activations(position_type="question")
        self.assertEqual(acts.shape[0], len(dataset.questions))
        self.assertEqual(acts.shape[1], 16)
        self.assertEqual(len(meta), len(dataset.questions))

    def test_discriminative_features(self):
        identifier = FeatureIdentifier(None, None, None, layer_idx=0)
        identifier.feature_acts = torch.tensor(
            [[2.0, 0.5], [1.8, 0.4], [0.2, 0.6], [0.1, 0.5]]
        ).numpy()
        identifier.metadata = [
            {"is_correct": True},
            {"is_correct": True},
            {"is_correct": False},
            {"is_correct": False},
        ]
        features = identifier.find_discriminative_features(threshold=2.0)
        self.assertIn(0, features)


if __name__ == "__main__":
    unittest.main()
