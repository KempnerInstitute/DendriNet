import unittest
import torch
from torch.utils.data import TensorDataset
from torch.optim import Adam

from dendritic_modeling.training import TrainerMLE


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.layer = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.layer(x)
    
    def log_prob(self, inputs, labels):
        logits = self.forward(inputs)
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return probs[range(labels.shape[0]), labels]

class TestTrainerMLE(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.optimizer = Adam(self.model.parameters())
        self.trainer = TrainerMLE(self.optimizer, suppress_prints=True)

        # Generate mock data
        inputs = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        self.train_data = TensorDataset(inputs, labels)
        self.valid_data = TensorDataset(inputs, labels)

    def test_train_runs(self):
        results = self.trainer.train(
            self.model, self.train_data, self.valid_data, epochs=5, 
            batch_size=10
        )
        self.assertIn("best epoch", results)
        self.assertIn("train losses", results)
        self.assertIn("valid losses", results)
        self.assertTrue(len(results["train losses"]) > 0)