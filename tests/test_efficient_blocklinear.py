import unittest

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)


class TestEfficientBlockLinear(unittest.TestCase):
    """Unit tests comparing EfficientBlockLinear to the reference BlockLinear."""

    def _parity_check(self, out_features: int, block_size: int, batch_size: int = 4):
        in_features = out_features * block_size
        ref = BlockLinear(in_features, out_features)
        eff = EfficientBlockLinear(in_features, out_features)

        # Initialise with identical random weights
        torch.manual_seed(123)
        rand_w = torch.randn_like(ref.log_weight)
        ref.log_weight.data.copy_(rand_w)
        eff.log_weight.data.copy_(rand_w)

        # Forward pass
        x = torch.randn(batch_size, in_features)
        y_ref = ref(x)
        y_eff = eff(x)
        self.assertTrue(
            torch.allclose(y_ref, y_eff, atol=1e-6),
            "Forward outputs mismatch between BlockLinear and EfficientBlockLinear",
        )

        # Backward pass parity
        y_ref.sum().backward()
        y_eff.sum().backward()
        self.assertTrue(
            torch.allclose(ref.log_weight.grad, eff.log_weight.grad, atol=1e-6),
            "Gradient mismatch between BlockLinear and EfficientBlockLinear",
        )

    def test_small(self):
        self._parity_check(out_features=4, block_size=3)

    def test_medium(self):
        self._parity_check(out_features=16, block_size=5)


if __name__ == "__main__":
    unittest.main()
