"""
Tests for math utilities to improve coverage.
"""

import pytest
import torch

from dendritic_modeling.utils.math.auc import (
    _binary_roc_auc_pytorch,
    _multiclass_roc_auc_pytorch,
    roc_auc_score,
)


class TestAUCUtils:
    """Test AUC computation utilities."""

    def test_roc_auc_score_binary_perfect(self):
        """Test AUC computation for perfect binary classification."""
        # Perfect classifier: all positive scores > all negative scores
        scores = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = torch.tensor([1, 1, 1, 0, 0, 0])

        auc = roc_auc_score(labels, scores)

        assert torch.isclose(auc, torch.tensor(1.0), atol=1e-6)

    def test_roc_auc_score_binary_random(self):
        """Test AUC computation for random binary classification."""
        # Random classifier should have AUC around 0.5
        torch.manual_seed(42)
        scores = torch.rand(100)
        labels = torch.randint(0, 2, (100,))

        auc = roc_auc_score(labels, scores)

        # Random classifier should be roughly around 0.5
        assert 0.3 < auc < 0.7

    def test_roc_auc_score_binary_worst(self):
        """Test AUC computation for worst binary classification."""
        # Worst classifier: all positive scores < all negative scores
        scores = torch.tensor([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = torch.tensor([1, 1, 1, 0, 0, 0])

        auc = roc_auc_score(labels, scores)

        assert torch.isclose(auc, torch.tensor(0.0), atol=1e-6)

    def test_binary_roc_auc_pytorch_basic(self):
        """Test binary ROC AUC PyTorch implementation."""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        labels = torch.tensor([1, 1, 0, 1, 0, 0])

        auc = _binary_roc_auc_pytorch(labels, scores)

        assert 0.0 <= auc <= 1.0
        assert isinstance(auc, torch.Tensor)

    def test_multiclass_roc_auc_pytorch_basic(self):
        """Test multiclass ROC AUC PyTorch implementation."""
        # 3-class problem
        scores = torch.tensor(
            [
                [0.8, 0.1, 0.1],  # Class 0
                [0.1, 0.8, 0.1],  # Class 1
                [0.1, 0.1, 0.8],  # Class 2
                [0.7, 0.2, 0.1],  # Class 0
                [0.2, 0.7, 0.1],  # Class 1
                [0.1, 0.2, 0.7],  # Class 2
            ]
        )
        labels = torch.tensor([0, 1, 2, 0, 1, 2])

        auc = _multiclass_roc_auc_pytorch(labels, scores)

        assert 0.0 <= auc <= 1.0
        assert isinstance(auc, torch.Tensor)

    def test_roc_auc_score_multiclass(self):
        """Test roc_auc_score with multiclass input."""
        scores = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        labels = torch.tensor([0, 1, 2])

        auc = roc_auc_score(labels, scores, multi_class="ovr")

        assert 0.0 <= auc <= 1.0


class TestAUCEdgeCases:
    """Test edge cases for AUC computation."""

    def test_roc_auc_score_all_same_labels(self):
        """Test AUC with all same labels."""
        scores = torch.tensor([0.1, 0.5, 0.9])
        labels = torch.tensor([1, 1, 1])  # All positive

        # Should handle gracefully - AUC undefined but function should not crash
        try:
            auc = roc_auc_score(labels, scores)
            # If it returns a value, it should be nan or handle appropriately
            assert torch.isnan(auc) or auc == 0.5
        except (ValueError, ZeroDivisionError):
            # It's acceptable to raise an error for this edge case
            pass

    def test_roc_auc_score_single_sample(self):
        """Test AUC with single sample."""
        scores = torch.tensor([0.5])
        labels = torch.tensor([1])

        # Should handle gracefully
        try:
            roc_auc_score(labels, scores)
        except (ValueError, IndexError):
            # Acceptable to raise error for insufficient data
            pass

    def test_roc_auc_score_empty_input(self):
        """Test AUC with empty input."""
        scores = torch.tensor([])
        labels = torch.tensor([])

        with pytest.raises((ValueError, RuntimeError)):
            roc_auc_score(labels, scores)

    def test_roc_auc_score_mismatched_dimensions(self):
        """Test AUC with mismatched score and label dimensions."""
        scores = torch.tensor([0.1, 0.5])
        labels = torch.tensor([1, 0, 1])  # Different length

        with pytest.raises((ValueError, RuntimeError)):
            roc_auc_score(labels, scores)

    def test_binary_roc_auc_pytorch_edge_case(self):
        """Test binary ROC AUC with edge cases."""
        # All same class
        scores = torch.tensor([0.1, 0.5, 0.9])
        labels = torch.tensor([1, 1, 1])

        try:
            auc = _binary_roc_auc_pytorch(labels, scores)
            # Should handle gracefully
            assert torch.isnan(auc) or auc == 0.5
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise error
            pass


class TestAUCNumericalStability:
    """Test numerical stability of AUC computations."""

    def test_roc_auc_score_extreme_values(self):
        """Test AUC computation with extreme score values."""
        scores = torch.tensor([1e10, 1e-10, 0.0, 1.0])
        labels = torch.tensor([1, 0, 0, 1])

        auc = roc_auc_score(labels, scores)

        assert 0.0 <= auc <= 1.0
        assert not torch.isnan(auc)

    def test_roc_auc_score_duplicate_scores(self):
        """Test AUC computation with many duplicate scores."""
        scores = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        labels = torch.tensor([1, 0, 1, 0, 1])

        auc = roc_auc_score(labels, scores)

        # Should be around 0.5 for tied scores
        assert torch.isclose(auc, torch.tensor(0.5), atol=0.1)

    def test_binary_roc_auc_pytorch_stability(self):
        """Test numerical stability of binary ROC AUC implementation."""
        torch.manual_seed(42)
        scores = torch.rand(100) * 1000  # Large values
        labels = torch.randint(0, 2, (100,))

        auc = _binary_roc_auc_pytorch(labels, scores)

        assert 0.0 <= auc <= 1.0
        assert not torch.isnan(auc)
