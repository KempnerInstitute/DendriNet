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
        """One class: the AUC is undefined, so the score refuses it."""
        scores = torch.tensor([0.1, 0.5, 0.9])
        labels = torch.tensor([1, 1, 1])
        with pytest.raises(ValueError, match="at least 2 classes"):
            roc_auc_score(labels, scores)

    def test_roc_auc_score_single_sample(self):
        with pytest.raises(ValueError, match="at least 2 classes"):
            roc_auc_score(torch.tensor([1]), torch.tensor([0.5]))

    def test_rows_with_one_class_are_undefined(self):
        labels = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]])
        scores = torch.tensor([[0.1, 0.9, 0.8, 0.2], [0.3, 0.4, 0.5, 0.6]])
        auc = roc_auc_score(labels, scores)
        assert auc[0] == 1.0
        assert torch.isnan(auc[1])

    @pytest.mark.parametrize("pair", [(1, 2), (-1, 1), (0, 1)])
    def test_the_greater_binary_label_is_positive(self, pair):
        negative, positive = pair
        scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
        labels = torch.tensor([negative, negative, positive, positive])
        assert roc_auc_score(labels, scores) == 0.75

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

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            roc_auc_score(labels, scores)

    def test_binary_roc_auc_pytorch_edge_case(self):
        """All one class: the AUC is undefined (NaN), not chance level."""
        scores = torch.tensor([0.1, 0.5, 0.9])
        labels = torch.tensor([1, 1, 1])
        assert torch.isnan(_binary_roc_auc_pytorch(labels, scores))


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


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_auc_ranks_by_probabilities_like_sklearn(n_classes):
    """A per-sample logit offset leaves the probabilities and the AUC unchanged."""
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc

    from dendritic_modeling.training.utils._evaluation.classification import (
        _compute_auc,
    )

    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, n_classes, (300,), generator=generator)
    logits = torch.randn(300, n_classes, generator=generator)
    logits = logits + 1.5 * torch.nn.functional.one_hot(labels, n_classes)
    logits = logits + 5.0 * torch.randn(300, 1, generator=generator)

    auc = float(_compute_auc(torch.nn.Identity(), logits, labels))

    probabilities = torch.softmax(logits, dim=-1).numpy()
    if n_classes == 2:
        expected = sklearn_roc_auc(labels.numpy(), probabilities[:, 1])
    else:
        expected = sklearn_roc_auc(
            labels.numpy(), probabilities, multi_class="ovr", average="macro"
        )
    assert auc == pytest.approx(expected, abs=1e-6)
