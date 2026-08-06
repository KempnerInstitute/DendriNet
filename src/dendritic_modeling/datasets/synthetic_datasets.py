"""
Synthetic datasets for theoretical validation experiments.

This module provides rich synthetic datasets for testing dendritic information
processing under controlled conditions. Datasets range from purely linear to
completely nonlinear classification tasks, with configurable correlation structures.

Key features:
- Linear to nonlinear classification tasks
- Controlled within-class correlations (rho_ee, rho_ii, rho_ei)
- Block-structured correlations (different correlation patterns in subsets)
- Balanced information across input dimensions
- Inputs bounded to [0,1] as required by theory
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset

from dendritic_modeling.utils.nested_tree_geometry import (
    dendritic_layer_sizes as _dendritic_layer_sizes,
    n_internal_nodes_from_branch_factors as _n_internal_nodes_from_branch_factors,
    n_leaves_from_branch_factors as _n_leaves_from_branch_factors,
    normalize_branch_factors as _normalize_branch_factors,
)

logger = logging.getLogger(__name__)

_SyntheticDatasetTriplet = tuple[TensorDataset, TensorDataset, TensorDataset]
_SyntheticDatasetFactory = Callable[..., _SyntheticDatasetTriplet]


@dataclass(frozen=True)
class _SyntheticDatasetSpec:
    """Constructor and default keyword values for one synthetic dataset."""

    factory: _SyntheticDatasetFactory
    defaults: Mapping[str, Any]

    def build(self, dataset_config: Mapping[str, Any]) -> _SyntheticDatasetTriplet:
        kwargs = {
            key: dataset_config.get(key, _copy_synthetic_default(default))
            for key, default in self.defaults.items()
        }
        return self.factory(**kwargs)


def _copy_synthetic_default(value: Any) -> Any:
    """Copy mutable factory defaults to preserve the old per-call literals."""
    if isinstance(value, list):
        return list(value)
    return value


def create_correlated_gaussian_dataset(
    n_samples: int = 1000,
    input_dim: int = 100,
    n_classes: int = 2,
    rho_ee: float = 0.0,
    rho_ii: float = 0.0,
    rho_ei: float = 0.0,
    ne: int = 50,
    ni: int = 50,
    class_separation: float = 0.3,
    noise_std: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create a synthetic dataset with controlled correlations between E/I dimensions.

    Args:
        n_samples: Number of samples per class
        input_dim: Total input dimensionality (should equal ne + ni)
        n_classes: Number of classes
        rho_ee: Correlation between excitatory dimensions
        rho_ii: Correlation between inhibitory dimensions
        rho_ei: Cross-correlation between excitatory and inhibitory dimensions
        ne: Number of excitatory dimensions
        ni: Number of inhibitory dimensions
        class_separation: Distance between class means
        noise_std: Standard deviation of within-class noise
        seed: Random seed for reproducibility

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert (
        ne + ni == input_dim
    ), f"ne ({ne}) + ni ({ni}) must equal input_dim ({input_dim})"

    # Create correlation matrix
    cov_matrix = torch.eye(input_dim)

    # Set within-excitatory correlations
    for i in range(ne):
        for j in range(i + 1, ne):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ee

    # Set within-inhibitory correlations
    for i in range(ne, ne + ni):
        for j in range(i + 1, ne + ni):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ii

    # Set cross E-I correlations
    for i in range(ne):
        for j in range(ne, ne + ni):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ei

    # Ensure positive semi-definite
    eigenvals = torch.linalg.eigvals(cov_matrix).real
    min_eigenval = eigenvals.min()
    if min_eigenval <= 0:
        logger.warning(
            f"Correlation matrix not PSD (min eigenval: {min_eigenval}), adding regularization"
        )
        cov_matrix += torch.eye(input_dim) * (1e-6 - min_eigenval)

    # Scale by noise variance
    cov_matrix = cov_matrix * (noise_std**2)

    # Generate class means
    class_means = []
    for c in range(n_classes):
        # Create class-specific mean pattern
        base_mean = torch.zeros(input_dim)
        # Simple pattern: alternate classes have different mean patterns
        if c % 2 == 0:
            base_mean[:ne] = class_separation  # Higher excitatory activity
            base_mean[ne:] = class_separation * 0.5  # Lower inhibitory activity
        else:
            base_mean[:ne] = class_separation * 0.5  # Lower excitatory activity
            base_mean[ne:] = class_separation  # Higher inhibitory activity
        class_means.append(base_mean)

    # Generate data
    all_data = []
    all_labels = []

    for c in range(n_classes):
        # Sample from multivariate normal
        dist = MultivariateNormal(class_means[c], cov_matrix)
        class_data = dist.sample((n_samples,))
        # Clamp to [0, 1] range as required by theory
        class_data = torch.clamp(class_data, 0, 1)

        all_data.append(class_data)
        all_labels.append(torch.full((n_samples,), c, dtype=torch.long))

    # Combine and shuffle
    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Shuffle
    perm = torch.randperm(len(data))
    data = data[perm]
    labels = labels[perm]

    # Split into train/valid/test
    n_total = len(data)
    n_train = int(0.6 * n_total)
    n_valid = int(0.2 * n_total)

    train_data = data[:n_train]
    train_labels = labels[:n_train]
    valid_data = data[n_train : n_train + n_valid]
    valid_labels = labels[n_train : n_train + n_valid]
    test_data = data[n_train + n_valid :]
    test_labels = labels[n_train + n_valid :]

    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    return train_dataset, valid_dataset, test_dataset


def create_nonlinear_interaction_dataset(
    n_samples: int = 1000,
    input_dim: int = 2,
    n_classes: int = 2,
    interaction_strength: float = 1.0,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create a dataset requiring nonlinear classification (e.g., XOR-like).
    Used to test linearity improvements with shunting.

    Args:
        n_samples: Number of samples per class
        input_dim: Input dimensionality (typically 2 for XOR)
        n_classes: Number of classes
        interaction_strength: Strength of nonlinear interaction
        noise_std: Standard deviation of noise
        seed: Random seed

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate base data uniformly in [0,1]^d
    total_samples = n_samples * n_classes
    data = torch.rand(total_samples, input_dim)

    # Create nonlinear decision boundary
    # For 2D: class depends on x1 * x2 interaction
    if input_dim == 2:
        x1, x2 = data[:, 0], data[:, 1]
        # XOR-like pattern: class 1 if x1*x2 > threshold
        threshold = 0.25
        labels = (x1 * x2 * interaction_strength > threshold).long()
    else:
        # For higher dimensions: use product of pairs
        products = []
        for i in range(0, input_dim - 1, 2):
            if i + 1 < input_dim:
                products.append(data[:, i] * data[:, i + 1])
        if products:
            interaction_score = torch.stack(products).mean(dim=0)
            labels = (interaction_score * interaction_strength > 0.25).long()
        else:
            # Fallback: use sum of squares
            labels = (data.pow(2).sum(dim=1) > 0.5).long()

    # Add noise
    noise = torch.randn_like(data) * noise_std
    data = torch.clamp(data + noise, 0, 1)

    # Balance classes
    class_0_mask = labels == 0
    class_1_mask = labels == 1

    n_class_0 = class_0_mask.sum()
    n_class_1 = class_1_mask.sum()
    min_class_size = min(n_class_0, n_class_1)

    # Subsample to balance
    class_0_indices = torch.where(class_0_mask)[0][:min_class_size]
    class_1_indices = torch.where(class_1_mask)[0][:min_class_size]

    balanced_indices = torch.cat([class_0_indices, class_1_indices])
    data = data[balanced_indices]
    labels = labels[balanced_indices]

    # Shuffle
    perm = torch.randperm(len(data))
    data = data[perm]
    labels = labels[perm]

    # Split
    n_total = len(data)
    n_train = int(0.6 * n_total)
    n_valid = int(0.2 * n_total)

    train_data = data[:n_train]
    train_labels = labels[:n_train]
    valid_data = data[n_train : n_train + n_valid]
    valid_labels = labels[n_train : n_train + n_valid]
    test_data = data[n_train + n_valid :]
    test_labels = labels[n_train + n_valid :]

    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    return train_dataset, valid_dataset, test_dataset


def create_linear_separable_dataset(
    n_samples: int = 1000,
    input_dim: int = 100,
    ne: int = 50,
    ni: int = 50,
    rho_ee: float = 0.0,
    rho_ii: float = 0.0,
    rho_ei: float = 0.0,
    signal_strength: float = 0.3,
    noise_std: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create a linearly separable dataset for testing linear vs nonlinear processing.

    Classes are separated by a hyperplane in the input space.
    This tests whether shunting provides advantages even for linear tasks.

    Args:
        n_samples: Number of samples per class
        input_dim: Total input dimensionality
        ne: Number of excitatory dimensions
        ni: Number of inhibitory dimensions
        rho_ee, rho_ii, rho_ei: Correlation parameters
        signal_strength: Distance between class means
        noise_std: Within-class noise
        seed: Random seed

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert ne + ni == input_dim

    # Create linear decision boundary: class based on weighted sum
    # w^T x > threshold defines the classes

    # Random direction for class separation
    direction = torch.randn(input_dim)
    direction = direction / direction.norm()  # Normalize

    # Create samples with correlation structure
    cov_matrix = torch.eye(input_dim)

    # Set correlations
    for i in range(ne):
        for j in range(i + 1, ne):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ee

    for i in range(ne, ne + ni):
        for j in range(i + 1, ne + ni):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ii

    for i in range(ne):
        for j in range(ne, ne + ni):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ei

    # Ensure PSD
    eigenvals = torch.linalg.eigvals(cov_matrix).real
    if eigenvals.min() <= 0:
        cov_matrix += torch.eye(input_dim) * (1e-6 - eigenvals.min())

    cov_matrix = cov_matrix * (noise_std**2)

    # Generate samples for each class
    all_data = []
    all_labels = []

    for class_id in range(2):
        # Class means separated along direction
        if class_id == 0:
            class_mean = -direction * signal_strength / 2
        else:
            class_mean = direction * signal_strength / 2

        # Add base offset to ensure positive values
        class_mean += 0.5

        # Sample from multivariate normal
        dist = MultivariateNormal(class_mean, cov_matrix)
        class_data = dist.sample((n_samples,))
        class_data = torch.clamp(class_data, 0, 1)

        all_data.append(class_data)
        all_labels.append(torch.full((n_samples,), class_id, dtype=torch.long))

    # Combine and split
    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    perm = torch.randperm(len(data))
    data, labels = data[perm], labels[perm]

    return _split_dataset(data, labels)


def create_nonlinear_separable_dataset(
    n_samples: int = 1000,
    input_dim: int = 100,
    ne: int = 50,
    ni: int = 50,
    nonlinearity_type: str = "quadratic",
    rho_ee: float = 0.0,
    rho_ii: float = 0.0,
    rho_ei: float = 0.0,
    noise_std: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create a nonlinearly separable dataset requiring nonlinear classification.

    Tests shunting advantages for nonlinear tasks. Classification requires
    detecting interactions between input dimensions (e.g., XOR-like patterns,
    quadratic boundaries, multiplicative interactions).

    Args:
        n_samples: Number of samples per class
        input_dim: Total input dimensionality
        ne: Number of excitatory dimensions
        ni: Number of inhibitory dimensions
        nonlinearity_type: Type of nonlinearity
            - "quadratic": Quadratic decision boundary
            - "xor": XOR-like pattern across dimension pairs
            - "multiplicative": Multiplicative interactions
            - "radial": Radial basis function separation
        rho_ee, rho_ii, rho_ei: Correlation parameters
        noise_std: Within-class noise
        seed: Random seed

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert ne + ni == input_dim

    # Generate base data uniformly
    total_samples = n_samples * 2
    data = torch.rand(total_samples, input_dim)

    # Apply nonlinear decision boundary
    if nonlinearity_type == "quadratic":
        # Decision based on quadratic form: x^T A x > threshold
        A = torch.randn(input_dim, input_dim)
        A = (A + A.T) / 2  # Symmetric matrix
        scores = torch.sum(data @ A * data, dim=1)
        threshold = scores.median()
        labels = (scores > threshold).long()

    elif nonlinearity_type == "xor":
        # XOR across dimension pairs
        scores = torch.zeros(total_samples)
        for i in range(0, input_dim - 1, 2):
            scores += (data[:, i] > 0.5).float() != (data[:, i + 1] > 0.5).float()
        labels = (scores > (input_dim / 4)).long()

    elif nonlinearity_type == "multiplicative":
        # Multiplicative interactions in log space avoid high-dimensional
        # product underflow and keep the generated classes balanced.
        eps = torch.finfo(data.dtype).tiny
        scores = torch.log(data[:, :ne].clamp_min(eps)).sum(dim=1)
        scores = scores + torch.log(data[:, ne:].clamp_min(eps)).sum(dim=1)
        threshold = scores.median()
        labels = (scores > threshold).long()

    elif nonlinearity_type == "radial":
        # Radial basis: distance from center
        center = torch.full((input_dim,), 0.5)
        distances = torch.norm(data - center, dim=1)
        threshold = distances.median()
        labels = (distances > threshold).long()

    else:
        raise ValueError(f"Unknown nonlinearity type: {nonlinearity_type}")

    # Add correlated noise
    if noise_std > 0:
        cov_matrix = _create_correlation_matrix(
            input_dim, ne, ni, rho_ee, rho_ii, rho_ei, noise_std
        )
        noise_dist = MultivariateNormal(torch.zeros(input_dim), cov_matrix)
        noise = noise_dist.sample((total_samples,))
        data = torch.clamp(data + noise, 0, 1)

    # Balance classes
    data, labels = _balance_classes(data, labels)

    return _split_dataset(data, labels)


def create_block_correlated_dataset(
    n_samples: int = 1000,
    input_dim: int = 100,
    ne: int = 50,
    ni: int = 50,
    n_blocks: int = 5,
    block_rho_ee: Optional[list[float]] = None,
    block_rho_ii: Optional[list[float]] = None,
    block_rho_ei: Optional[list[float]] = None,
    class_separation: float = 0.3,
    noise_std: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create dataset with block-structured correlations.

    Different subsets of dimensions have different correlation structures.
    Tests optimal correlation patterns and whether neurons can learn to
    sample from informative vs uninformative blocks.

    Args:
        n_samples: Number of samples per class
        input_dim: Total input dimensionality
        ne: Number of excitatory dimensions
        ni: Number of inhibitory dimensions
        n_blocks: Number of correlation blocks
        block_rho_ee: List of E-E correlations for each block
        block_rho_ii: List of I-I correlations for each block
        block_rho_ei: List of E-I correlations for each block
        class_separation: Signal strength
        noise_std: Noise level
        seed: Random seed

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert ne + ni == input_dim

    # Default correlation values if not provided
    if block_rho_ee is None:
        block_rho_ee = [0.0] * n_blocks
    if block_rho_ii is None:
        block_rho_ii = [0.0] * n_blocks
    if block_rho_ei is None:
        block_rho_ei = [0.0] * n_blocks

    # Create block-structured covariance matrix
    cov_matrix = torch.eye(input_dim)

    # Divide dimensions into blocks
    e_block_size = ne // n_blocks
    i_block_size = ni // n_blocks

    for block_idx in range(n_blocks):
        # Get correlation for this block
        rho_ee_block = block_rho_ee[min(block_idx, len(block_rho_ee) - 1)]
        rho_ii_block = block_rho_ii[min(block_idx, len(block_rho_ii) - 1)]
        rho_ei_block = block_rho_ei[min(block_idx, len(block_rho_ei) - 1)]

        # E dimensions in this block
        e_start = block_idx * e_block_size
        e_end = min((block_idx + 1) * e_block_size, ne)

        # I dimensions in this block
        i_start = ne + block_idx * i_block_size
        i_end = min(ne + (block_idx + 1) * i_block_size, input_dim)

        # Set E-E correlations within block
        for i in range(e_start, e_end):
            for j in range(i + 1, e_end):
                cov_matrix[i, j] = cov_matrix[j, i] = rho_ee_block

        # Set I-I correlations within block
        for i in range(i_start, i_end):
            for j in range(i + 1, i_end):
                cov_matrix[i, j] = cov_matrix[j, i] = rho_ii_block

        # Set E-I correlations within block
        for i in range(e_start, e_end):
            for j in range(i_start, i_end):
                cov_matrix[i, j] = cov_matrix[j, i] = rho_ei_block

    # Ensure PSD
    eigenvals = torch.linalg.eigvals(cov_matrix).real
    if eigenvals.min() <= 0:
        cov_matrix += torch.eye(input_dim) * (1e-6 - eigenvals.min())

    cov_matrix = cov_matrix * (noise_std**2)

    # Generate class means - different blocks carry different amounts of signal
    class_means = []
    for class_id in range(2):
        class_mean = torch.zeros(input_dim)

        for block_idx in range(n_blocks):
            # Signal strength varies across blocks
            block_signal = class_separation * (
                1.0 - 0.15 * block_idx
            )  # Decreasing signal

            e_start = block_idx * e_block_size
            e_end = min((block_idx + 1) * e_block_size, ne)
            i_start = ne + block_idx * i_block_size
            i_end = min(ne + (block_idx + 1) * i_block_size, input_dim)

            if class_id == 0:
                class_mean[e_start:e_end] = block_signal
                class_mean[i_start:i_end] = block_signal * 0.5
            else:
                class_mean[e_start:e_end] = block_signal * 0.5
                class_mean[i_start:i_end] = block_signal

        class_means.append(class_mean)

    # Generate samples
    all_data = []
    all_labels = []

    for class_id in range(2):
        dist = MultivariateNormal(class_means[class_id], cov_matrix)
        class_data = dist.sample((n_samples,))
        class_data = torch.clamp(class_data, 0, 1)

        all_data.append(class_data)
        all_labels.append(torch.full((n_samples,), class_id, dtype=torch.long))

    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    perm = torch.randperm(len(data))
    data, labels = data[perm], labels[perm]

    return _split_dataset(data, labels)


def create_balanced_information_dataset(
    n_samples: int = 1000,
    input_dim: int = 100,
    ne: int = 50,
    ni: int = 50,
    info_per_dim: str = "balanced",
    rho_ee: float = 0.0,
    rho_ii: float = 0.0,
    rho_ei: float = 0.0,
    noise_std: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create dataset with balanced or varying information across dimensions.

    Tests whether neurons can optimally select informative dimensions and
    whether information is distributed uniformly or concentrated.

    Args:
        n_samples: Number of samples per class
        input_dim: Total input dimensionality
        ne: Number of excitatory dimensions
        ni: Number of inhibitory dimensions
        info_per_dim: Information distribution mode:
            - "balanced": All dimensions carry equal information
            - "concentrated": Information concentrated in few dimensions
            - "gradient": Information decreases across dimensions
        rho_ee, rho_ii, rho_ei: Correlation parameters
        noise_std: Noise level
        seed: Random seed

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert ne + ni == input_dim

    # Create correlation matrix
    cov_matrix = _create_correlation_matrix(
        input_dim, ne, ni, rho_ee, rho_ii, rho_ei, noise_std
    )

    # Generate class means based on information distribution
    class_means = []

    for class_id in range(2):
        class_mean = torch.zeros(input_dim)

        if info_per_dim == "balanced":
            # All dimensions carry equal information
            signal_per_dim = 0.3 / np.sqrt(input_dim)  # Normalized signal
            if class_id == 0:
                class_mean[:] = signal_per_dim
            else:
                class_mean[:] = -signal_per_dim
            # Offset to positive range
            class_mean += 0.5

        elif info_per_dim == "concentrated":
            # Information concentrated in first 20% of dimensions
            n_informative = input_dim // 5
            signal = 0.4

            if class_id == 0:
                class_mean[:n_informative] = signal
                class_mean[n_informative:] = signal * 0.1  # Low signal
            else:
                class_mean[:n_informative] = signal * 0.1
                class_mean[n_informative:] = signal

        elif info_per_dim == "gradient":
            # Information decreases linearly across dimensions
            for dim_idx in range(input_dim):
                info_factor = 1.0 - (dim_idx / input_dim) * 0.8  # 100% to 20%
                signal = 0.3 * info_factor

                if class_id == 0:
                    class_mean[dim_idx] = 0.3 + signal
                else:
                    class_mean[dim_idx] = 0.7 - signal

        class_means.append(class_mean)

    # Generate samples
    all_data = []
    all_labels = []

    for class_id in range(2):
        dist = MultivariateNormal(class_means[class_id], cov_matrix)
        class_data = dist.sample((n_samples,))
        class_data = torch.clamp(class_data, 0, 1)

        all_data.append(class_data)
        all_labels.append(torch.full((n_samples,), class_id, dtype=torch.long))

    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    perm = torch.randperm(len(data))
    data, labels = data[perm], labels[perm]

    return _split_dataset(data, labels)


# Helper functions
def _create_correlation_matrix(
    input_dim: int,
    ne: int,
    ni: int,
    rho_ee: float,
    rho_ii: float,
    rho_ei: float,
    noise_std: float,
) -> torch.Tensor:
    """Create correlation matrix with specified structure."""
    cov_matrix = torch.eye(input_dim)

    # E-E correlations
    for i in range(ne):
        for j in range(i + 1, ne):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ee

    # I-I correlations
    for i in range(ne, ne + ni):
        for j in range(i + 1, ne + ni):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ii

    # E-I correlations
    for i in range(ne):
        for j in range(ne, ne + ni):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ei

    # Ensure PSD
    eigenvals = torch.linalg.eigvals(cov_matrix).real
    if eigenvals.min() <= 0:
        logger.warning(
            f"Correlation matrix not PSD (min eigenval: {eigenvals.min()}), adding regularization"
        )
        cov_matrix += torch.eye(input_dim) * (1e-6 - eigenvals.min())

    return cov_matrix * (noise_std**2)


def _balance_classes(
    data: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance class sizes."""
    class_0_mask = labels == 0
    class_1_mask = labels == 1

    n_class_0 = class_0_mask.sum()
    n_class_1 = class_1_mask.sum()
    min_class_size = min(n_class_0, n_class_1)

    class_0_indices = torch.where(class_0_mask)[0][:min_class_size]
    class_1_indices = torch.where(class_1_mask)[0][:min_class_size]

    balanced_indices = torch.cat([class_0_indices, class_1_indices])
    return data[balanced_indices], labels[balanced_indices]


def create_heterogeneous_information_dataset(
    n_samples: int = 2000,
    input_dim: int = 100,
    n_classes: int = 10,
    ne: int = 50,
    ni: int = 50,
    n_high_info_dims: int = 10,
    high_info_strength: float = 2.0,
    low_info_strength: float = 0.1,
    noise_std: float = 0.2,
    rho_ee: float = 0.0,
    rho_ii: float = 0.0,
    rho_ei: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create dataset with heterogeneous information across dimensions.

    Different dimensions carry different amounts of class-relevant information.
    Tests optimal synapse sampling strategies for shunting vs. linear.

    Args:
        n_samples: Samples per class
        input_dim: Total dimensions (ne + ni)
        n_classes: Number of classes
        ne, ni: E and I dimension counts
        n_high_info_dims: Number of highly informative dimensions
        high_info_strength: Signal strength for high-info dims
        low_info_strength: Signal strength for low-info dims
        noise_std: Within-class noise
        rho_ee, rho_ii, rho_ei: Correlation structure
        seed: Random seed

    Returns:
        train, valid, test datasets
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert ne + ni == input_dim

    # Create correlation matrix
    cov_matrix = _create_correlation_matrix(
        input_dim, ne, ni, rho_ee, rho_ii, rho_ei, noise_std
    )

    # Create heterogeneous class means
    all_data = []
    all_labels = []

    for class_id in range(n_classes):
        class_mean = torch.zeros(input_dim)

        # High-information dimensions get strong class-specific signal
        # Low-information dimensions get weak signal
        for j in range(input_dim):
            if j < n_high_info_dims:
                # High information: strong class-dependent signal
                class_mean[j] = high_info_strength * (class_id / n_classes)
            else:
                # Low information: weak signal
                class_mean[j] = low_info_strength * (class_id / n_classes)

        # Generate samples
        dist = MultivariateNormal(class_mean, cov_matrix)
        class_data = dist.sample((n_samples,))
        class_data = torch.clamp(class_data, 0, 1)

        all_data.append(class_data)
        all_labels.append(torch.full((n_samples,), class_id, dtype=torch.long))

    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Shuffle
    perm = torch.randperm(len(data))
    data, labels = data[perm], labels[perm]

    return _split_dataset(data, labels)


def create_nonlinear_transformation_dataset(
    n_samples: int = 2000,
    n_classes: int = 2,
    nonlinearity_type: str = "xor",
    ne: int = 2,
    ni: int = 2,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create dataset specifically designed to test linearity transformation.

    Inputs have nonlinear structure (XOR, quadratic, multiplicative).
    Tests whether shunting output is more linear than input.

    Args:
        n_samples: Samples per class
        n_classes: Number of classes (2 for XOR)
        nonlinearity_type: 'xor', 'quadratic', 'multiplicative'
        ne, ni: E and I input dimensions
        noise_std: Noise level
        seed: Random seed

    Returns:
        train, valid, test datasets
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    input_dim = ne + ni
    all_data = []
    all_labels = []

    for _ in range(n_samples * n_classes):
        if nonlinearity_type == "xor":
            # XOR on first two dimensions
            x1 = torch.rand(1).item()
            x2 = torch.rand(1).item()
            label = int((x1 > 0.5) != (x2 > 0.5))  # XOR

            # E dimensions: linear combinations
            x_e = torch.tensor([x1, x2] + [torch.rand(1).item() for _ in range(ne - 2)])
            # I dimensions: multiplicative
            x_i = torch.tensor(
                [x1 * x2, (1 - x1) * (1 - x2)]
                + [torch.rand(1).item() for _ in range(ni - 2)]
            )

        elif nonlinearity_type == "quadratic":
            # Quadratic boundary: x1^2 + x2^2 > r^2
            x1 = torch.rand(1).item()
            x2 = torch.rand(1).item()
            r = 0.5
            label = int(x1**2 + x2**2 > r**2)

            x_e = torch.tensor([x1, x2] + [torch.rand(1).item() for _ in range(ne - 2)])
            x_i = torch.tensor(
                [x1**2, x2**2] + [torch.rand(1).item() for _ in range(ni - 2)]
            )

        elif nonlinearity_type == "multiplicative":
            # Multiplicative interactions
            x1 = torch.rand(1).item()
            x2 = torch.rand(1).item()
            x3 = torch.rand(1).item()
            label = int(x1 * x2 + x2 * x3 > 0.5)

            x_e = torch.tensor(
                [x1, x2, x3] + [torch.rand(1).item() for _ in range(max(0, ne - 3))]
            )
            x_i = torch.tensor(
                [x1 * x2, x2 * x3, x1 * x3]
                + [torch.rand(1).item() for _ in range(max(0, ni - 3))]
            )

        else:
            raise ValueError(f"Unknown nonlinearity type: {nonlinearity_type}")

        # Pad to correct dimensionality
        x = torch.cat([x_e[:ne], x_i[:ni]])

        # Add noise
        x = x + torch.randn(input_dim) * noise_std
        x = torch.clamp(x, 0, 1)

        all_data.append(x)
        all_labels.append(label)

    data = torch.stack(all_data)
    labels = torch.tensor(all_labels, dtype=torch.long)

    return _split_dataset(data, labels)


def create_asymmetric_noise_dataset(
    n_samples: int = 2000,
    input_dim: int = 100,
    n_classes: int = 10,
    ne: int = 50,
    ni: int = 50,
    sigma_e: float = 0.2,
    sigma_i: float = 0.1,
    class_separation: float = 0.2,
    rho_ee: float = 0.0,
    rho_ii: float = 0.0,
    rho_ei: float = 0.0,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create dataset with asymmetric noise in E vs I dimensions.

    Tests exact theoretical prediction: (I_E/I_I)* = (sigma_I/sigma_E)^2

    Args:
        n_samples: Samples per class
        input_dim: Total dimensions (ne + ni)
        n_classes: Number of classes
        ne, ni: E and I dimension counts
        sigma_e: Noise std for E dimensions (can differ from sigma_i)
        sigma_i: Noise std for I dimensions
        class_separation: Signal strength (same for E and I)
        rho_ee, rho_ii, rho_ei: Correlations
        seed: Random seed

    Returns:
        train, valid, test datasets
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert ne + ni == input_dim

    # Create separate covariance matrices for E and I
    cov_e = torch.eye(ne) * (sigma_e**2)
    cov_i = torch.eye(ni) * (sigma_i**2)

    # Add correlations within E
    for i in range(ne):
        for j in range(i + 1, ne):
            cov_e[i, j] = cov_e[j, i] = rho_ee * sigma_e**2

    # Add correlations within I
    for i in range(ni):
        for j in range(i + 1, ni):
            cov_i[i, j] = cov_i[j, i] = rho_ii * sigma_i**2

    # Build full covariance with cross-correlations
    cov_matrix = torch.zeros(input_dim, input_dim)
    cov_matrix[:ne, :ne] = cov_e
    cov_matrix[ne:, ne:] = cov_i

    # Add E-I cross-correlations
    for i in range(ne):
        for j in range(ne, input_dim):
            cov_matrix[i, j] = cov_matrix[j, i] = rho_ei * sigma_e * sigma_i

    # Ensure PSD
    eigenvals = torch.linalg.eigvals(cov_matrix).real
    if eigenvals.min() <= 0:
        logger.warning("Cov matrix not PSD, adding regularization")
        cov_matrix += torch.eye(input_dim) * (1e-6 - eigenvals.min())

    # Generate class means (same separation for E and I to isolate noise effect)
    all_data = []
    all_labels = []

    for class_id in range(n_classes):
        class_mean = torch.zeros(input_dim)
        # Same signal strength for E and I dimensions
        class_mean[:] = class_separation * (class_id / n_classes)

        # Generate samples
        dist = MultivariateNormal(class_mean, cov_matrix)
        class_data = dist.sample((n_samples,))
        class_data = torch.clamp(class_data, 0, 1)

        all_data.append(class_data)
        all_labels.append(torch.full((n_samples,), class_id, dtype=torch.long))

    data = torch.cat(all_data, dim=0)
    labels = torch.cat(all_labels, dim=0)

    perm = torch.randperm(len(data))
    data, labels = data[perm], labels[perm]

    return _split_dataset(data, labels)


def create_population_code_dataset(
    n_samples: int = 2000,
    pop_dim: int = 200,
    ne: int = 50,
    ni: int = 50,
    n_classes: int = 2,
    code_type: str = "mixed_selectivity",
    noise_std: float = 0.1,
    kappa: float = 2.0,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create datasets that simulate complex population codes from an upstream layer.

    This generator models cases reported in systems neuroscience where downstream
    neurons read out from complex, nonlinear population representations:
      - mixed_selectivity: random nonlinear mixing of latent variables (Rigotti et al. 2013)
      - ring: circular manifold with von Mises tuning (orientation/head-direction)

    The upstream population (size pop_dim) is mixed through random synaptic
    projections into E and I input dimensions (ne, ni), bounded to [0,1].

    Args:
        n_samples: Samples per class (balanced expected)
        pop_dim: Size of upstream population code
        ne: Number of excitatory synaptic inputs
        ni: Number of inhibitory synaptic inputs
        n_classes: Number of classes (2 supported)
        code_type: 'mixed_selectivity' or 'ring'
        noise_std: Observation noise level in synaptic inputs
        kappa: Concentration parameter for von Mises (ring)
        seed: Random seed

    Returns:
        train, valid, test datasets
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    assert (
        n_classes == 2
    ), "population_code dataset currently supports binary classification"

    total_samples = n_samples * n_classes

    # Upstream population responses r: (total_samples, pop_dim)
    if code_type == "mixed_selectivity":
        # Latent variables s in R^2
        S = torch.rand(total_samples, 2) * 2.0 - 1.0  # Uniform in [-1,1]^2

        # Binary labels are linear in latent space
        a = torch.randn(2)
        a = a / (a.norm() + 1e-8)
        y = (S @ a > 0).long()

        # Upstream nonlinear mixed selectivity: r = sigmoid(tanh(W s + b) + noise)
        W = torch.randn(pop_dim, 2)
        b = 0.2 * torch.randn(pop_dim)
        z = torch.tanh(S @ W.T + b)
        r = torch.sigmoid(z + torch.randn_like(z) * noise_std)

    elif code_type == "ring":
        # Latent variable: angle on a ring
        theta = torch.rand(total_samples) * 2 * torch.pi

        # Labels: two halves of the ring (linear in sin/cos space)
        y = (torch.cos(theta) > 0).long()

        # Upstream population: von Mises tuning curves with centers phi_k
        phi = torch.linspace(0, 2 * torch.pi, pop_dim, endpoint=False)
        # r_k(theta) = exp(kappa cos(theta - phi_k)), then scale to [0,1]
        cos_diffs = torch.cos(theta.view(-1, 1) - phi.view(1, -1))
        vm = torch.exp(kappa * cos_diffs)
        vm_min = torch.exp(torch.tensor(-kappa))
        vm_max = torch.exp(torch.tensor(kappa))
        r = (vm - vm_min) / (vm_max - vm_min + 1e-8)
        r = torch.clamp(r + torch.randn_like(r) * noise_std, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown population code_type: {code_type}")

    # Random E/I mixing from population to synapses
    # Use small weights to keep values in a reasonable range before sigmoid
    A_e = torch.randn(ne, pop_dim) / np.sqrt(pop_dim)
    A_i = torch.randn(ni, pop_dim) / np.sqrt(pop_dim)
    b_e = 0.1 * torch.randn(ne)
    b_i = 0.1 * torch.randn(ni)

    x_e = torch.sigmoid(r @ A_e.T + b_e + torch.randn(total_samples, ne) * noise_std)
    x_i = torch.sigmoid(r @ A_i.T + b_i + torch.randn(total_samples, ni) * noise_std)
    data = torch.cat([x_e, x_i], dim=1)
    labels = y

    # Balance classes
    data, labels = _balance_classes(data, labels)

    # Split
    return _split_dataset(data, labels)


def create_contextual_stream_gain_shift_dataset(
    n_samples: int = 4000,
    stream_dim: int = 64,
    n_classes: int = 2,
    signal_strength: float = 0.35,
    relevant_noise_std: float = 0.08,
    irrelevant_noise_std: float = 0.18,
    train_gain_relevant_min: float = 0.9,
    train_gain_relevant_max: float = 1.1,
    train_gain_irrelevant_min: float = 0.8,
    train_gain_irrelevant_max: float = 1.2,
    test_gain_relevant: float = 1.0,
    test_gain_irrelevant: float = 3.0,
    test_irrelevant_alignment_alpha: float = 1.0,
    ood_mode: str = "irrelevant",
    valid_split_mode: str = "ood",
    context_signal_scale: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Create a two-stream context-gated classification task with branch-local gain shifts.

    Each sample contains:
      [stream_A | stream_B | context_onehot]

    The binary context cue selects which stream is behaviorally relevant. Stream-local
    "gain" is applied around the midpoint 0.5 rather than by raw multiplication, so the
    perturbation changes contrast within the bounded [0,1] input range instead of creating
    a trivial global rescaling.

    This is intended as a cleaner perturbation test for dendritic normalization and
    pathway-specific credit assignment than uniform global gain shifts.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if n_classes != 2:
        raise ValueError(
            "contextual_stream_gain_shift currently supports binary classification only."
        )

    midpoint = 0.5
    total_samples = int(n_samples)

    def _sample_gains(
        n: int,
        low: float,
        high: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return low + (high - low) * torch.rand(n, generator=generator)

    def _apply_midpoint_gain(x: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            midpoint + gains.unsqueeze(1) * (x - midpoint),
            0.0,
            1.0,
        )

    def _unit_vector(vec: torch.Tensor) -> torch.Tensor:
        return vec / torch.clamp(torch.linalg.norm(vec), min=1e-8)

    def _make_class_means(
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = torch.sign(torch.randn(stream_dim, generator=generator))
        direction = torch.where(direction == 0, torch.ones_like(direction), direction)
        direction = _unit_vector(direction)
        mean0 = torch.clamp(midpoint - 0.5 * signal_strength * direction, 0.0, 1.0)
        mean1 = torch.clamp(midpoint + 0.5 * signal_strength * direction, 0.0, 1.0)
        return mean0, mean1, direction

    def _orthogonal_direction(
        reference: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        candidate = torch.randn(stream_dim, generator=generator)
        candidate = candidate - torch.dot(candidate, reference) * reference
        if float(torch.linalg.norm(candidate)) < 1e-8:
            candidate = torch.zeros_like(reference)
            candidate[0] = 1.0
            candidate = candidate - torch.dot(candidate, reference) * reference
        return _unit_vector(candidate)

    def _apply_rotated_test_shift(
        x: torch.Tensor,
        gains: torch.Tensor,
        signal_dir: torch.Tensor,
        orth_dir: torch.Tensor,
        alignment_alpha: float,
    ) -> torch.Tensor:
        aligned_target = _apply_midpoint_gain(x, gains)
        aligned_delta = aligned_target - x

        alpha = float(np.clip(alignment_alpha, 0.0, 1.0))
        if alpha >= 1.0 - 1e-8:
            return aligned_target

        delta_norm = torch.linalg.norm(aligned_delta, dim=1, keepdim=True)
        signal_proj = (aligned_delta * signal_dir.unsqueeze(0)).sum(dim=1, keepdim=True)
        sign = torch.where(signal_proj >= 0.0, 1.0, -1.0)
        orth_delta = sign * orth_dir.unsqueeze(0) * delta_norm
        mixed_delta = (
            alpha * aligned_delta + np.sqrt(max(1.0 - alpha**2, 0.0)) * orth_delta
        )
        return torch.clamp(x + mixed_delta, 0.0, 1.0)

    def _draw_stream(
        labels: torch.Tensor,
        mean0: torch.Tensor,
        mean1: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        means = torch.where(
            labels.unsqueeze(1) == 0, mean0.unsqueeze(0), mean1.unsqueeze(0)
        )
        samples = means + 0.05 * torch.randn(
            labels.shape[0], stream_dim, generator=generator
        )
        return torch.clamp(samples, 0.0, 1.0)

    base_seed = int(seed or 0)
    mean_generator = torch.Generator().manual_seed(base_seed)
    mean_a0, mean_a1, signal_dir_a = _make_class_means(mean_generator)
    mean_b0, mean_b1, signal_dir_b = _make_class_means(mean_generator)
    orth_dir_a = _orthogonal_direction(signal_dir_a, mean_generator)
    orth_dir_b = _orthogonal_direction(signal_dir_b, mean_generator)

    def _generate_split(
        split_seed: int,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(split_seed)

        y_a = torch.randint(0, n_classes, (total_samples,), generator=generator)
        y_b = torch.randint(0, n_classes, (total_samples,), generator=generator)
        context = torch.randint(0, 2, (total_samples,), generator=generator)
        targets = torch.where(context == 0, y_a, y_b)

        x_a = _draw_stream(y_a, mean_a0, mean_a1, generator)
        x_b = _draw_stream(y_b, mean_b0, mean_b1, generator)

        relevant_a = context == 0
        relevant_b = ~relevant_a

        if split == "train":
            gains_rel = _sample_gains(
                total_samples,
                train_gain_relevant_min,
                train_gain_relevant_max,
                generator,
            )
            gains_irr = _sample_gains(
                total_samples,
                train_gain_irrelevant_min,
                train_gain_irrelevant_max,
                generator,
            )
            gain_a = torch.where(relevant_a, gains_rel, gains_irr)
            gain_b = torch.where(relevant_b, gains_rel, gains_irr)
        else:
            gain_a = torch.ones(total_samples)
            gain_b = torch.ones(total_samples)
            if ood_mode == "relevant":
                gain_a = torch.where(
                    relevant_a,
                    torch.full_like(gain_a, test_gain_relevant),
                    gain_a,
                )
                gain_b = torch.where(
                    relevant_b,
                    torch.full_like(gain_b, test_gain_relevant),
                    gain_b,
                )
            elif ood_mode == "irrelevant":
                gain_a = torch.where(
                    relevant_a,
                    gain_a,
                    torch.full_like(gain_a, test_gain_irrelevant),
                )
                gain_b = torch.where(
                    relevant_b,
                    gain_b,
                    torch.full_like(gain_b, test_gain_irrelevant),
                )
            elif ood_mode == "both":
                gain_a = torch.where(
                    relevant_a,
                    torch.full_like(gain_a, test_gain_relevant),
                    torch.full_like(gain_a, test_gain_irrelevant),
                )
                gain_b = torch.where(
                    relevant_b,
                    torch.full_like(gain_b, test_gain_relevant),
                    torch.full_like(gain_b, test_gain_irrelevant),
                )
            else:
                raise ValueError(f"Unknown ood_mode: {ood_mode}")

        if split == "train":
            x_a = _apply_midpoint_gain(x_a, gain_a)
            x_b = _apply_midpoint_gain(x_b, gain_b)
        else:
            x_a_shifted = _apply_midpoint_gain(x_a, gain_a)
            x_b_shifted = _apply_midpoint_gain(x_b, gain_b)
            if (
                ood_mode in {"irrelevant", "both"}
                and test_irrelevant_alignment_alpha < 1.0
            ):
                if (~relevant_a).any():
                    irr_mask_a = ~relevant_a
                    x_a_shifted[irr_mask_a] = _apply_rotated_test_shift(
                        x_a[irr_mask_a],
                        gain_a[irr_mask_a],
                        signal_dir_a,
                        orth_dir_a,
                        test_irrelevant_alignment_alpha,
                    )
                if (~relevant_b).any():
                    irr_mask_b = ~relevant_b
                    x_b_shifted[irr_mask_b] = _apply_rotated_test_shift(
                        x_b[irr_mask_b],
                        gain_b[irr_mask_b],
                        signal_dir_b,
                        orth_dir_b,
                        test_irrelevant_alignment_alpha,
                    )
            x_a = x_a_shifted
            x_b = x_b_shifted

        noise_a_std = torch.where(
            relevant_a,
            torch.full((total_samples,), relevant_noise_std),
            torch.full((total_samples,), irrelevant_noise_std),
        )
        noise_b_std = torch.where(
            relevant_b,
            torch.full((total_samples,), relevant_noise_std),
            torch.full((total_samples,), irrelevant_noise_std),
        )

        x_a = torch.clamp(
            x_a
            + torch.randn(total_samples, stream_dim, generator=generator)
            * noise_a_std.unsqueeze(1),
            0.0,
            1.0,
        )
        x_b = torch.clamp(
            x_b
            + torch.randn(total_samples, stream_dim, generator=generator)
            * noise_b_std.unsqueeze(1),
            0.0,
            1.0,
        )

        ctx = torch.zeros(total_samples, 2)
        ctx[torch.arange(total_samples), context] = float(context_signal_scale)

        data = torch.cat([x_a, x_b, ctx], dim=1)
        data, targets = _balance_classes(data, targets)

        perm = torch.randperm(len(data), generator=generator)
        return data[perm], targets[perm]

    if valid_split_mode not in {"ood", "train"}:
        raise ValueError(
            "valid_split_mode must be either 'ood' or 'train' for "
            "contextual_stream_gain_shift."
        )

    train_data, train_labels = _generate_split(base_seed, "train")
    valid_data, valid_labels = _generate_split(
        base_seed + 1,
        "train" if valid_split_mode == "train" else "test",
    )
    test_data, test_labels = _generate_split(base_seed + 2, "test")

    return (
        TensorDataset(train_data, train_labels),
        TensorDataset(valid_data, valid_labels),
        TensorDataset(test_data, test_labels),
    )


def _soft_bits(x: torch.Tensor) -> torch.Tensor:
    """Match nonlinear_separable xor bitization."""
    return (x > 0.5).long()


def _leaf_values(
    leaf_latents: torch.Tensor,
    leaf_op: str,
    threshold: float,
) -> torch.Tensor:
    """Reduce each leaf latent group to a binary value ``[N, n_leaves]``.

    ``leaf_latents`` has shape ``[N, n_leaves, latent_dims]`` and contains the
    low-dimensional factors that define each leaf bit (before population embedding).
    """
    if leaf_op == "xor":
        bits = _soft_bits(leaf_latents)
        out = bits[..., 0]
        for d in range(1, bits.shape[-1]):
            out = out ^ bits[..., d]
        return out
    if leaf_op == "and":
        # Reuse nonlinear_interaction idea: product threshold (soft conjunction).
        prod = leaf_latents.prod(dim=-1)
        return (prod > threshold).long()
    raise ValueError(f"Unknown leaf_op: {leaf_op}")


def _combine_binary(vals: torch.Tensor, op: str) -> torch.Tensor:
    """Combine binary values along the last dimension -> ``[N, ...]``."""
    out = vals[..., 0]
    for i in range(1, vals.shape[-1]):
        if op == "xor":
            out = out ^ vals[..., i]
        elif op == "and":
            out = out & vals[..., i]
        elif op == "or":
            out = out | vals[..., i]
        else:
            raise ValueError(f"Unknown op: {op}")
    return out


def _nested_reduce_by_branch_factors(
    leaf_bits: torch.Tensor,
    branch_factors: tuple[int, ...],
    internal_op: str,
) -> torch.Tensor:
    """
    Reduce ``[N, n_leaves]`` with an explicit branching schedule (children only).

    ``branch_factors`` follow the EINet convention (proximal→distal). Reduction
    walks distal→proximal by applying ``reversed(branch_factors)``.
    Example: ``[2, 4]`` means 8 leaves grouped first into two distal groups of 4,
    then those two results are combined at the proximal node.
    """
    values = leaf_bits
    expected_leaves = _n_leaves_from_branch_factors(branch_factors)
    if values.shape[-1] != expected_leaves:
        raise ValueError(
            f"leaf_bits trailing dim {values.shape[-1]} does not match "
            f"prod(branch_factors)={expected_leaves} for {branch_factors}"
        )

    for b in reversed(branch_factors):
        n = values.shape[-1]
        if n % b != 0:
            raise ValueError(
                f"Cannot group {n} values with branch factor {b} "
                f"(branch_factors={branch_factors})"
            )
        values = values.view(values.shape[0], n // b, b)
        values = _combine_binary(values, internal_op)
    return values.squeeze(-1)


def _nested_reduce_with_internal_aux(
    leaf_bits: torch.Tensor,
    aux_bits: torch.Tensor,
    branch_factors: tuple[int, ...],
    internal_op: str,
    auxiliary_root: bool,
    include_internal_latents: bool = False,
) -> torch.Tensor:
    """
    Reduce leaf bits with optional per-internal-node aux bits (distal→proximal).

    When ``include_internal_latents`` is False, every level is children-only
    (classic tree-over-leaves) and ``auxiliary_root`` / ``aux_bits`` are ignored
    aside from requiring an empty aux trailing dim.

    When True, each non-root internal node combines children with that node's
    aux bit via ``internal_op``. The root includes an aux bit only when
    ``auxiliary_root`` is True (children-only otherwise).

    ``aux_bits`` are ordered distal→proximal (root last when present), matching
    the reduction walk over ``reversed(branch_factors)``.
    """
    values = leaf_bits
    expected_leaves = _n_leaves_from_branch_factors(branch_factors)
    use_root_aux = bool(include_internal_latents) and bool(auxiliary_root)
    expected_aux = _n_internal_nodes_from_branch_factors(
        branch_factors,
        auxiliary_root=use_root_aux,
        include_internal_latents=include_internal_latents,
    )
    if values.shape[-1] != expected_leaves:
        raise ValueError(
            f"leaf_bits trailing dim {values.shape[-1]} does not match "
            f"prod(branch_factors)={expected_leaves} for {branch_factors}"
        )
    if aux_bits.ndim != 2 or aux_bits.shape[0] != values.shape[0]:
        raise ValueError(
            f"aux_bits must have shape [N, n_internal]=[{values.shape[0]}, "
            f"{expected_aux}], got {tuple(aux_bits.shape)}"
        )
    if aux_bits.shape[-1] != expected_aux:
        raise ValueError(
            f"aux_bits trailing dim {aux_bits.shape[-1]} does not match "
            f"n_internal={expected_aux} for branch_factors={branch_factors}, "
            f"include_internal_latents={include_internal_latents}, "
            f"auxiliary_root={auxiliary_root}"
        )

    aux_offset = 0
    factors = tuple(reversed(branch_factors))
    for step_idx, b in enumerate(factors):
        n = values.shape[-1]
        if n % b != 0:
            raise ValueError(
                f"Cannot group {n} values with branch factor {b} "
                f"(branch_factors={branch_factors})"
            )
        n_groups = n // b
        values = values.view(values.shape[0], n_groups, b)
        is_root_step = step_idx == len(factors) - 1
        children_only = (not include_internal_latents) or (
            is_root_step and not use_root_aux
        )
        if children_only:
            values = _combine_binary(values, internal_op)
            continue
        level_aux = aux_bits[:, aux_offset : aux_offset + n_groups]
        aux_offset += n_groups
        values = _combine_binary(
            torch.cat([values, level_aux.unsqueeze(-1)], dim=-1),
            internal_op,
        )

    if aux_offset != expected_aux:
        raise ValueError(
            f"Consumed {aux_offset} aux bits but expected {expected_aux} "
            f"(branch_factors={branch_factors}, "
            f"include_internal_latents={include_internal_latents}, "
            f"auxiliary_root={auxiliary_root})"
        )
    return values.squeeze(-1)


def _make_population_projection(
    latent_dims: int,
    dims_per_leaf: int,
) -> torch.Tensor:
    """Build a fixed nonnegative projection ``[latent_dims, dims_per_leaf]``.

    Each observed block coordinate is a convex combination of the node latents, so
    the latent is distributed across the block (population code) rather than sitting
    in a few privileged coordinates.
    """
    weights = torch.rand(latent_dims, dims_per_leaf)
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return weights


def _population_embed_leaf_blocks(
    latents: torch.Tensor,
    projection: torch.Tensor,
    distractor_scale: float,
) -> torch.Tensor:
    """Embed ``[N, n_nodes, latent_dims]`` into ``[N, n_nodes, dims_per_leaf]``.

    ``projection`` has shape ``[latent_dims, dims_per_leaf]``. Distractors are
    mixed in with weight ``distractor_scale`` in ``[0, 1]``.
    """
    if not 0.0 <= float(distractor_scale) <= 1.0:
        raise ValueError(
            f"distractor_scale must be in [0, 1], got {distractor_scale}"
        )
    signal = torch.einsum("nld,db->nlb", latents, projection)
    scale = float(distractor_scale)
    if scale <= 0.0:
        return signal.clamp(0.0, 1.0)
    distractors = torch.rand_like(signal)
    blocks = (1.0 - scale) * signal + scale * distractors
    return blocks.clamp(0.0, 1.0)


def create_nested_multixor_dataset(
    n_samples: int = 4000,
    branch_factors: list[int] | tuple[int, ...] = (2, 2),
    leaf_op: str = "and",
    internal_op: str = "xor",
    threshold: float = 0.25,
    shuffle_leaves: bool = False,
    dims_per_leaf: int = 64,
    latent_dims: int = 2,
    distractor_scale: float = 0.25,
    include_internal_latents: bool = False,
    auxiliary_root: bool = False,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Nested MultiXOR / tree-composition task with population-coded node blocks.

    ``branch_factors`` uses the EINet proximal→distal convention and may be any
    valid schedule (not only ``[2, 2]``). Leaf count is ``prod(branch_factors)``.
    Each leaf has a low-D latent in ``[0, 1]^{latent_dims}`` that determines its
    bit via ``leaf_op``.

    When ``include_internal_latents=False`` (default), only leaf blocks appear in
    the input and labels are classic children-only tree composition
    (``auxiliary_root`` is ignored). When ``True``, each non-root internal node
    also gets an aux latent embedded as a dedicated block and included in the
    cascading ``internal_op``. Root/soma aux requires both
    ``include_internal_latents=True`` and ``auxiliary_root=True`` (pair with
    model ``somatic_synapses=True``).

    Latents are **population-embedded** into contiguous blocks of size
    ``dims_per_leaf`` (shared nonnegative random projection + optional
    within-block distractor mix). Layout is leaf blocks in leaf order, then
    (when enabled) internal aux blocks in distal→proximal reduction order
    (root aux last if present), so
    ``input_dim = (n_leaves + n_internal) * dims_per_leaf``.

    Default ``leaf_op='and'`` + ``internal_op='xor'`` does **not** collapse to flat
    parity of the raw input dimensions. ``leaf_op='xor'`` + ``internal_op='xor'``
    does (parity over leaf bits and included aux bits).

    Set ``shuffle_leaves=True`` to jointly permute all node latents used for
    labels (morphology control): input block ``i`` still embeds latent ``i``,
    but labels use a permuted node order. Match ``branch_factors`` to the
    model's ``excitatory_branch_factors`` when testing morphology-aligned
    composition.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    branch_factors = _normalize_branch_factors(branch_factors)
    dims_per_leaf = int(dims_per_leaf)
    latent_dims = int(latent_dims)
    include_internal_latents = bool(include_internal_latents)
    auxiliary_root = bool(auxiliary_root)
    use_root_aux = include_internal_latents and auxiliary_root
    if latent_dims < 1:
        raise ValueError(f"latent_dims must be >= 1, got {latent_dims}")
    if dims_per_leaf < latent_dims:
        raise ValueError(
            f"dims_per_leaf ({dims_per_leaf}) must be >= latent_dims ({latent_dims})"
        )

    n_leaves = _n_leaves_from_branch_factors(branch_factors)
    n_internal = _n_internal_nodes_from_branch_factors(
        branch_factors,
        auxiliary_root=use_root_aux,
        include_internal_latents=include_internal_latents,
    )
    n_blocks = n_leaves + n_internal
    input_dim = n_blocks * dims_per_leaf
    total_samples = int(n_samples) * 2

    leaf_latents = torch.rand(total_samples, n_leaves, latent_dims)
    if n_internal > 0:
        internal_latents = torch.rand(total_samples, n_internal, latent_dims)
    else:
        internal_latents = torch.empty(total_samples, 0, latent_dims)
    node_latents = torch.cat([leaf_latents, internal_latents], dim=1)

    label_nodes = node_latents
    if shuffle_leaves:
        # Jointly permute all node→tree assignments for labels only; keep
        # block i = embed(latent i) so contiguous input blocks no longer match
        # the composition tree.
        perm = torch.randperm(n_blocks)
        label_nodes = node_latents[:, perm, :]

    label_leaf_latents = label_nodes[:, :n_leaves, :]
    label_internal_latents = label_nodes[:, n_leaves:, :]
    leaf_bits = _leaf_values(
        label_leaf_latents, leaf_op=leaf_op, threshold=threshold
    )
    if n_internal > 0:
        aux_bits = _leaf_values(
            label_internal_latents, leaf_op=leaf_op, threshold=threshold
        )
    else:
        aux_bits = torch.zeros(total_samples, 0, dtype=torch.long)
    labels = _nested_reduce_with_internal_aux(
        leaf_bits,
        aux_bits,
        branch_factors,
        internal_op=internal_op,
        auxiliary_root=use_root_aux,
        include_internal_latents=include_internal_latents,
    ).long()

    projection = _make_population_projection(latent_dims, dims_per_leaf)
    node_blocks = _population_embed_leaf_blocks(
        node_latents, projection, distractor_scale=distractor_scale
    )
    data = node_blocks.reshape(total_samples, input_dim)

    data, labels = _balance_classes(data, labels)
    perm = torch.randperm(len(data))
    return _split_dataset(data[perm], labels[perm])


def create_mnist_information_sampled_dataset(
    subset_size: int = 100,
    sampling_strategy: str = "info_balanced",
    n_classes: int = 10,
    precomputed_fisher_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict]:
    """
    Create MNIST dataset with information-based pixel sampling.

    Computes per-pixel Fisher information and samples pixels according to strategy.

    Args:
        subset_size: Number of pixels to sample (out of 784)
        sampling_strategy: 'info_balanced', 'top_k', or 'random'
        n_classes: MNIST classes to use (default 10)
        precomputed_fisher_path: Path to precomputed Fisher scores
        seed: Random seed

    Returns:
        train, valid, test datasets, info_dict
    """
    from torchvision import datasets, transforms

    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_mnist = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Convert to tensors
    train_data = train_mnist.data.float().reshape(-1, 784) / 255.0
    train_labels = train_mnist.targets
    test_data = test_mnist.data.float().reshape(-1, 784) / 255.0
    test_labels = test_mnist.targets

    # Filter to n_classes
    if n_classes < 10:
        train_mask = train_labels < n_classes
        train_data = train_data[train_mask]
        train_labels = train_labels[train_mask]
        test_mask = test_labels < n_classes
        test_data = test_data[test_mask]
        test_labels = test_labels[test_mask]

    # Compute per-pixel Fisher information
    fisher_scores = torch.zeros(784)
    for pixel_idx in range(784):
        pixel_values = train_data[:, pixel_idx]

        # Compute between-class variance
        class_means = torch.zeros(n_classes)
        class_vars = torch.zeros(n_classes)
        for c in range(n_classes):
            class_mask = train_labels == c
            class_pixels = pixel_values[class_mask]
            class_means[c] = class_pixels.mean()
            class_vars[c] = class_pixels.var()

        # Fisher information: between-class variance / within-class variance
        between_var = class_means.var()
        within_var = class_vars.mean()
        fisher_scores[pixel_idx] = between_var / (within_var + 1e-8)

    # Sample pixels based on strategy
    if sampling_strategy == "top_k":
        # Sample top-k most informative pixels for both E and I
        _, top_indices = torch.topk(fisher_scores, subset_size)
        e_indices = top_indices[: subset_size // 2]
        i_indices = top_indices[: subset_size // 2]

    elif sampling_strategy == "info_balanced":
        # Sample to balance information between E and I
        sorted_indices = torch.argsort(fisher_scores, descending=True)
        e_indices = []
        i_indices = []
        I_E = 0.0
        I_I = 0.0

        for idx in sorted_indices:
            if len(e_indices) < subset_size // 2 and I_E <= I_I:
                e_indices.append(idx.item())
                I_E += fisher_scores[idx].item()
            elif len(i_indices) < subset_size // 2:
                i_indices.append(idx.item())
                I_I += fisher_scores[idx].item()

            if (
                len(e_indices) >= subset_size // 2
                and len(i_indices) >= subset_size // 2
            ):
                break

        e_indices = torch.tensor(e_indices)
        i_indices = torch.tensor(i_indices)

    elif sampling_strategy == "random":
        # Random sampling
        perm = torch.randperm(784)
        e_indices = perm[: subset_size // 2]
        i_indices = perm[subset_size // 2 : subset_size]

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    # Create sampled datasets
    all_indices = torch.cat([e_indices, i_indices])
    train_data_sampled = train_data[:, all_indices]
    test_data_sampled = test_data[:, all_indices]

    # Split train into train/valid
    n_train = int(0.8 * len(train_data_sampled))
    valid_data_sampled = train_data_sampled[n_train:]
    valid_labels_sampled = train_labels[n_train:]
    train_data_sampled = train_data_sampled[:n_train]
    train_labels = train_labels[:n_train]

    # Create info dict
    info_dict = {
        "e_indices": e_indices.tolist(),
        "i_indices": i_indices.tolist(),
        "fisher_scores": fisher_scores.tolist(),
        "I_E": fisher_scores[e_indices].sum().item(),
        "I_I": fisher_scores[i_indices].sum().item(),
        "sampling_strategy": sampling_strategy,
    }

    return (
        TensorDataset(train_data_sampled, train_labels),
        TensorDataset(valid_data_sampled, valid_labels_sampled),
        TensorDataset(test_data_sampled, test_labels),
        info_dict,
    )


def _split_dataset(
    data: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float = 0.6,
    valid_frac: float = 0.2,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Split data into train/valid/test datasets."""
    n_total = len(data)
    n_train = int(train_frac * n_total)
    n_valid = int(valid_frac * n_total)

    train_data = data[:n_train]
    train_labels = labels[:n_train]
    valid_data = data[n_train : n_train + n_valid]
    valid_labels = labels[n_train : n_train + n_valid]
    test_data = data[n_train + n_valid :]
    test_labels = labels[n_train + n_valid :]

    return (
        TensorDataset(train_data, train_labels),
        TensorDataset(valid_data, valid_labels),
        TensorDataset(test_data, test_labels),
    )


_SYNTHETIC_DATASET_SPECS: dict[str, _SyntheticDatasetSpec] = {
    "correlated_gaussian": _SyntheticDatasetSpec(
        factory=create_correlated_gaussian_dataset,
        defaults={
            "n_samples": 1000,
            "input_dim": 100,
            "n_classes": 2,
            "rho_ee": 0.0,
            "rho_ii": 0.0,
            "rho_ei": 0.0,
            "ne": 50,
            "ni": 50,
            "class_separation": 0.3,
            "noise_std": 0.2,
            "seed": None,
        },
    ),
    "nonlinear_interaction": _SyntheticDatasetSpec(
        factory=create_nonlinear_interaction_dataset,
        defaults={
            "n_samples": 1000,
            "input_dim": 2,
            "n_classes": 2,
            "interaction_strength": 1.0,
            "noise_std": 0.1,
            "seed": None,
        },
    ),
    "linear_separable": _SyntheticDatasetSpec(
        factory=create_linear_separable_dataset,
        defaults={
            "n_samples": 1000,
            "input_dim": 100,
            "ne": 50,
            "ni": 50,
            "rho_ee": 0.0,
            "rho_ii": 0.0,
            "rho_ei": 0.0,
            "signal_strength": 0.3,
            "noise_std": 0.2,
            "seed": None,
        },
    ),
    "nonlinear_separable": _SyntheticDatasetSpec(
        factory=create_nonlinear_separable_dataset,
        defaults={
            "n_samples": 1000,
            "input_dim": 100,
            "ne": 50,
            "ni": 50,
            "nonlinearity_type": "quadratic",
            "rho_ee": 0.0,
            "rho_ii": 0.0,
            "rho_ei": 0.0,
            "noise_std": 0.2,
            "seed": None,
        },
    ),
    "block_correlated": _SyntheticDatasetSpec(
        factory=create_block_correlated_dataset,
        defaults={
            "n_samples": 1000,
            "input_dim": 100,
            "ne": 50,
            "ni": 50,
            "n_blocks": 5,
            "block_rho_ee": [0.0, 0.2, -0.1],
            "block_rho_ii": [0.0, 0.2, -0.1],
            "block_rho_ei": [0.0, 0.3, 0.5],
            "class_separation": 0.3,
            "noise_std": 0.2,
            "seed": None,
        },
    ),
    "balanced_information": _SyntheticDatasetSpec(
        factory=create_balanced_information_dataset,
        defaults={
            "n_samples": 1000,
            "input_dim": 100,
            "ne": 50,
            "ni": 50,
            "info_per_dim": "balanced",
            "rho_ee": 0.0,
            "rho_ii": 0.0,
            "rho_ei": 0.0,
            "noise_std": 0.2,
            "seed": None,
        },
    ),
    "heterogeneous_information": _SyntheticDatasetSpec(
        factory=create_heterogeneous_information_dataset,
        defaults={
            "n_samples": 2000,
            "input_dim": 100,
            "n_classes": 10,
            "ne": 50,
            "ni": 50,
            "n_high_info_dims": 10,
            "high_info_strength": 2.0,
            "low_info_strength": 0.1,
            "noise_std": 0.2,
            "rho_ee": 0.0,
            "rho_ii": 0.0,
            "rho_ei": 0.0,
            "seed": None,
        },
    ),
    "nonlinear_transformation": _SyntheticDatasetSpec(
        factory=create_nonlinear_transformation_dataset,
        defaults={
            "n_samples": 2000,
            "n_classes": 2,
            "nonlinearity_type": "xor",
            "ne": 2,
            "ni": 2,
            "noise_std": 0.1,
            "seed": None,
        },
    ),
    "asymmetric_noise": _SyntheticDatasetSpec(
        factory=create_asymmetric_noise_dataset,
        defaults={
            "n_samples": 2000,
            "input_dim": 100,
            "n_classes": 10,
            "ne": 50,
            "ni": 50,
            "sigma_e": 0.2,
            "sigma_i": 0.1,
            "class_separation": 0.2,
            "rho_ee": 0.0,
            "rho_ii": 0.0,
            "rho_ei": 0.0,
            "seed": None,
        },
    ),
    "population_code": _SyntheticDatasetSpec(
        factory=create_population_code_dataset,
        defaults={
            "n_samples": 2000,
            "pop_dim": 200,
            "ne": 50,
            "ni": 50,
            "n_classes": 2,
            "code_type": "mixed_selectivity",
            "noise_std": 0.1,
            "kappa": 2.0,
            "seed": None,
        },
    ),
    "contextual_stream_gain_shift": _SyntheticDatasetSpec(
        factory=create_contextual_stream_gain_shift_dataset,
        defaults={
            "n_samples": 4000,
            "stream_dim": 64,
            "n_classes": 2,
            "signal_strength": 0.35,
            "relevant_noise_std": 0.08,
            "irrelevant_noise_std": 0.18,
            "train_gain_relevant_min": 0.9,
            "train_gain_relevant_max": 1.1,
            "train_gain_irrelevant_min": 0.8,
            "train_gain_irrelevant_max": 1.2,
            "test_gain_relevant": 1.0,
            "test_gain_irrelevant": 3.0,
            "test_irrelevant_alignment_alpha": 1.0,
            "ood_mode": "irrelevant",
            "valid_split_mode": "ood",
            "context_signal_scale": 1.0,
            "seed": None,
        },
    ),
    "nested_multixor": _SyntheticDatasetSpec(
        factory=create_nested_multixor_dataset,
        defaults={
            "n_samples": 4000,
            "branch_factors": [2, 2],
            "leaf_op": "and",
            "internal_op": "xor",
            "threshold": 0.25,
            "shuffle_leaves": False,
            "dims_per_leaf": 64,
            "latent_dims": 2,
            "distractor_scale": 0.25,
            "include_internal_latents": False,
            "auxiliary_root": False,
            "seed": None,
        },
    ),
}


def get_synthetic_datasets(
    dataset_config: dict[str, Any],
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Factory function to create synthetic datasets based on config.

    Args:
        dataset_config: Configuration dictionary containing dataset parameters

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    dataset_type = dataset_config.get("type", "correlated_gaussian")
    spec = _SYNTHETIC_DATASET_SPECS.get(dataset_type)
    if spec is None:
        raise ValueError(f"Unknown synthetic dataset type: {dataset_type}")
    return spec.build(dataset_config)
