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
from typing import Any, Optional

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


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
        # Multiplicative interactions between E and I dimensions
        e_product = torch.prod(data[:, :ne], dim=1)
        i_product = torch.prod(data[:, ne:], dim=1)
        labels = (e_product * i_product > 0.5 ** (input_dim)).long()

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

    if dataset_type == "correlated_gaussian":
        return create_correlated_gaussian_dataset(
            n_samples=dataset_config.get("n_samples", 1000),
            input_dim=dataset_config.get("input_dim", 100),
            n_classes=dataset_config.get("n_classes", 2),
            rho_ee=dataset_config.get("rho_ee", 0.0),
            rho_ii=dataset_config.get("rho_ii", 0.0),
            rho_ei=dataset_config.get("rho_ei", 0.0),
            ne=dataset_config.get("ne", 50),
            ni=dataset_config.get("ni", 50),
            class_separation=dataset_config.get("class_separation", 0.3),
            noise_std=dataset_config.get("noise_std", 0.2),
            seed=dataset_config.get("seed", None),
        )

    elif dataset_type == "nonlinear_interaction":
        return create_nonlinear_interaction_dataset(
            n_samples=dataset_config.get("n_samples", 1000),
            input_dim=dataset_config.get("input_dim", 2),
            n_classes=dataset_config.get("n_classes", 2),
            interaction_strength=dataset_config.get("interaction_strength", 1.0),
            noise_std=dataset_config.get("noise_std", 0.1),
            seed=dataset_config.get("seed", None),
        )

    elif dataset_type == "linear_separable":
        return create_linear_separable_dataset(
            n_samples=dataset_config.get("n_samples", 1000),
            input_dim=dataset_config.get("input_dim", 100),
            ne=dataset_config.get("ne", 50),
            ni=dataset_config.get("ni", 50),
            rho_ee=dataset_config.get("rho_ee", 0.0),
            rho_ii=dataset_config.get("rho_ii", 0.0),
            rho_ei=dataset_config.get("rho_ei", 0.0),
            signal_strength=dataset_config.get("signal_strength", 0.3),
            noise_std=dataset_config.get("noise_std", 0.2),
            seed=dataset_config.get("seed", None),
        )

    elif dataset_type == "nonlinear_separable":
        return create_nonlinear_separable_dataset(
            n_samples=dataset_config.get("n_samples", 1000),
            input_dim=dataset_config.get("input_dim", 100),
            ne=dataset_config.get("ne", 50),
            ni=dataset_config.get("ni", 50),
            nonlinearity_type=dataset_config.get("nonlinearity_type", "quadratic"),
            rho_ee=dataset_config.get("rho_ee", 0.0),
            rho_ii=dataset_config.get("rho_ii", 0.0),
            rho_ei=dataset_config.get("rho_ei", 0.0),
            noise_std=dataset_config.get("noise_std", 0.2),
            seed=dataset_config.get("seed", None),
        )

    elif dataset_type == "block_correlated":
        return create_block_correlated_dataset(
            n_samples=dataset_config.get("n_samples", 1000),
            input_dim=dataset_config.get("input_dim", 100),
            ne=dataset_config.get("ne", 50),
            ni=dataset_config.get("ni", 50),
            n_blocks=dataset_config.get("n_blocks", 5),
            block_rho_ee=dataset_config.get("block_rho_ee", [0.0, 0.2, -0.1]),
            block_rho_ii=dataset_config.get("block_rho_ii", [0.0, 0.2, -0.1]),
            block_rho_ei=dataset_config.get("block_rho_ei", [0.0, 0.3, 0.5]),
            class_separation=dataset_config.get("class_separation", 0.3),
            noise_std=dataset_config.get("noise_std", 0.2),
            seed=dataset_config.get("seed", None),
        )

    elif dataset_type == "balanced_information":
        return create_balanced_information_dataset(
            n_samples=dataset_config.get("n_samples", 1000),
            input_dim=dataset_config.get("input_dim", 100),
            ne=dataset_config.get("ne", 50),
            ni=dataset_config.get("ni", 50),
            info_per_dim=dataset_config.get("info_per_dim", "balanced"),
            rho_ee=dataset_config.get("rho_ee", 0.0),
            rho_ii=dataset_config.get("rho_ii", 0.0),
            rho_ei=dataset_config.get("rho_ei", 0.0),
            noise_std=dataset_config.get("noise_std", 0.2),
            seed=dataset_config.get("seed", None),
        )

    else:
        raise ValueError(f"Unknown synthetic dataset type: {dataset_type}")
