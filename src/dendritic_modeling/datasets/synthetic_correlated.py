"""
Synthetic dataset with correlated Gaussian inputs for studying synaptic correlation learning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, , Any
import logging

logger = logging.getLogger(__name__)

class SyntheticCorrelatedDataset(Dataset):
    """
    Synthetic dataset with two classes generated from multivariate Gaussians.

    Features:
    - Dimension D inputs
    - First D/3 dimensions: class 1 mean = -0.5, class 2 mean = 0.5
    - Second D/3 dimensions: both classes mean = 0
    - Last D/3 dimensions: class 1 mean = 0.5, class 2 mean = -0.5
    - Correlation structure with off-diagonal elements in [-0.1, 0.1]
    - Sigmoid transform applied to final inputs
    """

    def __init__(
        self,
        n_samples: int,
        input_dim: int,
        correlation_range: tuple[float, float] = (-0.1, 0.1),
        random_seed: Optional[int] = None,
        apply_sigmoid: bool = True,
        mean_magnitude: float = 0.5
    ):
        """
        Initialize synthetic correlated dataset.

        Args:
            n_samples: Number of samples per class
            input_dim: Dimension D of input features
            correlation_range: Range for off-diagonal correlations
            random_seed: Random seed for reproducibility
            apply_sigmoid: Whether to apply sigmoid transform to inputs
            mean_magnitude: Magnitude of mean separation between classes (default 0.5)
        """
        super().__init__()

        if input_dim % 3 != 0:
            raise ValueError(f"input_dim must be divisible by 3, got {input_dim}")

        self.n_samples = n_samples
        self.input_dim = input_dim
        self.correlation_range = correlation_range
        self.apply_sigmoid = apply_sigmoid
        self.mean_magnitude = mean_magnitude

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Generate data
        self.data, self.labels = self._generate_data()

    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate correlation matrix with specified structure."""
        # Start with identity matrix (variance = 1)
        corr_matrix = np.eye(self.input_dim)

        # Add random correlations in specified range
        for i in range(self.input_dim):
            for j in range(i + 1, self.input_dim):
                corr = np.random.uniform(*self.correlation_range)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # Ensure positive semi-definite by eigenvalue correction
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        if np.min(eigenvalues) < 0:
            # Shift eigenvalues to make matrix positive semi-definite
            min_eig = np.min(eigenvalues)
            corr_matrix = corr_matrix - min_eig * np.eye(self.input_dim) + 1e-6 * np.eye(self.input_dim)

        return corr_matrix

    def _generate_means(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate mean vectors for two classes."""
        d_third = self.input_dim // 3

        # Class 1 means
        mean1 = np.zeros(self.input_dim)
        mean1[:d_third] = -self.mean_magnitude  # First third
        mean1[d_third:2*d_third] = 0.0  # Second third
        mean1[2*d_third:] = self.mean_magnitude  # Last third

        # Class 2 means
        mean2 = np.zeros(self.input_dim)
        mean2[:d_third] = self.mean_magnitude  # First third
        mean2[d_third:2*d_third] = 0.0  # Second third
        mean2[2*d_third:] = -self.mean_magnitude  # Last third

        return mean1, mean2

    def _generate_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the full dataset."""
        # Generate correlation matrix
        cov_matrix = self._generate_correlation_matrix()

        # Generate means
        mean1, mean2 = self._generate_means()

        # Generate samples for each class
        samples_class1 = np.random.multivariate_normal(mean1, cov_matrix, self.n_samples)
        samples_class2 = np.random.multivariate_normal(mean2, cov_matrix, self.n_samples)

        # Combine samples
        all_samples = np.vstack([samples_class1, samples_class2])

        # Apply sigmoid if requested
        if self.apply_sigmoid:
            all_samples = 1 / (1 + np.exp(-all_samples))

        # Create labels
        labels = np.array([0] * self.n_samples + [1] * self.n_samples)

        # Convert to torch tensors
        data = torch.FloatTensor(all_samples)
        labels = torch.LongTensor(labels)

        # Shuffle
        perm = torch.randperm(len(data))
        data = data[perm]
        labels = labels[perm]

        logger.info(f"Generated synthetic dataset: {len(data)} samples, {self.input_dim} dimensions")
        logger.info(f"Correlation range: {self.correlation_range}")
        logger.info(f"Sigmoid applied: {self.apply_sigmoid}")
        logger.info(f"Mean magnitude: {self.mean_magnitude}")

        return data, labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    def get_correlation_matrix(self) -> np.ndarray:
        """Get the empirical correlation matrix of the generated data."""
        return np.corrcoef(self.data.numpy().T)

    def get_class_means(self) -> tuple[np.ndarray, np.ndarray]:
        """Get empirical means for each class."""
        class0_data = self.data[self.labels == 0].numpy()
        class1_data = self.data[self.labels == 1].numpy()
        return class0_data.mean(axis=0), class1_data.mean(axis=0)

def load_synthetic_correlated(
    train_size: int = 5000,
    valid_size: int = 1000,
    test_size: int = 1000,
    input_dim: int = 300,
    correlation_range: tuple[float, float] = (-0.1, 0.1),
    batch_size: int = 128,
    random_seed: Optional[int] = None,
    mean_magnitude: float = 0.5,
    **kwargs
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """
    Load synthetic correlated dataset with train/valid/test splits.

    Returns:
        train_loader, valid_loader, test_loader, dataset_info
    """
    # Generate datasets
    train_dataset = SyntheticCorrelatedDataset(
        n_samples=train_size // 2,  # Per class
        input_dim=input_dim,
        correlation_range=correlation_range,
        random_seed=random_seed,
        apply_sigmoid=True,
        mean_magnitude=mean_magnitude
    )

    valid_dataset = SyntheticCorrelatedDataset(
        n_samples=valid_size // 2,
        input_dim=input_dim,
        correlation_range=correlation_range,
        random_seed=random_seed + 1 if random_seed else None,
        apply_sigmoid=True,
        mean_magnitude=mean_magnitude
    )

    test_dataset = SyntheticCorrelatedDataset(
        n_samples=test_size // 2,
        input_dim=input_dim,
        correlation_range=correlation_range,
        random_seed=random_seed + 2 if random_seed else None,
        apply_sigmoid=True,
        mean_magnitude=mean_magnitude
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Dataset info
    dataset_info = {
        "input_dim": input_dim,
        "output_dim": 2,  # Binary classification
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "test_size": len(test_dataset),
        "correlation_range": correlation_range,
        "train_correlation_matrix": train_dataset.get_correlation_matrix(),
        "train_class_means": train_dataset.get_class_means()
    }

    return train_loader, valid_loader, test_loader, dataset_info
