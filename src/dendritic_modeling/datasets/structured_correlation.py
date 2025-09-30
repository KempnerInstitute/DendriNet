import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import norm

class StructuredCorrelationDataset(Dataset):
    def __init__(
        self,
        n_samples=10000,
        n_dims=10,
        block_size=2,
        mean=0,
        sigma=0.5,
        rho_modular=0.9,
        rho_nested_local=0.9,
        rho_nested_global=0.3,
        condition="intra_block_corr",
        shift=0.0,
        difference=0.2,
        permute_ratio=None,
        noise_ratio=0.5,
        seed=None
    ):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.n_dims = n_dims
        self.n_samples = n_samples // 2
        self.mean = mean
        self.sigma = sigma
        self.rho_modular = rho_modular
        self.condition = condition

        self.R0, self.R1 = self.get_condition_matrices(
            condition,
            n_dims,
            block_size,
            rho_modular,
            rho_nested_local,
            rho_nested_global,
            shift,
            difference,
            permute_ratio,
            noise_ratio
        )

        self.X0, self.Z0 = self.sample_from_gaussian_copula(self.R0, self.n_samples, mean, sigma)
        self.X1, self.Z1 = self.sample_from_gaussian_copula(self.R1, self.n_samples, mean, sigma)

        X_combined = np.vstack([self.X0, self.X1])
        X_normalized = X_combined # Removing robust clipping to make analytically tractable
        self.X0 = X_normalized[:self.n_samples]
        self.X1 = X_normalized[self.n_samples:]

        self.data = torch.tensor(np.vstack([self.X0, self.X1]), dtype=torch.float32)
        self.labels = torch.tensor([0] * self.n_samples + [1] * self.n_samples, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def make_block_diagonal_matrix(n_dims, block_size, rho):
        R = np.eye(n_dims)
        for i in range(0, n_dims, block_size):
            for j in range(i, min(i + block_size, n_dims)):
                for k in range(i, min(i + block_size, n_dims)):
                    if j != k:
                        R[j, k] = rho
        return R

    @staticmethod
    def make_nested_block_matrix(n_dims, block_size, rho_local, rho_global):
        R = np.full((n_dims, n_dims), rho_global)
        for i in range(0, n_dims, block_size):
            for j in range(i, min(i + block_size, n_dims)):
                for k in range(i, min(i + block_size, n_dims)):
                    if j == k:
                        R[j, k] = 1.0
                    else:
                        R[j, k] = rho_local
        return R

    @staticmethod
    def make_partially_permuted_nested_matrix(n_dims, block_size, rho_local, rho_global, permute_ratio):
        perm = np.arange(n_dims)
        n_permute = int(permute_ratio * n_dims)
        permuted_indices = np.random.choice(n_dims, n_permute, replace=False)
        permuted_perm = np.random.permutation(permuted_indices)
        perm[permuted_indices] = permuted_perm
        R = StructuredCorrelationDataset.make_nested_block_matrix(n_dims, block_size, rho_local, rho_global)
        return R[np.ix_(perm, perm)]

    @staticmethod
    def make_nested_block_matrix_subset(n_dims, block_size, rho_local, rho_global, noise_ratio=0.5):
        R = np.full((n_dims, n_dims), rho_global)
        
        # Calculate how many blocks to apply rho_local to
        n_blocks = n_dims // block_size
        n_local_blocks = int(noise_ratio * n_blocks)
        
        for i in range(0, n_dims, block_size):
            block_index = i // block_size
            for j in range(i, min(i + block_size, n_dims)):
                for k in range(i, min(i + block_size, n_dims)):
                    if j == k:
                        R[j, k] = 1.0
                    else:
                        # Apply rho_local only to the first half of blocks
                        if block_index < n_local_blocks:
                            R[j, k] = rho_local
                        # else keep rho_global (already set)
        return R

    @staticmethod
    def make_partially_permuted_nested_matrix_subset(n_dims, block_size, rho_local, rho_global, permute_ratio, noise_ratio=0.5):
        perm = np.arange(n_dims)
        n_permute = int(permute_ratio * n_dims)
        permuted_indices = np.random.choice(n_dims, n_permute, replace=False)
        permuted_perm = np.random.permutation(permuted_indices)
        perm[permuted_indices] = permuted_perm
        R = StructuredCorrelationDataset.make_nested_block_matrix_subset(n_dims, block_size, rho_local, rho_global, noise_ratio)
        return R[np.ix_(perm, perm)]

    @staticmethod
    def sample_from_gaussian_copula(R, n_samples=1000, mean=0, sigma=0.5):
        L = np.linalg.cholesky(R)
        z = np.random.randn(n_samples, R.shape[0]) @ L.T
        u = norm.cdf(z)
        # x = lognorm.ppf(u, s=sigma, scale=np.exp(mean)) # Removing lognorm to make analytically tractable
        return u, z # Return both u and z to make analytically tractable

    @staticmethod
    def robust_clip_and_normalize(x, lower_pct=0.5, upper_pct=99.5):
        low = np.percentile(x, lower_pct, axis=0)
        high = np.percentile(x, upper_pct, axis=0)
        x_clipped = np.clip(x, low, high)
        x_normalized = (x_clipped - low) / (high - low + 1e-8)
        return x_normalized

    @staticmethod
    def is_valid_correlation_matrix(R, tol=1e-8):
        if not np.allclose(R, R.T, atol=tol):
            return False
        if not np.allclose(np.diag(R), 1.0, atol=tol):
            return False
        if not ((R >= -1.0 - tol).all() and (R <= 1.0 + tol).all()):
            return False
        eigenvalues = np.linalg.eigvalsh(R)
        return np.all(eigenvalues >= -tol)

    @staticmethod
    def get_condition_matrices(condition, n_dims, block_size, rho_modular, rho_nested_local, rho_nested_global, shift=None, difference=None, permute_ratio=None, noise_ratio=0.5):
        if shift is None or difference is None:
            raise ValueError("Both 'shift' and 'difference' must be specified for this condition.")

        if condition == "intra_block_corr":
            rho0 = shift
            rho1 = shift + difference
            R0 = StructuredCorrelationDataset.make_block_diagonal_matrix(n_dims, block_size, rho0)
            R1 = StructuredCorrelationDataset.make_block_diagonal_matrix(n_dims, block_size, rho1)
        elif condition == "modularity_blur":
            rho0 = shift
            rho1 = shift + difference
            R0 = StructuredCorrelationDataset.make_nested_block_matrix(n_dims, block_size, rho_nested_local, rho0)
            R1 = StructuredCorrelationDataset.make_nested_block_matrix(n_dims, block_size, rho_nested_local, rho1)
        elif condition == "modularity_blur_structural":
            rho = shift
            if permute_ratio is None:
                permute_ratio = 1.0
            R0 = StructuredCorrelationDataset.make_nested_block_matrix(n_dims, block_size, rho_nested_local, rho)
            R1 = StructuredCorrelationDataset.make_partially_permuted_nested_matrix(n_dims, block_size, rho_nested_local, rho, permute_ratio)
        elif condition == "noisy_permute":
            rho = shift
            if permute_ratio is None:
                permute_ratio = 1.0
            R0 = StructuredCorrelationDataset.make_nested_block_matrix_subset(n_dims, block_size, rho_nested_local, rho, noise_ratio)
            R1 = StructuredCorrelationDataset.make_partially_permuted_nested_matrix_subset(n_dims, block_size, rho_nested_local, rho, permute_ratio, noise_ratio)
        else:
            raise ValueError(f"Invalid condition: {condition}")

        assert StructuredCorrelationDataset.is_valid_correlation_matrix(R0), "R0 is not a valid correlation matrix"
        assert StructuredCorrelationDataset.is_valid_correlation_matrix(R1), "R1 is not a valid correlation matrix"

        return R0, R1