import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import norm

class StructuredCorrelationDataset(Dataset):
    def __init__(
        self,
        n_classes=2,
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
        self.n_classes = n_classes
        self.block_size = block_size
        self.samples_per_class = n_samples // n_classes
        self.mean = mean
        self.sigma = sigma
        self.rho_modular = rho_modular
        self.condition = condition

        # Get list of correlation matrices (one per class) and per-class permutations
        self.correlation_matrices, self.class_permutations = self.get_condition_matrices(
            condition,
            n_dims,
            block_size,
            rho_modular,
            rho_nested_local,
            rho_nested_global,
            shift,
            difference,
            permute_ratio,
            noise_ratio,
            n_classes,
        )
        # class_permutations: {class_idx: perm} for permuted classes only.
        # perm[d] = canonical dimension that input dim d maps to.
        # class_block_assignments[k][d] = block index of input dim d under class k's correlation structure.
        #   For class 0 (canonical):  assignments[d] = d // block_size
        #   For class k>0 (permuted): assignments[d] = perm_k[d] // block_size
        self.class_block_assignments: dict = self._compute_class_block_assignments()

        # Sample from each class's correlation matrix
        self.X_per_class = []
        self.Z_per_class = []
        for R in self.correlation_matrices:
            X, Z = self.sample_from_gaussian_copula(R, self.samples_per_class, mean, sigma)
            self.X_per_class.append(X)
            self.Z_per_class.append(Z)

        # Combine all samples
        X_combined = np.vstack(self.X_per_class)
        # X_normalized = X_combined  # Keeping your comment about removing robust clipping

        self.data = torch.tensor(X_combined, dtype=torch.float32)
        self.labels = torch.tensor(
            [i for i in range(n_classes) for _ in range(self.samples_per_class)],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _compute_class_block_assignments(self) -> dict:
        """
        Build {class_idx: np.ndarray(n_dims, dtype=int)} mapping each input dim to
        its block index under that class's correlation structure.

        For class 0 (canonical, no permutation):
            assignments[d] = d // block_size

        For class k > 0 (with permutation perm_k):
            assignments[d] = perm_k[d] // block_size
            because R_k[i,j] = R0[perm_k[i], perm_k[j]], so dims i and j are in the
            same correlated group iff perm_k[i] // block_size == perm_k[j] // block_size.
        """
        canonical = np.arange(self.n_dims) // self.block_size
        assignments = {0: canonical.copy()}
        for cls_idx, perm in self.class_permutations.items():
            assignments[cls_idx] = perm // self.block_size
        return assignments

    def get_block_assignments_for_class(self, class_idx: int) -> np.ndarray:
        """
        Return block assignment array for a given class.

        Parameters
        ----------
        class_idx : int
            Class whose block structure to return.

        Returns
        -------
        np.ndarray, shape (n_dims,), dtype int
            assignments[d] = block index for input dimension d under class class_idx.
            Suitable for direct use in ``block_alignment_score(attrs, assignments, target_block=class_idx)``.
        """
        if class_idx in self.class_block_assignments:
            return self.class_block_assignments[class_idx].copy()
        # Fallback: class not explicitly stored means canonical (e.g. conditions without permutation)
        return np.arange(self.n_dims) // self.block_size

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
        """Return (R, perm) where perm[d] gives the canonical dimension that input dim d maps to."""
        perm = np.arange(n_dims)
        n_permute = int(permute_ratio * n_dims)
        permuted_indices = np.random.choice(n_dims, n_permute, replace=False)
        permuted_perm = np.random.permutation(permuted_indices)
        perm[permuted_indices] = permuted_perm
        R = StructuredCorrelationDataset.make_nested_block_matrix(n_dims, block_size, rho_local, rho_global)
        return R[np.ix_(perm, perm)], perm.copy()

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
        """Return (R, perm) where perm[d] gives the canonical dimension that input dim d maps to."""
        perm = np.arange(n_dims)
        n_permute = int(permute_ratio * n_dims)
        permuted_indices = np.random.choice(n_dims, n_permute, replace=False)
        permuted_perm = np.random.permutation(permuted_indices)
        perm[permuted_indices] = permuted_perm
        R = StructuredCorrelationDataset.make_nested_block_matrix_subset(n_dims, block_size, rho_local, rho_global, noise_ratio)
        return R[np.ix_(perm, perm)], perm.copy()

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
    def get_condition_matrices(condition, n_dims, block_size, rho_modular, rho_nested_local, rho_nested_global, shift=None, difference=None, permute_ratio=None, noise_ratio=0.5, n_classes=2):
        """
        Returns
        -------
        matrices : list of np.ndarray
            One correlation matrix per class.
        perms : dict[int, np.ndarray]
            class_idx -> perm array for every class that has a non-trivial permutation.
            perm[d] = canonical dimension that input dim d maps to.
            Class 0 (canonical) is never present; identity can be inferred.
            Empty dict for conditions without permutations.
        """
        if shift is None or difference is None:
            raise ValueError("Both 'shift' and 'difference' must be specified for this condition.")

        matrices = []
        perms = {}

        if condition == "intra_block_corr":
            for i in range(n_classes):
                rho = shift + i * difference
                R = StructuredCorrelationDataset.make_block_diagonal_matrix(n_dims, block_size, rho)
                matrices.append(R)

        elif condition == "modularity_blur":
            for i in range(n_classes):
                rho = shift + i * difference
                R = StructuredCorrelationDataset.make_nested_block_matrix(n_dims, block_size, rho_nested_local, rho)
                matrices.append(R)

        elif condition == "modularity_blur_structural":
            rho = shift
            if permute_ratio is None:
                permute_ratio = 1.0
            # Class 0 is the base (unpermuted)
            R0 = StructuredCorrelationDataset.make_nested_block_matrix(n_dims, block_size, rho_nested_local, rho)
            matrices.append(R0)
            # Classes 1..n_classes-1 each get an independent permutation
            for i in range(1, n_classes):
                R, perm = StructuredCorrelationDataset.make_partially_permuted_nested_matrix(
                    n_dims, block_size, rho_nested_local, rho, permute_ratio)
                matrices.append(R)
                perms[i] = perm

        elif condition == "noisy_permute":
            rho = rho_nested_global
            if permute_ratio is None:
                permute_ratio = 1.0
            # Class 0 is the base (unpermuted subset)
            R0 = StructuredCorrelationDataset.make_nested_block_matrix_subset(
                n_dims, block_size, rho_nested_local, rho, noise_ratio)
            matrices.append(R0)
            # Classes 1..n_classes-1 each get an independent permutation
            for i in range(1, n_classes):
                R, perm = StructuredCorrelationDataset.make_partially_permuted_nested_matrix_subset(
                    n_dims, block_size, rho_nested_local, rho, permute_ratio, noise_ratio)
                matrices.append(R)
                perms[i] = perm
        else:
            raise ValueError(f"Invalid condition: {condition}")

        for i, R in enumerate(matrices):
            assert StructuredCorrelationDataset.is_valid_correlation_matrix(R), f"R{i} is not a valid correlation matrix"

        return matrices, perms