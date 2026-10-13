"""Merge centered batch moments without retaining observations."""

import numpy as np


class StreamingCovariance:
    """Merge batch-centered moments; O(features**2) state, no retained observations."""

    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros((n_features, n_features), dtype=np.float64)

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != len(self.mean):
            raise ValueError("batch must have shape (observations, features)")
        if not np.isfinite(batch).all():
            raise ValueError("batch must be finite")
        if len(batch) == 0:
            return
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        delta = batch_mean - self.mean
        total = self.n + len(batch)
        self.m2 += (
            centered.T @ centered + np.outer(delta, delta) * self.n * len(batch) / total
        )
        self.mean += delta * len(batch) / total
        self.n = total

    @property
    def covariance(self):
        if self.n < 2:
            raise ValueError("Need at least two observations")
        return self.m2 / (self.n - 1)
