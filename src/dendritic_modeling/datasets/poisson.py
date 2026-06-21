"""Poisson-noise dataset wrappers."""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PoissonGeneratorDataset(Dataset):
    """Dataset that applies Poisson noise to base dataset samples."""

    def __init__(
        self,
        base_inputs: torch.Tensor,
        base_labels: torch.Tensor,
        multiplicative_gain: bool = True,
        fixed_gain_factor: Optional[float] = None,
        max_gain_factor: Optional[float] = None,
        uniform_gain: bool = True,
        poisson_sampling: bool = True,
        stimulus_duration: float = 1.0,
        max_gain_tau_ratio: Optional[float] = None,
    ):
        stimulus_duration = float(stimulus_duration)

        if fixed_gain_factor is not None:
            fixed_gain_factor = float(fixed_gain_factor)
        elif max_gain_factor is not None:
            max_gain_factor = float(max_gain_factor)
        elif max_gain_tau_ratio is not None:
            max_gain_factor = float(max_gain_tau_ratio) * stimulus_duration

        if (
            multiplicative_gain
            and fixed_gain_factor is None
            and max_gain_factor is None
        ):
            if uniform_gain:
                max_gain_factor = 1.0
            else:
                raise ValueError(
                    "PoissonGeneratorDataset with multiplicative_gain=True and "
                    "uniform_gain=False requires fixed_gain_factor, "
                    "max_gain_factor, or max_gain_tau_ratio."
                )
        if max_gain_factor is not None and max_gain_factor < 1.0:
            raise ValueError(
                f"max_gain_factor must be >= 1.0 when provided, got {max_gain_factor}"
            )

        self.base_inputs = base_inputs
        self.base_labels = base_labels
        self.multiplicative_gain = multiplicative_gain
        self.fixed_gain_factor = fixed_gain_factor
        self.max_gain_factor = max_gain_factor
        self.uniform_gain = uniform_gain
        self.poisson_sampling = poisson_sampling
        self.stimulus_duration = stimulus_duration
        self.tau = stimulus_duration

        self._fixed_gf = None
        self._max_gf = None

    def _set_gain_factor(self, gain_factor: float | torch.Tensor):
        self._fixed_gf = gain_factor

    def _set_max_gain_factor(self, max_gain_factor: float | torch.Tensor):
        self._max_gf = max_gain_factor

    @staticmethod
    def _is_batched_index(idx) -> bool:
        if isinstance(idx, slice):
            return True
        if isinstance(idx, (list, tuple, np.ndarray)):
            return True
        if torch.is_tensor(idx):
            return idx.ndim > 0
        return False

    @staticmethod
    def _sample_uniform_gain(
        input: torch.Tensor,
        max_gain_factor: float | torch.Tensor,
        *,
        batched: bool,
    ) -> torch.Tensor:
        if not torch.is_tensor(max_gain_factor):
            max_gain_factor = torch.as_tensor(
                max_gain_factor, dtype=input.dtype, device=input.device
            )
        else:
            max_gain_factor = max_gain_factor.to(dtype=input.dtype, device=input.device)

        if batched and input.ndim > 0:
            shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        else:
            shape = ()

        random_gain = torch.rand(shape, dtype=input.dtype, device=input.device)
        return (random_gain * (max_gain_factor - 1)) + 1

    def __len__(self):
        return len(self.base_inputs)

    def __getitem__(self, idx):
        batched = self._is_batched_index(idx)
        input = self.base_inputs[idx]
        label = self.base_labels[idx]

        if self._fixed_gf is not None:
            input = input * self._fixed_gf
            self._fixed_gf = None

        elif self._max_gf is not None:
            input = input * ((torch.rand_like(input) * (self._max_gf - 1)) + 1)
            self._max_gf = None

        elif self.multiplicative_gain:
            if self.fixed_gain_factor is not None:
                gain_factor = self.fixed_gain_factor

            elif self.uniform_gain:
                gain_factor = self._sample_uniform_gain(
                    input, self.max_gain_factor, batched=batched
                )

            else:
                gain_factor = (torch.rand_like(input) * (self.max_gain_factor - 1)) + 1

            input = gain_factor * input

        if self.poisson_sampling:
            noisy_input = torch.poisson(self.tau * input)
            noisy_input = noisy_input / self.tau
            return noisy_input, label
        else:
            return input, label


__all__ = ["PoissonGeneratorDataset"]
