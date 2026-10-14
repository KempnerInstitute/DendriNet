"""Poisson-noise dataset wrappers."""

import copy
from math import log
from types import MappingProxyType
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

#: What the ``poisson_generator`` loader uses for each key left out of
#: ``data.dataset_params.poisson_generator``. The training pipeline and the
#: tools that check a Poisson-trained checkpoint read these values, so an
#: omitted key means the same in both.
POISSON_GENERATOR_DEFAULTS = MappingProxyType(
    {
        "base_dataset": "mnist",
        "poisson_sampling": True,
        "stimulus_duration": 1.0,
        "multiplicative_gain": True,
        "fixed_gain_factor": None,
        "max_gain_factor": None,
        "max_gain_tau_ratio": None,
        "uniform_gain": True,
        "gain_sampling": "log_uniform",
    }
)


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
        gain_sampling: str = "log_uniform",
        poisson_sampling: bool = True,
        stimulus_duration: float = 1.0,
        max_gain_tau_ratio: Optional[float] = None,
    ):
        stimulus_duration = float(stimulus_duration)
        gain_sampling = str(gain_sampling).strip().lower().replace("-", "_")
        gain_sampling_aliases = {
            "log": "log_uniform",
            "loguniform": "log_uniform",
            "linear": "linear_uniform",
            "uniform": "linear_uniform",
            "linearuniform": "linear_uniform",
        }
        gain_sampling = gain_sampling_aliases.get(gain_sampling, gain_sampling)
        if gain_sampling not in {"log_uniform", "linear_uniform"}:
            raise ValueError(
                "gain_sampling must be 'log_uniform' or 'linear_uniform', "
                f"got {gain_sampling!r}"
            )

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
        self.log_max_gain_factor = (
            None if max_gain_factor is None else log(max_gain_factor)
        )
        self.uniform_gain = uniform_gain
        self.gain_sampling = gain_sampling
        self.poisson_sampling = poisson_sampling
        self.stimulus_duration = stimulus_duration
        self.tau = stimulus_duration

        # Analysis gain override applied to every item, set on copies made by
        # with_gain_override: ("fixed", g) or ("max", g).
        self._gain_override: Optional[tuple[str, float | torch.Tensor]] = None

    @property
    def gain_is_unity(self) -> bool:
        """Whether every item's multiplicative gain is exactly 1.

        True when the gain is off, fixed at 1, or drawn from ``[1, 1]``: no
        ``max_gain_factor`` or ``max_gain_tau_ratio`` (the bound defaults to
        1), or ones that give a bound of 1. Such a gain multiplies every input
        by 1 exactly. An analysis override counts in place of the training
        gain.
        """
        if self._gain_override is not None:
            _kind, gain = self._gain_override
            return bool(torch.all(torch.as_tensor(gain) == 1))
        if not self.multiplicative_gain:
            return True
        if self.fixed_gain_factor is not None:
            return float(self.fixed_gain_factor) == 1.0
        return self.max_gain_factor == 1.0

    def with_gain_override(
        self,
        *,
        fixed: Optional[float | torch.Tensor] = None,
        max_gain: Optional[float | torch.Tensor] = None,
    ) -> "PoissonGeneratorDataset":
        """Return a copy whose items use an analysis gain instead of the training one.

        ``fixed`` multiplies every input by one gain. ``max_gain`` multiplies
        each input element by its own gain drawn uniformly from
        ``[1, max_gain]``. Poisson sampling is applied afterwards as usual. The
        copy shares the base tensors, and the override holds for every item and
        every DataLoader worker, so all materialization modes see the same gain.
        """
        if (fixed is None) == (max_gain is None):
            raise ValueError("give exactly one of fixed or max_gain")
        view = copy.copy(self)
        view._gain_override = (
            ("fixed", fixed) if fixed is not None else ("max", max_gain)
        )
        return view

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
    def _sample_gain(
        input: torch.Tensor,
        max_gain_factor: float,
        *,
        gain_sampling: str,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        random_gain = torch.rand(shape, dtype=input.dtype, device=input.device)
        if gain_sampling == "log_uniform":
            return torch.exp(random_gain * log(max_gain_factor))
        if gain_sampling == "linear_uniform":
            max_gain = torch.as_tensor(
                max_gain_factor,
                dtype=input.dtype,
                device=input.device,
            )
            return (random_gain * (max_gain - 1)) + 1
        raise ValueError(f"Unknown gain_sampling={gain_sampling!r}")

    @staticmethod
    def _uniform_gain_shape(input: torch.Tensor, *, batched: bool) -> tuple[int, ...]:
        if batched and input.ndim > 0:
            return (input.shape[0],) + (1,) * (input.ndim - 1)
        return ()

    def __len__(self):
        return len(self.base_inputs)

    def __getitem__(self, idx):
        batched = self._is_batched_index(idx)
        input = self.base_inputs[idx]
        label = self.base_labels[idx]

        if self._gain_override is not None:
            kind, gain = self._gain_override
            if kind == "fixed":
                input = input * gain
            else:
                input = input * ((torch.rand_like(input) * (gain - 1)) + 1)

        elif self.multiplicative_gain:
            if self.fixed_gain_factor is not None:
                gain_factor = self.fixed_gain_factor

            elif self.uniform_gain:
                gain_factor = self._sample_gain(
                    input,
                    self.max_gain_factor,
                    gain_sampling=self.gain_sampling,
                    shape=self._uniform_gain_shape(input, batched=batched),
                )

            else:
                gain_factor = self._sample_gain(
                    input,
                    self.max_gain_factor,
                    gain_sampling=self.gain_sampling,
                    shape=tuple(input.shape),
                )

            input = gain_factor * input

        if self.poisson_sampling:
            noisy_input = torch.poisson(self.tau * input)
            noisy_input = noisy_input / self.tau
            return noisy_input, label
        else:
            return input, label


__all__ = ["POISSON_GENERATOR_DEFAULTS", "PoissonGeneratorDataset"]
