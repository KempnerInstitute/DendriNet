"""Poisson-noise dataset wrappers."""

from math import log
from typing import Optional

from sklearn.decomposition import NMF

import numpy as np
import torch
from torch.utils.data import Dataset


def _inputs_to_nonneg_matrix(inputs) -> np.ndarray:
    """Flatten inputs to a nonnegative ``(n_samples, n_features)`` matrix for NMF."""
    if torch.is_tensor(inputs):
        arr = inputs.detach().cpu().numpy()
    else:
        arr = np.asarray(inputs)
    arr = arr.reshape(arr.shape[0], -1).astype(np.float64)
    # NMF requires nonnegative inputs; raw (unnormalized) intensities/rates are >= 0,
    # but clip defensively so callers passing slightly-negative values don't error.
    np.clip(arr, 0.0, None, out=arr)
    return arr


def _build_multiscale_group_components(
    inputs,
    n_groups_per_scale: list[int],
    *,
    var_eps: float = 1e-6,
    nmf_max_iter: int = 1000,
) -> torch.Tensor:
    """Assign each input feature to a coarse gain region via NMF.

    Runs non-negative matrix factorization on the active (non-constant) features
    of the training inputs and hard-assigns each active feature to its dominant
    NMF component. Always-zero / constant features (variance <= ``var_eps``) carry
    no covariation signal and are labeled ``-1`` (background). Background features
    receive only the global gain stage downstream, so they cannot pollute a real
    component via argmax tie-breaking.

    Args:
        inputs: Training inputs, shape ``(n_samples, ...)``; flattened internally.
        n_groups_per_scale: Target number of groups per scale (NMF components).
        var_eps: Features with across-sample variance <= this are background.
        random_state: Seed for the NMF solver.
        nmf_max_iter: Maximum NMF iterations.

    Returns:
        ``LongTensor`` of shape ``(n_features,)`` mapping each flattened feature
        index to a coarse region id in ``[0, n_regions)``, or ``-1`` for
        background features (which receive only the global gain).
    """
    X = _inputs_to_nonneg_matrix(inputs)
    n_samples, n_features = X.shape
    n_scales = len(n_groups_per_scale) + 1

    active: np.ndarray = X.var(axis=0) > var_eps
    n_components_list = []
    for n_groups in n_groups_per_scale:
        n_components_list.append(min(int(n_groups), int(active.sum()), n_samples))
    group_components = np.zeros((sum(n_components_list), n_features))

    if active.any():
        H_list = []
        for n_components in n_components_list:
            H: np.ndarray = (
                NMF(
                    n_components=n_components,
                    init="nndsvd",
                    max_iter=nmf_max_iter,
                )
                .fit(X[:, active])
                .components_
            )
            H = H / (H.sum(axis=0, keepdims=True) * n_scales)
            H_list.append(H)
        G = np.concatenate(H_list, axis=0)
        group_components[:, active] = G

        background_component = np.zeros((1, n_features))
        background_component[:, ~active] = 1.0
        background_component = background_component / n_scales
        
        group_components = np.concatenate([group_components, background_component], axis=0)
    return torch.from_numpy(group_components)


class PoissonGeneratorDataset(Dataset):
    """Dataset that applies Poisson noise to base dataset samples."""

    def __init__(
        self,
        base_inputs: torch.Tensor,
        base_labels: torch.Tensor,
        multiplicative_gain: bool = True,
        multiscale_gain: bool = False,
        group_components: Optional[torch.Tensor] = None,
        fixed_gain_factor: Optional[float] = None,
        min_gain_factor: Optional[float] = None,
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
        if min_gain_factor is not None:
            min_gain_factor = float(min_gain_factor)
        if max_gain_factor is not None:
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
        
        if multiscale_gain and isinstance(group_components, torch.Tensor):
            assert (
                group_components.ndim == 2 
                and group_components.shape[1] == sum(base_inputs.shape[1:])
            ), (
                f"group_components must have shape (n_groups, n_features), "
                f"got {group_components.shape}"
            )
            assert uniform_gain == True, "uniform_gain must be True when using multiscale_gain"
            group_components = group_components.to(base_inputs.dtype)
        elif multiscale_gain:
            multiscale_gain = False
            group_components = None

        if (
            min_gain_factor is not None 
            and max_gain_factor is not None
            and min_gain_factor > max_gain_factor
        ):
            raise ValueError(
                f"min_gain_factor must be <= max_gain_factor when provided, "
                f"got {min_gain_factor} > {max_gain_factor}"
            )

        self.base_inputs = base_inputs
        self.base_labels = base_labels
        self.multiplicative_gain = multiplicative_gain
        self.multiscale_gain = multiscale_gain
        self.group_components = group_components
        self.fixed_gain_factor = fixed_gain_factor
        self.min_gain_factor = min_gain_factor
        self.max_gain_factor = max_gain_factor
        self.uniform_gain = uniform_gain
        self.gain_sampling = gain_sampling
        self.poisson_sampling = poisson_sampling
        self.stimulus_duration = stimulus_duration
        self.tau = stimulus_duration

        self._fixed_gf = None
        self._max_gf = None

    def _set_gain_factor(self, gain_factor: float | torch.Tensor):
        self._fixed_gf = gain_factor

    def _set_max_gain_factor(self, max_gain_factor: float | torch.Tensor):
        self._max_gf = max_gain_factor
    
    def _clear_gain(self):
        self._fixed_gf = None
        self._max_gf = None

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
    def _sample_gain_singlestage(
        min_gain_factor: float,
        max_gain_factor: float,
        *,
        gain_sampling: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        random_gain = torch.rand(shape, dtype=dtype, device=device)
        if gain_sampling == "log_uniform":
            log_min_gain_factor = log(min_gain_factor)
            log_max_gain_factor = log(max_gain_factor)
            log_range = log_max_gain_factor - log_min_gain_factor
            return torch.exp((random_gain * log_range) + log_min_gain_factor)
        if gain_sampling == "linear_uniform":
            linear_range = max_gain_factor - min_gain_factor
            return (random_gain * linear_range) + min_gain_factor
        raise ValueError(f"Unknown gain_sampling={gain_sampling!r}")
    
    @staticmethod
    def _sample_gain_multiscale(
        group_components: torch.Tensor,
        min_gain_factor: float,
        max_gain_factor: float,
        *,
        gain_sampling: str,
        batch_shape: tuple[int, ...],
        feature_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # ``group_components`` has shape ``(n_groups, n_features)`` and its columns
        # are normalized to sum to 1: entry ``[r, i]`` is the soft membership weight
        # of feature ``i`` in group ``r`` (a convex combination over groups, per
        # feature, including the background group). Features index the flattened
        # (row-major) input, matching ``input.reshape(B, -1)``.
        #
        # We sample one gain per group per sample -- shape ``(*batch_shape,
        # n_groups)`` -- and combine into a per-feature gain by contracting the
        # group axis against ``group_components``:
        #   G = g_group @ group_components  -> (*batch_shape, n_features)
        # For ``log_uniform`` we contract the *log* gains (a weighted geometric
        # mean, the natural combination for multiplicative gains); for
        # ``linear_uniform`` we contract the linear gains (weighted arithmetic
        # mean). Because the per-feature weights sum to 1, the combined per-feature
        # gain always stays within ``[min_gain_factor, max_gain_factor]``.
        group_components = group_components.to(dtype=dtype, device=device)
        n_groups = group_components.shape[0]

        random_group = torch.rand(*batch_shape, n_groups, dtype=dtype, device=device)

        if gain_sampling == "log_uniform":
            log_min_gain_factor = log(min_gain_factor)
            log_max_gain_factor = log(max_gain_factor)
            log_range = log_max_gain_factor - log_min_gain_factor
            log_g_group = (random_group * log_range) + log_min_gain_factor
            gain_factor = torch.exp(log_g_group @ group_components)

        elif gain_sampling == "linear_uniform":
            linear_range = max_gain_factor - min_gain_factor
            g_group = (random_group * linear_range) + min_gain_factor
            gain_factor = g_group @ group_components

        else:
            raise ValueError(f"Unknown gain_sampling={gain_sampling!r}")

        # Map flattened per-feature gain back onto the input layout.
        gain_factor = gain_factor.reshape(*batch_shape, *feature_shape)
        return gain_factor

    def _sample_gain(
        self,
        group_components: Optional[torch.Tensor],
        min_gain_factor: float,
        max_gain_factor: float,
        *,
        gain_sampling: str,
        shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        feature_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.multiscale_gain:
            return self._sample_gain_multiscale(
                group_components=group_components,
                min_gain_factor=min_gain_factor,
                max_gain_factor=max_gain_factor,
                gain_sampling=gain_sampling,
                batch_shape=batch_shape,
                feature_shape=feature_shape,
                dtype=dtype,
                device=device,
            )
        else:
            return self._sample_gain_singlestage(
                min_gain_factor=min_gain_factor,
                max_gain_factor=max_gain_factor,
                gain_sampling=gain_sampling,
                shape=shape,
                dtype=dtype,
                device=device,
            )

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

        # Per-sample batch/feature layout, used by the multistage gain path to
        # sample region gains over the batch dim only and reshape back to the
        # input layout (supports flattened and image-shaped inputs).
        batch_shape = (input.shape[0],) if batched and input.ndim > 0 else ()
        feature_shape = tuple(input.shape[len(batch_shape):])

        if self._fixed_gf is not None:
            input = input * self._fixed_gf

        elif self._max_gf is not None:
            if self.uniform_gain:
                shape = self._uniform_gain_shape(input, batched=batched)
            else:
                shape = tuple(input.shape)
            
            gain_factor = self._sample_gain(
                group_components=self.group_components,
                min_gain_factor=self.min_gain_factor,
                max_gain_factor=self._max_gf,
                gain_sampling=self.gain_sampling,
                shape=shape,
                batch_shape=batch_shape,
                feature_shape=feature_shape,
                dtype=input.dtype,
                device=input.device,
            )
            input = gain_factor * input

        elif self.multiplicative_gain:
            if self.fixed_gain_factor is not None:
                gain_factor = self.fixed_gain_factor
            else:
                if self.uniform_gain:
                    shape = self._uniform_gain_shape(input, batched=batched)
                else:
                    shape = tuple(input.shape)
                
                gain_factor = self._sample_gain(
                    group_components=self.group_components,
                    min_gain_factor=self.min_gain_factor,
                    max_gain_factor=self.max_gain_factor,
                    gain_sampling=self.gain_sampling,
                    shape=shape,
                    batch_shape=batch_shape,
                    feature_shape=feature_shape,
                    dtype=input.dtype,
                    device=input.device,
                )
            input = gain_factor * input

        if self.poisson_sampling:
            noisy_input = torch.poisson(self.tau * input)
            noisy_input = noisy_input / self.tau
            return noisy_input, label
        else:
            return input, label


__all__ = ["PoissonGeneratorDataset"]
