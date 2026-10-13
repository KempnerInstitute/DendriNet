"""Independent random-stream resolution for reproducible experiments."""

from __future__ import annotations

import os
import random
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

_INDEPENDENT_SEED_FIELDS = (
    "dataset_seed",
    "split_seed",
    "model_seed",
    "topology_seed",
    "loader_seed",
    "evaluation_seed",
    "probe_seed",
)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _set_config_value(config: Any, key: str, value: Any) -> None:
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


@dataclass(frozen=True)
class ResolvedExperimentSeeds:
    """Concrete RNG seeds used by one experiment run."""

    seed: int
    dataset_seed: int
    split_seed: int
    model_seed: int
    topology_seed: int
    loader_seed: int
    evaluation_seed: int
    probe_seed: int
    strict_deterministic: bool

    def asdict(self) -> dict[str, int | bool]:
        return asdict(self)


def resolve_experiment_seeds(
    experiment_config: Any,
    *,
    write_back: bool = False,
) -> ResolvedExperimentSeeds:
    """Resolve optional RNG streams against the legacy ``seed`` field.

    Resolution happens after configuration/CLI overrides. This means a legacy
    config containing only ``seed`` retains one familiar control, while an
    explicit stream remains independent from later changes to the base seed.
    """
    base_seed = int(_config_value(experiment_config, "seed", 42))
    resolved_values: dict[str, int] = {}
    for field_name in _INDEPENDENT_SEED_FIELDS:
        configured = _config_value(experiment_config, field_name, None)
        resolved_values[field_name] = (
            base_seed if configured is None else int(configured)
        )

    resolved = ResolvedExperimentSeeds(
        seed=base_seed,
        strict_deterministic=bool(
            _config_value(experiment_config, "strict_deterministic", False)
        ),
        **resolved_values,
    )
    if write_back:
        for field_name, value in resolved.asdict().items():
            _set_config_value(experiment_config, field_name, value)
    return resolved


MATMUL_PRECISIONS = ("highest", "high", "medium")
_RUNTIME_FLAGS = (
    "deterministic",
    "cudnn_benchmark",
    "allow_tf32",
    "float32_matmul_precision",
)


def resolve_runtime_performance_settings(
    experiment_config: Any, *, write_back: bool = False
) -> dict[str, Any]:
    """Resolve the determinism and TF32 flags of a run.

    - ``strict_deterministic`` requires exact math: deterministic kernels, no
      cuDNN autotuning, no TF32 and ``float32_matmul_precision: highest``.
      A written flag that contradicts it raises.
    - ``allow_tf32`` and ``float32_matmul_precision`` describe one matmul
      setting (``highest`` means no TF32). When only one is written it sets
      the other; when both are written they must agree.
    - Flags left unset (missing or ``None``) follow ``fast_mode``.

    With ``write_back`` the resolved values are stored on the config, so the
    saved ``config.json`` records what the run used.
    """

    def configured(key: str) -> Any:
        return _config_value(experiment_config, key, None)

    fast_mode = bool(configured("fast_mode") or False)
    strict = bool(configured("strict_deterministic") or False)

    precision = configured("float32_matmul_precision")
    if precision is not None:
        precision = str(precision).strip().lower()
        if precision not in MATMUL_PRECISIONS:
            raise ValueError(
                "experiment.float32_matmul_precision must be one of "
                f"{list(MATMUL_PRECISIONS)}, got {precision!r}"
            )
    allow_tf32 = configured("allow_tf32")
    if allow_tf32 is not None:
        allow_tf32 = bool(allow_tf32)
    if allow_tf32 is not None and precision is not None:
        if allow_tf32 != (precision != "highest"):
            raise ValueError(
                f"experiment.allow_tf32={allow_tf32} contradicts "
                f"experiment.float32_matmul_precision={precision!r}: 'highest' "
                "disables TF32 matmuls and 'high'/'medium' enable them. Set one "
                "of the two."
            )
    elif allow_tf32 is not None:
        precision = "high" if allow_tf32 else "highest"
    elif precision is not None:
        allow_tf32 = precision != "highest"

    deterministic = configured("deterministic")
    cudnn_benchmark = configured("cudnn_benchmark")
    if strict:
        conflicts = [
            name
            for name, value, exact in (
                ("deterministic", deterministic, True),
                ("cudnn_benchmark", cudnn_benchmark, False),
                ("allow_tf32", allow_tf32, False),
            )
            if value is not None and bool(value) != exact
        ]
        if precision not in (None, "highest"):
            conflicts.append("float32_matmul_precision")
        if conflicts:
            raise ValueError(
                "experiment.strict_deterministic requires exact math, which "
                + ", ".join(f"experiment.{name}" for name in conflicts)
                + " contradicts; remove these settings"
            )
        resolved = {
            "deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
    else:
        deterministic = (
            (not fast_mode) if deterministic is None else bool(deterministic)
        )
        if cudnn_benchmark is None:
            cudnn_benchmark = fast_mode and not deterministic
        if allow_tf32 is None:
            allow_tf32 = fast_mode
            precision = "high" if fast_mode else "highest"
        resolved = {
            "deterministic": deterministic,
            "cudnn_benchmark": bool(cudnn_benchmark),
            "allow_tf32": allow_tf32,
            "float32_matmul_precision": precision,
        }
    if write_back and experiment_config is not None:
        for key, value in resolved.items():
            try:
                _set_config_value(experiment_config, key, value)
            except (AttributeError, TypeError):
                pass
    return resolved


def set_seed(
    seed: int,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    allow_tf32: bool = False,
    float32_matmul_precision: str = "highest",
    strict_deterministic: bool = False,
) -> dict[str, Any]:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device) and set numerics.

    Args:
        seed: Random seed value.
        deterministic: Use deterministic cuDNN kernels.
        cudnn_benchmark: Let cuDNN autotune kernels (ignored when deterministic).
        allow_tf32: Allow TF32 in cuDNN convolutions and float32 matmuls.
        float32_matmul_precision: ``highest`` (no TF32), ``high`` or ``medium``;
            it must agree with ``allow_tf32``.
        strict_deterministic: Require PyTorch's deterministic algorithms and a
            deterministic cuBLAS workspace, and force exact math (no TF32,
            ``highest`` precision, deterministic cuDNN). Unsupported
            nondeterministic operations then raise when they run. The cuBLAS
            workspace setting applies only to cuBLAS handles created afterwards,
            so call this before any CUDA work.

    Returns:
        The numerics state in effect afterwards, for the run record.
    """
    if strict_deterministic:
        deterministic, cudnn_benchmark = True, False
        allow_tf32, float32_matmul_precision = False, "highest"
        cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif cublas_workspace not in {":4096:8", ":16:8"}:
            warnings.warn(
                "strict_deterministic requested, but CUBLAS_WORKSPACE_CONFIG="
                f"{cublas_workspace!r} is not a deterministic setting; use "
                "':4096:8' or ':16:8' before launching Python.",
                RuntimeWarning,
                stacklevel=2,
            )
    precision = str(float32_matmul_precision).strip().lower()
    if precision not in MATMUL_PRECISIONS:
        raise ValueError(
            f"float32_matmul_precision must be one of {list(MATMUL_PRECISIONS)}, "
            f"got {float32_matmul_precision!r}"
        )
    if bool(allow_tf32) != (precision != "highest"):
        raise ValueError(
            f"allow_tf32={allow_tf32} contradicts float32_matmul_precision="
            f"{precision!r}; 'highest' disables TF32 matmuls"
        )

    torch.manual_seed(seed)  # seeds the CPU generator and every CUDA device
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and not deterministic
    if strict_deterministic:
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.set_float32_matmul_precision(precision)
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def alert_not_deterministic(operation: str, remedy: str) -> None:
    """Refuse ``operation`` when PyTorch's deterministic algorithms are required.

    Library kernels that accumulate with atomic adds call this before they
    run, as PyTorch's own nondeterministic operations do: under
    ``torch.use_deterministic_algorithms(True)`` it raises ``RuntimeError``, in
    the ``warn_only`` mode it warns, and otherwise it does nothing.
    ``remedy`` names a deterministic alternative.
    """
    if not torch.are_deterministic_algorithms_enabled():
        return
    message = (
        f"{operation} is not deterministic, but "
        "torch.use_deterministic_algorithms(True) is set (for example by "
        f"experiment.strict_deterministic). {remedy}"
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(message, UserWarning, stacklevel=3)
        return
    raise RuntimeError(message)


@contextmanager
def preserved_random_state():
    """Restore Python, NumPy, CPU, and visible CUDA RNGs after a block."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def isolated_random_seed(seed: int):
    """Temporarily seed Python, NumPy, CPU, and visible CUDA RNGs.

    Analysis code still contains a mixture of explicit generators and legacy
    global draws. This context makes the latter reproducible without allowing
    evaluation to perturb the subsequent training RNG state.
    """
    with preserved_random_state():
        normalized_seed = int(seed) % ((1 << 63) - 1)
        random.seed(normalized_seed)
        np.random.seed(normalized_seed % (1 << 32))
        torch.manual_seed(normalized_seed)
        yield


__all__ = [
    "MATMUL_PRECISIONS",
    "ResolvedExperimentSeeds",
    "alert_not_deterministic",
    "isolated_random_seed",
    "preserved_random_state",
    "resolve_experiment_seeds",
    "resolve_runtime_performance_settings",
    "set_seed",
]
