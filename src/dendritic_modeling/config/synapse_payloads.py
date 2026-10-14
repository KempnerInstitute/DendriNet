"""Typed option payloads of the dense-to-sparse and DeepST synapse layers.

The synapse layers read these payloads; configuration loading checks
``model.core.sparsity.dense_to_sparse`` and ``model.core.sparsity.deepst``
keys against them without importing the layers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseToSparseConfig:
    """Annealed dense-to-sparse schedule options."""

    initial_density: float = 1.0
    initial_k: int | None = None
    start_step: int = 0
    end_step: int = 1000
    update_interval: int = 1
    schedule: str = "cubic"
    freeze_on_end: bool = False
    advance_on_forward: bool = True
    prune_metric: str = "weight"


@dataclass(frozen=True)
class DeepSTConfig:
    """DeepST sparse-synapse options retained for compatibility."""

    use_noise: bool = True
    rewiring_mode: str = "global"
    rewire_frequency: int = 1
    synapses_per_branch: int | None = None
    sigma: float = 0.05
    excitatory_target_density: float = 0.1
    inhibitory_target_density: float = 0.1
    freeze_excitatory_connectivity: bool = False
    freeze_inhibitory_connectivity: bool = False
    # Initializer of the dense weight table and of regrown contacts. A
    # registered method regrows a contact at weight_threshold; None draws the
    # table and regrown values from the uniform log-space weight_init_range.
    init_method: str | None = "xavier_normal"
    weight_threshold: float = 1e-6
    credit_trace_decay: float = 0.95
    credit_candidate_pool_size: int = 8
    credit_turnover_fraction: float = 0.1
    credit_weak_active_pool_fraction: float = 0.5
    credit_min_observations: int = 5
    credit_swap_margin: float = 0.0
    credit_force_turnover: bool = False
    credit_selection: str = "credit"
    credit_warmup_steps: int = 5


__all__ = ["DeepSTConfig", "DenseToSparseConfig"]
