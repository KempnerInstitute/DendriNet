"""Typed configuration objects for dendritic branch synapse construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TopKConfig:
    """Sparse synapse selection and initialization options."""

    init_method: str = "xavier_normal"
    noise_level: float = 0.0
    type: str = "standard"
    weight_norm_order: int | None = None
    gamma: float = 1.0
    temperature: float = 0.5
    ultrafast: bool = False
    strategy: str = "none"


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
class IndexedSynapseConfig:
    """Indexed sparse-synapse and indexed-rewire options."""

    seed: int | None = None
    output_chunk_size: int = 2048
    candidate_size: int | None = None
    selection: str = "standard"
    rewire_frequency: int = 100
    rewire_quantile: float = 0.05
    rewire_until_step: int | None = None


@dataclass(frozen=True)
class ReactivationConfig:
    """Dendritic reactivation options."""

    enabled: bool = False
    type: str = "param_tanh"
    dendritic_activation: str | None = None
    init_m: float = 1
    init_b: float = 0.5
    init_policy: str = "analytical"
    occupancy_quantile_low: float | None = None
    occupancy_quantile_high: float | None = None
    occupancy_target_low: float | None = None
    occupancy_target_high: float | None = None
    calibration_min_quantile_width: float = 1e-3
    calibration_max_m: float = 50.0
    calibration_revert_on_invalid: bool = True
    sigma_aware_k: float = 0.25
    strategy: str = "none"


@dataclass(frozen=True)
class MorphologyConfig:
    """Branch voltage, morphology, and output aggregation options."""

    use_shunting: bool = True
    use_additive_normalization: bool = False
    epsilon: float = 1e-8
    weight_transform: str = "exp"
    efficient_blocklinear: bool = False
    blocklinear_strategy: str = "none"


@dataclass(frozen=True)
class AdaptiveInitConfig:
    """Branch-layer initialization options."""

    dbl_init_method: str = "analytical_expectation"
    adaptive_initialization: bool = True
    adaptive_initialization_policy: str = "legacy_scale"
    adaptive_target_conductance: float = 5.0
    branch_factors: tuple[int, ...] | None = None


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
    init_method: str | None = None
    weight_threshold: float = 1e-6


@dataclass(frozen=True)
class DendriticSynapseConfig:
    """Grouped branch-layer options with legacy-kwarg conversion helpers."""

    topk: TopKConfig = TopKConfig()
    dense_to_sparse: DenseToSparseConfig = DenseToSparseConfig()
    indexed: IndexedSynapseConfig = IndexedSynapseConfig()
    reactivation: ReactivationConfig = ReactivationConfig()
    morphology: MorphologyConfig = MorphologyConfig()
    initialization: AdaptiveInitConfig = AdaptiveInitConfig()
    deepst: DeepSTConfig = DeepSTConfig()
    print_hooks: bool = False

    @classmethod
    def from_config(
        cls,
        config: DendriticSynapseConfig | Mapping[str, Any],
    ) -> DendriticSynapseConfig:
        """Normalize an existing config object or legacy kwarg mapping."""
        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            return cls.from_kwargs(config)
        raise TypeError("synapse_config must be a DendriticSynapseConfig or mapping")

    @classmethod
    def from_kwargs(
        cls,
        mapping: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> DendriticSynapseConfig:
        """Build a typed config from legacy ``DendriticBranchLayer`` kwargs."""
        source = dict(mapping or {})
        source.update(kwargs)
        branch_factors = source.get("branch_factors", None)
        if branch_factors is not None:
            branch_factors = tuple(branch_factors)

        return cls(
            topk=TopKConfig(
                init_method=source.get("topk_init_method", "xavier_normal"),
                noise_level=source.get("topk_noise_level", 0.0),
                type=source.get("topk_type", "standard"),
                weight_norm_order=source.get("topk_weight_norm_order", None),
                gamma=source.get("topk_gamma", 1.0),
                temperature=source.get("topk_temperature", 0.5),
                ultrafast=source.get("topk_ultrafast", False),
                strategy=source.get("topk_strategy", "none"),
            ),
            dense_to_sparse=DenseToSparseConfig(
                initial_density=source.get("dense_to_sparse_initial_density", 1.0),
                initial_k=source.get("dense_to_sparse_initial_k", None),
                start_step=source.get("dense_to_sparse_start_step", 0),
                end_step=source.get("dense_to_sparse_end_step", 1000),
                update_interval=source.get("dense_to_sparse_update_interval", 1),
                schedule=source.get("dense_to_sparse_schedule", "cubic"),
                freeze_on_end=source.get("dense_to_sparse_freeze_on_end", False),
                advance_on_forward=source.get(
                    "dense_to_sparse_advance_on_forward", True
                ),
                prune_metric=source.get("dense_to_sparse_prune_metric", "weight"),
            ),
            indexed=IndexedSynapseConfig(
                seed=source.get("indexed_seed", None),
                output_chunk_size=source.get("indexed_output_chunk_size", 2048),
                candidate_size=source.get("indexed_candidate_size", None),
                selection=source.get("indexed_selection", "standard"),
                rewire_frequency=source.get("indexed_rewire_frequency", 100),
                rewire_quantile=source.get("indexed_rewire_quantile", 0.05),
                rewire_until_step=source.get("indexed_rewire_until_step", None),
            ),
            reactivation=ReactivationConfig(
                enabled=source.get("reactivate", False),
                type=source.get("reactivation_type", "param_tanh"),
                dendritic_activation=source.get("dendritic_activation", None),
                init_m=source.get("reactivation_init_m", 1),
                init_b=source.get("reactivation_init_b", 0.5),
                init_policy=source.get("reactivation_init_policy", "analytical"),
                occupancy_quantile_low=source.get(
                    "reactivation_occupancy_quantile_low", None
                ),
                occupancy_quantile_high=source.get(
                    "reactivation_occupancy_quantile_high", None
                ),
                occupancy_target_low=source.get(
                    "reactivation_occupancy_target_low", None
                ),
                occupancy_target_high=source.get(
                    "reactivation_occupancy_target_high", None
                ),
                calibration_min_quantile_width=source.get(
                    "reactivation_calibration_min_quantile_width", 1e-3
                ),
                calibration_max_m=source.get("reactivation_calibration_max_m", 50.0),
                calibration_revert_on_invalid=source.get(
                    "reactivation_calibration_revert_on_invalid", True
                ),
                sigma_aware_k=source.get("reactivation_sigma_aware_k", 0.25),
                strategy=source.get("reactivation_strategy", "none"),
            ),
            morphology=MorphologyConfig(
                use_shunting=source.get("use_shunting", True),
                use_additive_normalization=source.get(
                    "use_additive_normalization", False
                ),
                epsilon=source.get("epsilon", 1e-8),
                weight_transform=source.get("weight_transform", "exp"),
                efficient_blocklinear=source.get("efficient_blocklinear", False),
                blocklinear_strategy=source.get("blocklinear_strategy", "none"),
            ),
            initialization=AdaptiveInitConfig(
                dbl_init_method=source.get("dbl_init_method", "analytical_expectation"),
                adaptive_initialization=source.get("adaptive_initialization", True),
                adaptive_initialization_policy=source.get(
                    "adaptive_initialization_policy", "legacy_scale"
                ),
                adaptive_target_conductance=source.get(
                    "adaptive_target_conductance", 5.0
                ),
                branch_factors=branch_factors,
            ),
            deepst=DeepSTConfig(
                use_noise=source.get("use_noise", True),
                rewiring_mode=source.get("rewiring_mode", "global"),
                rewire_frequency=source.get("rewire_frequency", 1),
                synapses_per_branch=source.get("synapses_per_branch", None),
                sigma=source.get("sigma", 0.05),
                excitatory_target_density=source.get("excitatory_target_density", 0.1),
                inhibitory_target_density=source.get("inhibitory_target_density", 0.1),
                freeze_excitatory_connectivity=source.get(
                    "freeze_excitatory_connectivity", False
                ),
                freeze_inhibitory_connectivity=source.get(
                    "freeze_inhibitory_connectivity", False
                ),
                init_method=source.get("init_method", None),
                weight_threshold=source.get("weight_threshold", 1e-6),
            ),
            print_hooks=source.get("print_hooks", False),
        )

    @classmethod
    def split_kwargs(
        cls,
        kwargs: Mapping[str, Any],
    ) -> tuple[DendriticSynapseConfig, dict[str, Any]]:
        """Return the config built from known keys plus unconsumed kwargs."""
        known = {key: value for key, value in kwargs.items() if key in LEGACY_KEYS}
        extra = {key: value for key, value in kwargs.items() if key not in LEGACY_KEYS}
        return cls.from_kwargs(known), extra

    def to_kwargs(self) -> dict[str, Any]:
        """Convert back to legacy branch-layer kwargs."""
        return {
            "topk_init_method": self.topk.init_method,
            "topk_noise_level": self.topk.noise_level,
            "topk_type": self.topk.type,
            "topk_weight_norm_order": self.topk.weight_norm_order,
            "topk_gamma": self.topk.gamma,
            "topk_temperature": self.topk.temperature,
            "topk_ultrafast": self.topk.ultrafast,
            "topk_strategy": self.topk.strategy,
            "dense_to_sparse_initial_density": self.dense_to_sparse.initial_density,
            "dense_to_sparse_initial_k": self.dense_to_sparse.initial_k,
            "dense_to_sparse_start_step": self.dense_to_sparse.start_step,
            "dense_to_sparse_end_step": self.dense_to_sparse.end_step,
            "dense_to_sparse_update_interval": self.dense_to_sparse.update_interval,
            "dense_to_sparse_schedule": self.dense_to_sparse.schedule,
            "dense_to_sparse_freeze_on_end": self.dense_to_sparse.freeze_on_end,
            "dense_to_sparse_advance_on_forward": (
                self.dense_to_sparse.advance_on_forward
            ),
            "dense_to_sparse_prune_metric": self.dense_to_sparse.prune_metric,
            "indexed_seed": self.indexed.seed,
            "indexed_output_chunk_size": self.indexed.output_chunk_size,
            "indexed_candidate_size": self.indexed.candidate_size,
            "indexed_selection": self.indexed.selection,
            "indexed_rewire_frequency": self.indexed.rewire_frequency,
            "indexed_rewire_quantile": self.indexed.rewire_quantile,
            "indexed_rewire_until_step": self.indexed.rewire_until_step,
            "use_shunting": self.morphology.use_shunting,
            "use_additive_normalization": self.morphology.use_additive_normalization,
            "epsilon": self.morphology.epsilon,
            "weight_transform": self.morphology.weight_transform,
            "efficient_blocklinear": self.morphology.efficient_blocklinear,
            "blocklinear_strategy": self.morphology.blocklinear_strategy,
            "reactivate": self.reactivation.enabled,
            "reactivation_type": self.reactivation.type,
            "dendritic_activation": self.reactivation.dendritic_activation,
            "reactivation_init_m": self.reactivation.init_m,
            "reactivation_init_b": self.reactivation.init_b,
            "reactivation_init_policy": self.reactivation.init_policy,
            "reactivation_occupancy_quantile_low": (
                self.reactivation.occupancy_quantile_low
            ),
            "reactivation_occupancy_quantile_high": (
                self.reactivation.occupancy_quantile_high
            ),
            "reactivation_occupancy_target_low": (
                self.reactivation.occupancy_target_low
            ),
            "reactivation_occupancy_target_high": (
                self.reactivation.occupancy_target_high
            ),
            "reactivation_calibration_min_quantile_width": (
                self.reactivation.calibration_min_quantile_width
            ),
            "reactivation_calibration_max_m": self.reactivation.calibration_max_m,
            "reactivation_calibration_revert_on_invalid": (
                self.reactivation.calibration_revert_on_invalid
            ),
            "reactivation_sigma_aware_k": self.reactivation.sigma_aware_k,
            "reactivation_strategy": self.reactivation.strategy,
            "dbl_init_method": self.initialization.dbl_init_method,
            "adaptive_initialization": self.initialization.adaptive_initialization,
            "adaptive_initialization_policy": (
                self.initialization.adaptive_initialization_policy
            ),
            "adaptive_target_conductance": (
                self.initialization.adaptive_target_conductance
            ),
            "branch_factors": (
                None
                if self.initialization.branch_factors is None
                else list(self.initialization.branch_factors)
            ),
            "use_noise": self.deepst.use_noise,
            "rewiring_mode": self.deepst.rewiring_mode,
            "rewire_frequency": self.deepst.rewire_frequency,
            "synapses_per_branch": self.deepst.synapses_per_branch,
            "sigma": self.deepst.sigma,
            "excitatory_target_density": self.deepst.excitatory_target_density,
            "inhibitory_target_density": self.deepst.inhibitory_target_density,
            "freeze_excitatory_connectivity": (
                self.deepst.freeze_excitatory_connectivity
            ),
            "freeze_inhibitory_connectivity": (
                self.deepst.freeze_inhibitory_connectivity
            ),
            "init_method": self.deepst.init_method,
            "weight_threshold": self.deepst.weight_threshold,
            "print_hooks": self.print_hooks,
        }


LEGACY_KEYS = frozenset(DendriticSynapseConfig().to_kwargs())


__all__ = [
    "AdaptiveInitConfig",
    "DeepSTConfig",
    "DendriticSynapseConfig",
    "DenseToSparseConfig",
    "IndexedSynapseConfig",
    "MorphologyConfig",
    "ReactivationConfig",
    "TopKConfig",
]
