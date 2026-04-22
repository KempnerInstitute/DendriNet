from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.networks.architectures.classical.autoencoder import (
    CNNAutoencoder,
    CNNVariationalAutoencoder,
    MLPAutoencoder,
    MLPVariationalAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    EffectiveMatchedParamMLP,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    PointMatchedParamMLP,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)

# All EINet type strings handled via ConfigurableEINetwork (+ dendritic_mlp special-cased)
_EINET_TYPES: set[str] = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_mlp",
    "flat_shunting",
    "flat_additive",
    "flat_mlp",
}

_UNIFIED_EI_TYPES: set[str] = {
    "ei_unified",
    "unified_ei",
    "ei_net",
    "unified_einet",
    # Parallel aliases that mirror the BP / feedforward vocabulary
    # (`dendritic_shunting`, `dendritic_additive`, `flat_shunting`,
    # `flat_additive`). An `rnn_` prefix disambiguates them from the
    # non-recurrent BP types. The factory below canonicalizes each alias
    # to the appropriate `use_shunting` / `use_additive_normalization`
    # pair on every excitatory population, leaving the `ei_unified` path
    # unchanged for backward compatibility.
    "rnn_dendritic_shunting",
    "rnn_dendritic_additive",
    "rnn_flat_shunting",
    "rnn_flat_additive",
}

# Type-alias implied overrides on each excitatory population.
# The alias is resolved at build time; `ei_unified` is a no-op (explicit
# config wins). The BP `canonicalize_model_core_flags` uses the same idea.
#
# For `*_additive` aliases: we also force an additive-safe reactivation-gate
# init (m=0.1, b=0.0, policy="fixed"). This bypasses the analytical init
# rule in `analytical_expectation_dbl_init`, which on non-leaf additive
# layers computes `E_Vinf = E_g_exc - E_g_inh + 0.5 * g_branch` ≈ 1.02 and
# places the gate center above the useful voltage range. The BP-path
# `dendritic_additive` type ships equivalent additive-safe defaults via its
# own canonicalization; the aliases here give the recurrent/unified-EI
# path the same safety net. Existing `ei_unified` configs with
# `use_shunting: false` and `reactivation_init_policy: analytical`
# continue to dispatch the (buggy) analytical rule unchanged — users opt
# into the fix by switching `core.type` to the alias.
_UNIFIED_EI_ALIAS_OVERRIDES: dict[str, dict[str, object]] = {
    "rnn_dendritic_shunting": {
        "use_shunting": True,
    },
    "rnn_dendritic_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
        "reactivation_init_m": 0.1,
        "reactivation_init_b": 0.0,
        "reactivation_init_policy": "fixed",
    },
    "rnn_flat_shunting": {
        "use_shunting": True,
        "branch_factors": [1],
    },
    "rnn_flat_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
        "reactivation_init_m": 0.1,
        "reactivation_init_b": 0.0,
        "reactivation_init_policy": "fixed",
        "branch_factors": [1],
    },
}

# Spatial dendritic cores: flatten encoder spatial output → EINet → project.
# Tensor-map baseline: flatten → DendriNet → project (no spatial locality).
_SPATIAL_DENDRITIC_TYPES: set[str] = {
    "hierarchical_dendritic_tensor_map",
    "dendritic_tensor_map",
    "tensor_map",
}

# Spatial dendritic conv: F.unfold → shared DendriNet on local patches → fold.
# Preserves spatial structure (B, C_out, H_out, W_out).
# Canonical name: hierarchical_dendritic_conv
_SPATIAL_DENDRITIC_CONV_TYPES: set[str] = {
    "hierarchical_dendritic_conv",
    "spatial_dendritic_conv",
}

_BASELINE_RNN_TYPES: dict[str, str] = {
    "gru": "gru",
    "lstm": "lstm",
    "vanilla_rnn": "vanilla",
    "rnn": "vanilla",
}


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of config-like objects into a plain mapping."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, DictConfig):
        from omegaconf import OmegaConf

        converted = OmegaConf.to_container(obj, resolve=True)
        return converted if isinstance(converted, dict) else {}
    if is_dataclass(obj):
        converted = asdict(obj)
        return converted if isinstance(converted, dict) else {}
    # Fallback for custom config objects exposing attrs.
    try:
        return dict(obj)
    except Exception:
        return {}


def _layer_value(
    value: Any, layer_idx: int, default: Any = 0, *, allow_empty_list: bool = False
) -> Any:
    """Read scalar/list per-layer values with repeat-last semantics."""
    if isinstance(value, list):
        if not value:
            return [] if allow_empty_list else default
        if layer_idx < len(value):
            return value[layer_idx]
        return value[-1]
    if value is None:
        return default
    return value


def _normalize_sparsity_type(sparsity_type: Any) -> str:
    """Normalize public sparsity aliases to the internal TopK strategy names."""
    normalized = "standard" if sparsity_type is None else str(sparsity_type).lower()
    if normalized == "topk":
        return "standard"
    return normalized


def _build_recurrent_einet_from_structured_config(
    *,
    type: str,
    raw_params: dict[str, Any],
    input_dim: int,
):
    """Translate structured EINet config + recurrent synapse counts to recurrent EINetwork.

    Recurrence is triggered when any ``connectivity.rec_*`` synapse count is > 0,
    or (for backward compatibility) when ``recurrent_ei.enabled`` is True.
    """
    from dendritic_modeling.networks.architectures.recurrent import (
        EILayerConfig,
        EINetwork,
        EINetworkConfig,
        PopulationConfig,
    )

    recurrent_cfg = _to_plain_mapping(raw_params.get("recurrent_ei", {}))
    architecture = _to_plain_mapping(raw_params.get("architecture", {}))
    connectivity = _to_plain_mapping(raw_params.get("connectivity", {}))
    transfer = _to_plain_mapping(raw_params.get("transfer", {}))
    morphology = _to_plain_mapping(raw_params.get("morphology", {}))
    sparsity = _to_plain_mapping(raw_params.get("sparsity", {}))
    reactivation = _to_plain_mapping(raw_params.get("reactivation", {}))
    blocklinear = _to_plain_mapping(raw_params.get("blocklinear", {}))
    implementation = _to_plain_mapping(raw_params.get("implementation", {}))
    deepst = _to_plain_mapping(sparsity.get("deepst", {}))

    # --- Resolve recurrent synapse counts (new location first, old fallback) ---
    def _rec_syn(new_key, *old_keys, fallback=0):
        if new_key in connectivity:
            return connectivity[new_key]
        for k in old_keys:
            if k in recurrent_cfg:
                return recurrent_cfg[k]
        return fallback

    rec_ee = _rec_syn(
        "rec_ee_synapses_per_branch",
        "rec_excitatory_synapses_per_layer",
        "rec_excitatory_synapses",
    )
    rec_ie = _rec_syn(
        "rec_ie_synapses_per_branch",
        "rec_inhibitory_synapses_per_layer",
        "rec_inhibitory_synapses",
    )
    rec_ei = _rec_syn(
        "rec_ei_synapses_per_branch",
        "inh_rec_excitatory_synapses_per_layer",
        "inh_rec_excitatory_synapses",
        fallback=rec_ee,
    )
    rec_ii = _rec_syn(
        "rec_ii_synapses_per_branch",
        "inh_rec_inhibitory_synapses_per_layer",
        "inh_rec_inhibitory_synapses",
        fallback=rec_ie,
    )

    # --- Determine whether recurrence is active ---
    def _any_positive(val):
        if isinstance(val, list):
            return any(v > 0 for v in val)
        return val is not None and val > 0

    has_rec_synapses = any(_any_positive(v) for v in [rec_ee, rec_ie, rec_ei, rec_ii])
    # Backward compat: old configs use recurrent_ei.enabled: true to activate
    # recurrence.  New configs rely solely on connectivity.rec_* > 0.
    explicit_enabled = recurrent_cfg.get("enabled", None)
    if not has_rec_synapses and not explicit_enabled:
        return None

    if type in {"dendritic_mlp", "flat_shunting", "flat_additive"}:
        raise ValueError(
            f"Recurrent mode is not supported for architecture type '{type}'. "
            "Use 'EINet', 'dendritic_shunting', or 'dendritic_additive'."
        )

    excitatory_sizes = list(architecture.get("excitatory_layer_sizes", []))
    if not excitatory_sizes:
        raise ValueError(
            "Recurrent mode requires non-empty architecture.excitatory_layer_sizes"
        )

    inhibitory_sizes = list(architecture.get("inhibitory_layer_sizes", []))
    input_mode = int(transfer.get("input_mode", 0))

    # use_transfer: keep legacy EINet behavior by default (TransferLayer always on),
    # with backward-compatible explicit override via recurrent_ei.use_transfer.
    if "use_transfer" in recurrent_cfg:
        use_transfer = bool(recurrent_cfg["use_transfer"])
    else:
        use_transfer = True

    # Preserve legacy EINet type aliases for shunting/additive behavior.
    use_shunting = bool(morphology.get("use_shunting", True))
    if type == "dendritic_shunting":
        use_shunting = True
    elif type == "dendritic_additive":
        use_shunting = False

    # Per-layer recurrence: infer from synapse counts, backward compat with explicit flag
    explicit_recurrent_layers = recurrent_cfg.get(
        "recurrent_layers", recurrent_cfg.get("recurrent_per_layer", None)
    )

    dt_value = recurrent_cfg.get("dt", 1.0)

    # transfer_inhibitory_mode: new location first, old fallback
    transfer_inhibitory_mode = str(
        transfer.get(
            "inhibitory_mode",
            recurrent_cfg.get("transfer_inhibitory_mode", "first"),
        )
    ).lower()
    if transfer_inhibitory_mode not in {"none", "first", "all"}:
        raise ValueError(
            "transfer.inhibitory_mode must be one of ('none', 'first', 'all')"
        )

    layers: list[EILayerConfig] = []
    for layer_idx, n_exc in enumerate(excitatory_sizes):
        n_inh = int(_layer_value(inhibitory_sizes, layer_idx, default=0))
        build_inhibitory_population = n_inh > 0 and input_mode != 1

        # Per-layer recurrence
        if explicit_recurrent_layers is not None:
            rec_enabled = bool(
                _layer_value(explicit_recurrent_layers, layer_idx, default=True)
            )
        else:
            layer_rec_total = sum(
                int(_layer_value(v, layer_idx, default=0))
                for v in [rec_ee, rec_ie, rec_ei, rec_ii]
            )
            rec_enabled = layer_rec_total > 0

        dt = float(_layer_value(dt_value, layer_idx, default=1.0))

        # Per-layer tau_base / tau_ratio
        tau_base = float(
            _layer_value(recurrent_cfg.get("tau_base", 50.0), layer_idx, default=50.0)
        )
        tau_ratio = float(
            _layer_value(recurrent_cfg.get("tau_ratio", 3.0), layer_idx, default=3.0)
        )

        exc_cfg = PopulationConfig(
            n_neurons=int(n_exc),
            branch_factors=list(
                architecture.get("excitatory_branch_factors", [1]) or [1]
            ),
            ff_excitatory_synapses=int(
                _layer_value(
                    connectivity.get("ee_synapses_per_branch_per_layer", []),
                    layer_idx,
                    default=0,
                )
            ),
            ff_inhibitory_synapses=int(
                _layer_value(
                    connectivity.get("ie_synapses_per_branch_per_layer", []),
                    layer_idx,
                    default=0,
                )
            ),
            rec_excitatory_synapses=int(_layer_value(rec_ee, layer_idx, default=0)),
            rec_inhibitory_synapses=int(_layer_value(rec_ie, layer_idx, default=0)),
            topk_init_method=str(sparsity.get("init_method", "xavier_normal")),
            topk_noise_level=float(sparsity.get("noise_level", 0.0)),
            topk_type=_normalize_sparsity_type(sparsity.get("type", "standard")),
            topk_weight_norm_order=sparsity.get("weight_norm_order"),
            topk_gamma=float(sparsity.get("gamma", 1.0)),
            topk_temperature=float(sparsity.get("temperature", 0.0)),
            topk_ultrafast=bool(sparsity.get("ultrafast", False)),
            topk_strategy=str(sparsity.get("gradient_scaling", "none")),
            weight_transform=str(morphology.get("weight_transform", "softplus")),
            use_shunting=use_shunting,
            use_additive_normalization=bool(
                morphology.get("use_additive_normalization", False)
            ),
            allow_self_recurrence=bool(morphology.get("allow_self_recurrence", True)),
            reactivate=bool(reactivation.get("enabled", True)),
            reactivation_type=str(reactivation.get("type", "param_tanh")),
            reactivation_init_m=float(reactivation.get("init_m", 1.5)),
            reactivation_init_b=float(reactivation.get("init_b", 0.5)),
            reactivation_init_policy=str(reactivation.get("init_policy", "analytical")),
            reactivation_occupancy_quantile_low=reactivation.get(
                "occupancy_quantile_low"
            ),
            reactivation_occupancy_quantile_high=reactivation.get(
                "occupancy_quantile_high"
            ),
            reactivation_occupancy_target_low=reactivation.get("occupancy_target_low"),
            reactivation_occupancy_target_high=reactivation.get(
                "occupancy_target_high"
            ),
            reactivation_calibration_min_quantile_width=float(
                reactivation.get("calibration_min_quantile_width", 1e-3)
            ),
            reactivation_calibration_max_m=float(
                reactivation.get("calibration_max_m", 50.0)
            ),
            reactivation_calibration_revert_on_invalid=bool(
                reactivation.get("calibration_revert_on_invalid", True)
            ),
            reactivation_sigma_aware_k=float(reactivation.get("sigma_aware_k", 0.25)),
            reactivation_strategy=str(reactivation.get("gradient_scaling", "none")),
            blocklinear_strategy=str(blocklinear.get("gradient_scaling", "none")),
            level_taus=list(recurrent_cfg.get("level_taus", [])),
            tau_base=tau_base,
            tau_ratio=tau_ratio,
            learnable_tau=bool(recurrent_cfg.get("learnable_tau", False)),
            epsilon=float(recurrent_cfg.get("epsilon", 1e-8)),
            somatic_synapses=bool(morphology.get("somatic_synapses", False)),
            dbl_init_method=str(
                morphology.get("dbl_init_method", "analytical_expectation")
            ),
            efficient_blocklinear=bool(blocklinear.get("efficient", False)),
            print_hooks=bool(implementation.get("print_hooks", False)),
            excitatory_target_density=float(
                deepst.get("excitatory_target_density", 0.1)
            ),
            inhibitory_target_density=float(
                deepst.get("inhibitory_target_density", 0.1)
            ),
            use_noise=bool(deepst.get("use_noise", True)),
            sigma=float(deepst.get("sigma", 0.05)),
            rewiring_mode=str(deepst.get("rewiring_mode", "global")),
            rewire_frequency=int(deepst.get("rewire_frequency", 1)),
            synapses_per_branch=deepst.get("synapses_per_branch"),
            freeze_excitatory_connectivity=bool(
                deepst.get("freeze_excitatory_connectivity", False)
            ),
            freeze_inhibitory_connectivity=bool(
                deepst.get("freeze_inhibitory_connectivity", False)
            ),
            init_method=str(deepst.get("init_method", "xavier_normal")),
            weight_threshold=float(deepst.get("weight_threshold", 1e-6)),
            adaptive_initialization=bool(
                implementation.get("adaptive_initialization", False)
            ),
        )

        inh_cfg = None
        if build_inhibitory_population:
            inh_cfg = PopulationConfig(
                n_neurons=n_inh,
                branch_factors=list(
                    architecture.get("inhibitory_branch_factors", [1]) or [1]
                ),
                ff_excitatory_synapses=int(
                    _layer_value(
                        connectivity.get("ei_synapses_per_branch_per_layer", []),
                        layer_idx,
                        default=0,
                    )
                ),
                ff_inhibitory_synapses=int(
                    _layer_value(
                        connectivity.get("ii_synapses_per_branch_per_layer", []),
                        layer_idx,
                        default=0,
                    )
                ),
                rec_excitatory_synapses=int(_layer_value(rec_ei, layer_idx, default=0)),
                rec_inhibitory_synapses=int(_layer_value(rec_ii, layer_idx, default=0)),
                topk_init_method=exc_cfg.topk_init_method,
                topk_noise_level=exc_cfg.topk_noise_level,
                topk_type=exc_cfg.topk_type,
                topk_weight_norm_order=exc_cfg.topk_weight_norm_order,
                topk_gamma=exc_cfg.topk_gamma,
                topk_temperature=exc_cfg.topk_temperature,
                topk_ultrafast=exc_cfg.topk_ultrafast,
                topk_strategy=exc_cfg.topk_strategy,
                weight_transform=exc_cfg.weight_transform,
                use_shunting=exc_cfg.use_shunting,
                use_additive_normalization=exc_cfg.use_additive_normalization,
                allow_self_recurrence=exc_cfg.allow_self_recurrence,
                reactivate=exc_cfg.reactivate,
                reactivation_type=exc_cfg.reactivation_type,
                reactivation_init_m=exc_cfg.reactivation_init_m,
                reactivation_init_b=exc_cfg.reactivation_init_b,
                reactivation_init_policy=exc_cfg.reactivation_init_policy,
                reactivation_sigma_aware_k=exc_cfg.reactivation_sigma_aware_k,
                reactivation_strategy=exc_cfg.reactivation_strategy,
                blocklinear_strategy=exc_cfg.blocklinear_strategy,
                level_taus=exc_cfg.level_taus,
                tau_base=exc_cfg.tau_base,
                tau_ratio=exc_cfg.tau_ratio,
                learnable_tau=exc_cfg.learnable_tau,
                epsilon=exc_cfg.epsilon,
                somatic_synapses=exc_cfg.somatic_synapses,
                dbl_init_method=exc_cfg.dbl_init_method,
                efficient_blocklinear=exc_cfg.efficient_blocklinear,
                print_hooks=exc_cfg.print_hooks,
                excitatory_target_density=exc_cfg.excitatory_target_density,
                inhibitory_target_density=exc_cfg.inhibitory_target_density,
                use_noise=exc_cfg.use_noise,
                sigma=exc_cfg.sigma,
                rewiring_mode=exc_cfg.rewiring_mode,
                rewire_frequency=exc_cfg.rewire_frequency,
                synapses_per_branch=exc_cfg.synapses_per_branch,
                freeze_excitatory_connectivity=exc_cfg.freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=exc_cfg.freeze_inhibitory_connectivity,
                init_method=exc_cfg.init_method,
                weight_threshold=exc_cfg.weight_threshold,
                adaptive_initialization=exc_cfg.adaptive_initialization,
            )

        direct_ff_inh = False
        if input_mode == 1:
            if transfer_inhibitory_mode == "all":
                direct_ff_inh = True
            elif transfer_inhibitory_mode == "first":
                direct_ff_inh = layer_idx == 0

        layers.append(
            EILayerConfig(
                excitatory=exc_cfg,
                inhibitory=inh_cfg,
                recurrent=rec_enabled,
                dt=dt,
                direct_ff_inhibitory_to_excitatory=direct_ff_inh,
            )
        )

    transfer_params = dict(transfer)
    transfer_params.update(_to_plain_mapping(recurrent_cfg.get("transfer", {})))
    transfer_params.update(_to_plain_mapping(recurrent_cfg.get("transfer_params", {})))

    # input_projection_dims: new location first, old fallback
    raw_ipd = architecture.get(
        "input_projection_dims",
        recurrent_cfg.get("input_projection_dims", []),
    )
    ipd = list(raw_ipd) if isinstance(raw_ipd, list) else []

    # store_routing: new location first, old fallback
    store_routing = bool(
        implementation.get("store_routing", recurrent_cfg.get("store_routing", False))
    )

    cfg = EINetworkConfig(
        layers=layers,
        input_dim=input_dim,
        use_transfer=use_transfer,
        transfer_params=transfer_params,
        input_projection_dims=ipd,
        output_mode=str(recurrent_cfg.get("output_mode", "last")),
        store_routing=store_routing,
    )
    return EINetwork(cfg)


def get_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int] = None,
    suffix_input_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory method for creating network architectures.

    Args:
        type: Architecture type name
        parameters: Either flat parameters dict or structured config
        input_dim: Optional input dimension (needed for EINet from config)
        suffix_input_dim: Optional suffix input dimension. Used by spatial
            dendritic cores (e.g., ``hierarchical_dendritic_tensor_map``) to
            add an output projection when the EINet output dim doesn't match
            the downstream decoder's expected input.

    Returns:
        Initialized network module
    """
    type = (type or "").lower()

    # ------------------------------------------------------------------
    # Unified E-I architecture (opt-in, isolated from legacy code paths)
    # ------------------------------------------------------------------
    if type in _UNIFIED_EI_TYPES:
        from dendritic_modeling.networks.architectures.recurrent import (
            EILayerConfig,
            EINetwork,
            EINetworkConfig,
            PopulationConfig,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating unified E-I architecture (type='{type}')"
            )

        raw_params = _to_plain_mapping(parameters)
        unified_cfg = raw_params.get("unified_ei", raw_params.get("ei_unified", {}))
        if not unified_cfg and "layers" in raw_params:
            # Allow passing EINetworkConfig-shaped dict directly.
            unified_cfg = raw_params

        if not isinstance(unified_cfg, dict):
            raise ValueError(
                "Unified E-I config must be a mapping, got "
                f"{unified_cfg.__class__.__name__}"
            )

        unified_cfg = dict(unified_cfg)
        unified_cfg["input_dim"] = input_dim

        # Resolve recurrent-type aliases to explicit population flags. When
        # `type` is `ei_unified` / `unified_ei` / `ei_net`, this is a no-op;
        # otherwise it sets use_shunting / use_additive_normalization / flat
        # branch_factors to match the alias. Explicit values in the YAML
        # are preserved unless they conflict with the alias, in which case
        # the alias wins (matching the BP `canonicalize_model_core_flags`
        # contract and producing a self-consistent config).
        alias_overrides = _UNIFIED_EI_ALIAS_OVERRIDES.get(type, {})

        layers_cfg = []
        for layer in unified_cfg.get("layers", []):
            if isinstance(layer, EILayerConfig):
                layers_cfg.append(layer)
                continue
            if not isinstance(layer, dict):
                raise ValueError(
                    "Each unified E-I layer config must be a dict or EILayerConfig"
                )
            layer = dict(layer)
            exc = layer.get("excitatory")
            inh = layer.get("inhibitory")
            if isinstance(exc, dict):
                if alias_overrides:
                    exc = dict(exc)
                    for k, v in alias_overrides.items():
                        # `branch_factors` override only applies when the
                        # config did not specify its own tree (otherwise we
                        # would silently erase a user-provided hierarchy).
                        if k == "branch_factors" and exc.get("branch_factors"):
                            continue
                        exc[k] = v
                # Warn about the additive-init-bug trap: `use_shunting=False`
                # combined with the analytical init policy hits the buggy
                # `E_Vinf = E_g_exc - E_g_inh + 0.5` branch on non-leaf
                # layers and places the gate center above the useful
                # voltage range. Users should switch to the
                # `rnn_dendritic_additive` alias (which sets a safe
                # fixed-init) or explicitly set `reactivation_init_policy`
                # to "fixed" with a sensible `reactivation_init_b`.
                if not bool(exc.get("use_shunting", True)):
                    policy = str(
                        exc.get("reactivation_init_policy", "analytical")
                    ).lower()
                    if policy in {
                        "analytical",
                        "auto",
                        "quantile",
                        "occupancy_quantile",
                    }:
                        import warnings

                        warnings.warn(
                            "Additive-integration ablation with "
                            f"reactivation_init_policy='{policy}' triggers a known "
                            "non-leaf gate-center bug (E[V_inf] ~ 1.02). Prefer "
                            "core.type='rnn_dendritic_additive' (or 'fixed' policy "
                            "with b=0.0) for a correctly-initialised additive path.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                layer["excitatory"] = PopulationConfig(**exc)
            if isinstance(inh, dict):
                layer["inhibitory"] = PopulationConfig(**inh)
            layers_cfg.append(EILayerConfig(**layer))

        if not layers_cfg:
            raise ValueError(
                "Unified E-I architecture requires at least one layer in unified_ei.layers"
            )

        unified_cfg["layers"] = layers_cfg
        return EINetwork(EINetworkConfig(**unified_cfg))

    # ------------------------------------------------------------------
    # Baseline RNN models (GRU, LSTM, Vanilla RNN)
    # ------------------------------------------------------------------
    if type in _BASELINE_RNN_TYPES:
        from dendritic_modeling.networks.architectures.recurrent import (
            BaselineRNN,
            BaselineRNNConfig,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating baseline RNN (type='{type}')"
            )

        raw_params = _to_plain_mapping(parameters)
        baseline_cfg = raw_params.get("baseline_rnn", raw_params)
        if not isinstance(baseline_cfg, dict):
            baseline_cfg = {}
        baseline_cfg = dict(baseline_cfg)
        baseline_cfg["cell_type"] = _BASELINE_RNN_TYPES[type]
        baseline_cfg["input_dim"] = input_dim
        return BaselineRNN(BaselineRNNConfig(**baseline_cfg))

    # ------------------------------------------------------------------
    # Legacy recurrent type aliases (removed -- use unified E-I instead)
    # ------------------------------------------------------------------
    if type in ("dendritic_rnn", "dendrnn"):
        raise ValueError(
            f"Architecture type '{type}' has been removed. "
            f"Use 'ei_unified' (or 'unified_ei', 'ei_net') with the new "
            f"EINetworkConfig format instead."
        )

    # ------------------------------------------------------------------
    # Spatial dendritic cores (flatten → EINet → project)
    # ------------------------------------------------------------------
    if type in _SPATIAL_DENDRITIC_TYPES:
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
            HierarchicalDendriticTensorMap,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating spatial dendritic core (type='{type}')"
            )

        return HierarchicalDendriticTensorMap(
            config=parameters,
            input_dim=input_dim,
            suffix_input_dim=suffix_input_dim,
        )

    # ------------------------------------------------------------------
    # Spatial dendritic conv (unfold → shared DendriNet → spatial output)
    # ------------------------------------------------------------------
    if type in _SPATIAL_DENDRITIC_CONV_TYPES:
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
            HierarchicalDendriticConv,
        )

        raw_params = _to_plain_mapping(parameters)
        spatial_cfg = _to_plain_mapping(raw_params.get("spatial", {}))
        kernel_size = spatial_cfg.get("kernel_size", 3)
        stride = spatial_cfg.get("stride", 1)
        padding = spatial_cfg.get("padding", 0)

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating spatial dendritic conv (type='{type}')"
            )

        # input_dim here is the number of input channels (C_in).
        # For spatial cores, input_dim = prefix_output_spec.channels.
        return HierarchicalDendriticConv(
            config=parameters,
            in_channels=input_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    # ------------------------------------------------------------------
    # EINet variants (require structured config)
    # ------------------------------------------------------------------
    if type in _EINET_TYPES:
        if not (
            isinstance(parameters, DictConfig)
            or (
                isinstance(parameters, dict)
                and "architecture" in parameters
                and "connectivity" in parameters
            )
        ):
            raise ValueError(
                "EINet now requires structured config with 'architecture', 'connectivity', etc. "
                "sections. Flat parameters are no longer supported."
            )
        if input_dim is None:
            raise ValueError(
                "input_dim required when passing structured config to EINet"
            )

        raw_params = _to_plain_mapping(parameters)
        recurrent_einet = _build_recurrent_einet_from_structured_config(
            type=type, raw_params=raw_params, input_dim=input_dim
        )
        if recurrent_einet is not None:
            return recurrent_einet

        # Derive settings from type name using string matching
        synapse_mode: Optional[str] = None
        use_shunting: Optional[bool] = None
        weight_transform: Optional[str] = None
        flatten_dendrites: bool = False

        if "shunting" in type:
            use_shunting = True
        elif "additive" in type:
            use_shunting = False

        if "flat" in type:
            flatten_dendrites = True

        if "mlp" in type:
            synapse_mode = "mlp"
            weight_transform = "identity"
            use_shunting = False

        params_for_build = _to_plain_mapping(parameters)
        params_for_build.setdefault("type", type)

        return ConfigurableEINetwork(
            config=params_for_build,
            input_dim=input_dim,
            synapse_mode=synapse_mode,
            use_shunting=use_shunting,
            weight_transform=weight_transform,
            flatten_dendrites=flatten_dendrites,
        )

    # ------------------------------------------------------------------
    # Point-neuron MLP baseline (parameter-matched, structured config)
    # ------------------------------------------------------------------
    if type in ["point_mlp", "mlp"] and (
        isinstance(parameters, DictConfig)
        or (
            isinstance(parameters, dict)
            and "architecture" in parameters
            and "connectivity" in parameters
        )
    ):
        if input_dim is None:
            raise ValueError("input_dim required when passing structured config to mlp")

        from dendritic_modeling.scripts.script_utils.config_utils import (
            prepare_ei_network_params,
        )

        params = prepare_ei_network_params(parameters, input_dim)
        return PointMatchedParamMLP(**params)

    # ------------------------------------------------------------------
    # MR-style structured-config MLP baselines
    # ------------------------------------------------------------------
    if type in ["matchedtotalparammlp", "total_param_mlp"]:
        if type == "matchedtotalparammlp":
            warnings.warn(
                "Architecture type 'matchedtotalparammlp' is deprecated; use "
                "'total_param_mlp' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return MatchedTotalParamMLP(parameters, input_dim)

    if type in ["matchedactiveparammlp", "active_param_mlp"]:
        if type == "matchedactiveparammlp":
            warnings.warn(
                "Architecture type 'matchedactiveparammlp' is deprecated; use "
                "'active_param_mlp' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return MatchedActiveParamMLP(parameters, input_dim)

    if type == "ss_mlp":
        return SparseStructuredMLP(parameters, input_dim)

    if type == "ss_mlp_flat":
        return SparseStructuredMLP(parameters, input_dim, flatten_dendrites=True)

    # ------------------------------------------------------------------
    # All other architectures expect flat parameters
    # ------------------------------------------------------------------
    if isinstance(parameters, DictConfig):
        raise ValueError(
            f"Architecture {type} expects flat parameters, not structured config"
        )

    if type == "alexnet":
        return AlexNet(**parameters)
    elif type in ["cnn", "cnn_down", "cnndownsample", "cnn_downsample"]:
        return CNNDownsample(**parameters)
    elif type in ["cnn_up", "cnnupsample", "cnn_upsample"]:
        return CNNUpsample(**parameters)
    elif type == "mlp":
        return MLP(**parameters)
    elif type == "effectivematchedparammlp":
        return EffectiveMatchedParamMLP(**parameters)
    elif type == "mlpautoencoder":
        return MLPAutoencoder(**parameters)
    elif type == "cnnautoencoder":
        return CNNAutoencoder(**parameters)
    elif type == "mlpvariationalautoencoder":
        return MLPVariationalAutoencoder(**parameters)
    elif type == "cnnvariationalautoencoder":
        return CNNVariationalAutoencoder(**parameters)
    elif type in ["pathway_router", "router"]:
        return PathwayRouter(**parameters)
    elif type in ["identity", "none", None]:
        return Identity(**parameters)
    else:
        raise ValueError(f"Invalid architecture type: {type}")
