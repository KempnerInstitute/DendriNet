"""
Excitation-Inhibition Network.

This module implements the ExcitationInhibitionNetwork which uses separate
excitatory and inhibitory pathways with dendritic computation.
"""

import logging
from copy import deepcopy
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch._dynamo import disable as _dynamo_disable

from dendritic_modeling.config.conversion import (
    has_enabled_synapse_types,
    to_plain_dict,
)
from dendritic_modeling.config.inhibitory_wiring import (
    check_inhibitory_wiring,
    check_retired_inhibitory_routing,
)
from dendritic_modeling.config.legacy import normalize_transfer_config
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.utils.weight_transforms import (
    NONNEGATIVE_TRANSFER_ACTIVATIONS,
    POSITIVE_WEIGHT_TRANSFORMS,
)

logger = logging.getLogger(__name__)


def _extend_synapse_schedule(
    values: list[int],
    *,
    num_layers: int,
    required: bool,
    empty_error: str,
    default_value: int = 0,
) -> list[int]:
    """Extend a per-layer synapse schedule without inventing nonzero defaults."""

    schedule = list(values)
    while len(schedule) < num_layers:
        if schedule:
            schedule.append(schedule[-1])
        elif required:
            raise ValueError(empty_error)
        else:
            schedule.append(default_value)
    return schedule


def _prepare_ei_synapse_schedules(
    *,
    ee_synapses_per_branch_per_layer: list[int],
    ei_synapses_per_branch_per_layer: list[int],
    ie_synapses_per_branch_per_layer: list[int],
    ii_synapses_per_branch_per_layer: list[int],
    num_layers: int,
    has_inhibitory_neurons: bool,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Prepare per-layer E/I synapse schedules with legacy required fields."""

    ee_synapses = _extend_synapse_schedule(
        ee_synapses_per_branch_per_layer,
        num_layers=num_layers,
        required=True,
        empty_error="ee_synapses_per_branch_per_layer cannot be empty",
    )
    ei_synapses = _extend_synapse_schedule(
        ei_synapses_per_branch_per_layer,
        num_layers=num_layers,
        required=has_inhibitory_neurons,
        empty_error=(
            "ei_synapses_per_branch_per_layer cannot be empty when "
            "inhibitory neurons exist"
        ),
    )
    ie_synapses = _extend_synapse_schedule(
        ie_synapses_per_branch_per_layer,
        num_layers=num_layers,
        required=has_inhibitory_neurons,
        empty_error=(
            "ie_synapses_per_branch_per_layer cannot be empty when "
            "inhibitory neurons exist"
        ),
    )
    ii_synapses = _extend_synapse_schedule(
        ii_synapses_per_branch_per_layer,
        num_layers=num_layers,
        required=has_inhibitory_neurons,
        empty_error=(
            "ii_synapses_per_branch_per_layer cannot be empty when "
            "inhibitory neurons exist"
        ),
    )
    return ee_synapses, ei_synapses, ie_synapses, ii_synapses


def _prepare_ei_layer_size_sequences(
    *,
    input_dim: int,
    excitatory_layer_sizes: list[int],
    inhibitory_layer_sizes: list[int],
) -> tuple[list[int], list[int | None]]:
    """Prepare per-layer E/I sizes including the input sentinel."""

    prepared_excitatory_sizes = deepcopy(excitatory_layer_sizes)
    prepared_inhibitory_sizes: list[int | None] = deepcopy(inhibitory_layer_sizes)
    prepared_excitatory_sizes.insert(0, input_dim)
    prepared_inhibitory_sizes.insert(0, None)
    return prepared_excitatory_sizes, prepared_inhibitory_sizes


def _resolve_ei_layer_build_spec(
    *,
    layer_idx: int,
    excitatory_layer_sizes: list[int],
    inhibitory_layer_sizes: list[int | None],
    excitatory_input_dim: int,
    inhibitory_input_dim: int | None,
    ei_synapses_per_branch,
    ii_synapses_per_branch,
) -> dict[str, Any]:
    """Resolve dimensions and inhibitory-population build flag for one EI layer."""

    inhibitory_population_has_drive = (
        int(ei_synapses_per_branch or 0) > 0 or int(ii_synapses_per_branch or 0) > 0
    )
    n_inhibitory_cells = (
        inhibitory_layer_sizes[layer_idx + 1]
        if layer_idx + 1 < len(inhibitory_layer_sizes)
        else None
    )
    build_inhibitory_cells = (
        n_inhibitory_cells is not None
        and n_inhibitory_cells > 0
        and inhibitory_population_has_drive
    )

    if layer_idx == 0:
        layer_excitatory_input_dim = excitatory_input_dim
        layer_inhibitory_input_dim = inhibitory_input_dim
    else:
        layer_excitatory_input_dim = excitatory_layer_sizes[layer_idx]
        layer_inhibitory_input_dim = (
            inhibitory_layer_sizes[layer_idx]
            if layer_idx < len(inhibitory_layer_sizes)
            else None
        )

    return {
        "n_excitatory_cells": excitatory_layer_sizes[layer_idx + 1],
        "n_inhibitory_cells": n_inhibitory_cells,
        "excitatory_input_dim": layer_excitatory_input_dim,
        "inhibitory_input_dim": layer_inhibitory_input_dim,
        "build_inhibitory_cells": build_inhibitory_cells,
    }


_RECURRENT_SYNAPSE_KEYS = (
    "rec_ee_synapses_per_branch",
    "rec_ie_synapses_per_branch",
    "rec_ei_synapses_per_branch",
    "rec_ii_synapses_per_branch",
)


def _consume_recurrent_only_options(options: dict[str, Any]) -> None:
    """Remove recurrent-EINet options, failing when a value would be ignored.

    The structured config passes these fields to every EINet. The feedforward
    network has no recurrent state, so they must keep their inert defaults;
    they are removed here so they cannot reach the branch layers as unknown
    options.
    """
    unused = []
    projection_dims = options.pop("input_projection_dims", None)
    if projection_dims:
        unused.append(f"architecture.input_projection_dims={list(projection_dims)!r}")
    if bool(options.pop("store_routing", False)):
        unused.append("implementation.store_routing=true")
    if not bool(options.pop("allow_self_recurrence", True)):
        unused.append("morphology.allow_self_recurrence=false")
    for key in _RECURRENT_SYNAPSE_KEYS:
        value = options.pop(key, None)
        values = value if isinstance(value, (list, tuple)) else [value]
        if any(item is not None and float(item) > 0 for item in values):
            unused.append(f"connectivity.{key}={value!r}")
    synapse_types = options.pop("synapse_types", None) or {}
    if has_enabled_synapse_types(synapse_types):
        unused.append("synapse_types.enabled")
    dynamics = to_plain_dict(options.pop("dynamics", None) or {})
    if str(dynamics.get("mode", dynamics.get("dynamics_mode", "rate"))).lower() == (
        "spike"
    ):
        unused.append("dynamics.mode=spike")
    for key in ("soma_feedback_enabled", "dendritic_spikes_enabled"):
        if bool(dynamics.get(key, False)):
            unused.append(f"dynamics.{key}=true")
    for key in ("soma_feedback", "dendritic_spikes"):
        block = to_plain_dict(dynamics.get(key) or {})
        if bool(block.get("enabled", False)):
            unused.append(f"dynamics.{key}.enabled=true")
    if unused:
        raise ValueError(
            "Feedforward EINet does not apply "
            + ", ".join(unused)
            + "; these options are used only by recurrent EINet "
            "(connectivity.rec_* > 0 or stateful dynamics). Remove them for "
            "feedforward models."
        )


class ExcitationInhibitionNetwork(BaseNetwork):
    """Excitation-Inhibition network for dendritic modeling.

    Args:
        input_dim: Input dimension
        transfer_params: Parameters for the transfer function.
        excitatory_layer_sizes: List of integers specifying the number of excitatory cells in each layer.
        inhibitory_layer_sizes: List of integers specifying the number of inhibitory cells in each layer.
        excitatory_branch_factors: List of integers specifying the number of branches for excitatory cells.
        inhibitory_branch_factors: List of integers specifying the number of branches for inhibitory cells.
        reactivate: Whether to reactivate cells.
        reactivation_init_m: Initial m parameter for reactivation.
        reactivation_init_b: Initial b parameter for reactivation.
        dbl_init_method: Method for initializing dendritic branch layers.
        somatic_synapses: Whether to use somatic synapses.
        ee_synapses_per_branch_per_layer: List of integers specifying the number of excitatory-excitatory synapses per branch per layer.
        ei_synapses_per_branch_per_layer: List of integers specifying the number of excitatory-inhibitory synapses per branch per layer.
        ie_synapses_per_branch_per_layer: List of integers specifying the number of inhibitory-excitatory synapses per branch per layer.
        ii_synapses_per_branch_per_layer: List of integers specifying the number of inhibitory-inhibitory synapses per branch per layer.
        use_shunting: Whether to use shunting inhibition.
        reactivation_type: Type of reactivation function.
        reactivation_strategy: Strategy for reactivation.
        blocklinear_strategy: Strategy for blocklinear computation.
        topk_init_method: Method for initializing topk.
        topk_strategy: Strategy for topk.
        topk_type: Type of topk.
        print_hooks: Whether to print hooks.
        excitatory_target_density: Target density for excitatory cells.
        inhibitory_target_density: Target density for inhibitory cells.
        use_noise: Whether to use noise.
        sigma: Standard deviation for noise.
        rewiring_mode: Mode for rewiring.
        rewire_frequency: Frequency for rewiring.
        synapses_per_branch: Number of synapses per branch.
        efficient_blocklinear: Whether to use efficient blocklinear.
        freeze_excitatory_connectivity: Whether to freeze excitatory connectivity.
        freeze_inhibitory_connectivity: Whether to freeze inhibitory connectivity.
        init_method: Method for initializing weights.
        weight_threshold: Threshold for weight pruning.
        adaptive_initialization: Whether to use adaptive initialization.
    """

    def __init__(
        self,
        input_dim: int,
        transfer_params: dict[str, Any],
        excitatory_layer_sizes: list[int],
        inhibitory_layer_sizes: list[int],
        excitatory_branch_factors: list[int],
        inhibitory_branch_factors: list[int],
        reactivate: bool = True,
        reactivation_init_m: float = 1.5,
        reactivation_init_b: float = 0.5,
        reactivation_init_policy: str = "analytical",
        reactivation_sigma_aware_k: float = 0.25,
        dbl_init_method: str = "analytical_expectation",
        somatic_synapses: bool = True,
        ee_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ei_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ie_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ii_synapses_per_branch_per_layer: Optional[list[int]] = None,
        inhibitory_network_type: str = "dendritic",
        use_shunting: bool = True,
        synapse_mode: str = "ei",
        reactivation_type: str = "param_tanh",
        reactivation_soma_type: str | None = None,
        reactivation_soma_init_m: float = 1.0,
        reactivation_soma_init_b: float = 0.0,
        reactivation_strategy: str = "none",
        blocklinear_strategy: str = "none",
        topk_init_method: str = "xavier_normal",
        topk_noise_level: float = 0.0,
        topk_strategy: str = "none",
        topk_type: str = "standard",
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        weight_transform="exp",  # Weight transformation for positive weights
        print_hooks: bool = False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        efficient_blocklinear: bool = False,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        adaptive_initialization=True,
        adaptive_initialization_policy="preserve_shunting_center",
        adaptive_target_conductance=5.0,
        initial_child_conductance=1.0,
        initial_conductance_scale=1.0,
        initial_child_conductance_log_sd=0.0,
        compile_forward: bool = False,
        structured_connectivity=None,
        **kwargs,
    ):
        super().__init__()
        _consume_recurrent_only_options(kwargs)
        self.input_dim = input_dim
        self.transfer_params = normalize_transfer_config(transfer_params)
        self.output_dim = excitatory_layer_sizes[-1]
        self.excitatory_layer_sizes = excitatory_layer_sizes
        self.inhibitory_layer_sizes = inhibitory_layer_sizes
        self.excitatory_branch_factors = excitatory_branch_factors
        self.inhibitory_branch_factors = inhibitory_branch_factors
        self.reactivate = reactivate
        self.reactivation_init_m = reactivation_init_m
        self.reactivation_init_b = reactivation_init_b
        self.reactivation_init_policy = reactivation_init_policy
        self.reactivation_sigma_aware_k = reactivation_sigma_aware_k
        self.dbl_init_method = dbl_init_method
        self.somatic_synapses = somatic_synapses
        self.inhibitory_network_type = inhibitory_network_type
        self.ee_synapses_per_branch_per_layer = (
            ee_synapses_per_branch_per_layer if ee_synapses_per_branch_per_layer else []
        )
        self.ei_synapses_per_branch_per_layer = (
            ei_synapses_per_branch_per_layer if ei_synapses_per_branch_per_layer else []
        )
        self.ie_synapses_per_branch_per_layer = (
            ie_synapses_per_branch_per_layer if ie_synapses_per_branch_per_layer else []
        )
        self.ii_synapses_per_branch_per_layer = (
            ii_synapses_per_branch_per_layer if ii_synapses_per_branch_per_layer else []
        )
        self.use_shunting = use_shunting
        self.synapse_mode = synapse_mode
        self.reactivation_type = reactivation_type
        self.reactivation_soma_type = reactivation_soma_type
        self.reactivation_soma_init_m = reactivation_soma_init_m
        self.reactivation_soma_init_b = reactivation_soma_init_b
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.topk_init_method = topk_init_method
        self.topk_strategy = topk_strategy
        self.topk_type = topk_type
        self.topk_weight_norm_order = topk_weight_norm_order
        self.topk_gamma = topk_gamma
        self.weight_transform = weight_transform
        self.topk_noise_level = topk_noise_level
        self.print_hooks = print_hooks
        self.excitatory_target_density = excitatory_target_density
        self.inhibitory_target_density = inhibitory_target_density
        self.use_noise = use_noise
        self.sigma = sigma
        self.rewiring_mode = rewiring_mode
        self.rewire_frequency = rewire_frequency
        self.efficient_blocklinear = efficient_blocklinear
        self.synapses_per_branch = synapses_per_branch
        self.freeze_excitatory_connectivity = freeze_excitatory_connectivity
        self.freeze_inhibitory_connectivity = freeze_inhibitory_connectivity
        self.init_method = init_method
        self.weight_threshold = weight_threshold
        self.adaptive_initialization = adaptive_initialization
        self.adaptive_initialization_policy = adaptive_initialization_policy
        self.adaptive_target_conductance = adaptive_target_conductance
        self.initial_child_conductance = initial_child_conductance
        self.initial_conductance_scale = initial_conductance_scale
        self.initial_child_conductance_log_sd = initial_child_conductance_log_sd
        self.compile_forward = compile_forward
        self.structured_connectivity = structured_connectivity

        # Records whether the nonnegative conductance-input validation has run.
        # The validation itself is performed on every forward pass for shunting
        # networks with positive weights, because later batches can differ from
        # the first one.
        self._nonneg_input_checked = False

        # Set up transfer function
        self.setup_transfer_function()

        # Set up network layers
        self.setup_network_layers(**kwargs)

    def setup_transfer_function(self):
        transfer_params = deepcopy(self.transfer_params)
        output_activation = transfer_params.get("output_activation", None)
        output_activation_norm = (
            None if output_activation is None else str(output_activation).lower()
        )
        weight_transform_norm = str(self.weight_transform).lower()

        # Positive conductance weights should not receive explicitly signed
        # presynaptic drive. We keep the runtime behavior config-driven rather
        # than silently mutating the transfer function here, so unrelated
        # backprop experiments are not changed behind the scenes. Configs that
        # require nonnegative first-layer drive set
        # transfer.output_activation explicitly.
        if weight_transform_norm in POSITIVE_WEIGHT_TRANSFORMS:
            if (
                output_activation_norm not in (None, "none")
                and output_activation_norm not in NONNEGATIVE_TRANSFER_ACTIVATIONS
            ):
                raise ValueError(
                    "Positive-conductance EI network requires nonnegative transfer "
                    f"output, but got output_activation={output_activation!r}."
                )

        self.transfer_params = transfer_params
        self.transfer_fn = TransferLayer(
            input_dim=self.input_dim, transfer_params=self.transfer_params
        )

    def _make_ei_layer(
        self,
        *,
        layer_idx: int,
        layer_spec: dict[str, Any],
        ee_synapses_per_branch: int,
        ei_synapses_per_branch: int,
        ie_synapses_per_branch: int,
        ii_synapses_per_branch: int,
        extra_kwargs: dict[str, Any],
    ) -> ExcitationInhibitionLayer:
        """Build one EI layer from resolved dimensions and per-layer schedules."""

        extra_kwargs = dict(extra_kwargs)
        if extra_kwargs.get("initialization_seed") is not None:
            # Keyed initialization streams must differ between network layers.
            base = extra_kwargs.get("initialization_namespace") or "einet"
            extra_kwargs["initialization_namespace"] = f"{base}.layer.{layer_idx}"
        return ExcitationInhibitionLayer(
            n_excitatory_cells=layer_spec["n_excitatory_cells"],
            n_inhibitory_cells=layer_spec["n_inhibitory_cells"],
            excitatory_branch_factors=self.excitatory_branch_factors,
            inhibitory_branch_factors=self.inhibitory_branch_factors,
            excitatory_input_dim=layer_spec["excitatory_input_dim"],
            ee_synapses_per_branch=ee_synapses_per_branch,
            ei_synapses_per_branch=ei_synapses_per_branch,
            build_inhibitory_cells=layer_spec["build_inhibitory_cells"],
            inhibitory_network_type=self.inhibitory_network_type,
            inhibitory_input_dim=layer_spec["inhibitory_input_dim"],
            ie_synapses_per_branch=ie_synapses_per_branch,
            ii_synapses_per_branch=ii_synapses_per_branch,
            reactivate=self.reactivate,
            reactivation_init_m=self.reactivation_init_m,
            reactivation_init_b=self.reactivation_init_b,
            reactivation_init_policy=self.reactivation_init_policy,
            reactivation_sigma_aware_k=self.reactivation_sigma_aware_k,
            dbl_init_method=self.dbl_init_method,
            somatic_synapses=self.somatic_synapses,
            topk_init_method=self.topk_init_method,
            use_shunting=self.use_shunting,
            synapse_mode=self.synapse_mode,
            reactivation_strategy=self.reactivation_strategy,
            blocklinear_strategy=self.blocklinear_strategy,
            reactivation_type=self.reactivation_type,
            reactivation_soma_type=self.reactivation_soma_type,
            reactivation_soma_init_m=self.reactivation_soma_init_m,
            reactivation_soma_init_b=self.reactivation_soma_init_b,
            topk_noise_level=self.topk_noise_level,
            topk_strategy=self.topk_strategy,
            topk_type=self.topk_type,
            topk_weight_norm_order=self.topk_weight_norm_order,
            topk_gamma=self.topk_gamma,
            weight_transform=self.weight_transform,
            print_hooks=self.print_hooks,
            excitatory_target_density=self.excitatory_target_density,
            inhibitory_target_density=self.inhibitory_target_density,
            use_noise=self.use_noise,
            sigma=self.sigma,
            rewiring_mode=self.rewiring_mode,
            rewire_frequency=self.rewire_frequency,
            synapses_per_branch=self.synapses_per_branch,
            freeze_excitatory_connectivity=self.freeze_excitatory_connectivity,
            freeze_inhibitory_connectivity=self.freeze_inhibitory_connectivity,
            init_method=self.init_method,
            weight_threshold=self.weight_threshold,
            efficient_blocklinear=self.efficient_blocklinear,
            adaptive_initialization=self.adaptive_initialization,
            adaptive_initialization_policy=self.adaptive_initialization_policy,
            adaptive_target_conductance=self.adaptive_target_conductance,
            initial_child_conductance=self.initial_child_conductance,
            initial_conductance_scale=self.initial_conductance_scale,
            initial_child_conductance_log_sd=self.initial_child_conductance_log_sd,
            structured_connectivity=self.structured_connectivity,
            structured_layer_idx=layer_idx,
            **extra_kwargs,
        )

    def _register_ei_layers(self, layers: list[ExcitationInhibitionLayer]) -> None:
        """Register EI layers and cached layer/branch config views."""

        self.layers = nn.ModuleList(layers)
        self.synapse_configs = tuple(layer.synapse_config for layer in layers)
        self.excitatory_synapse_configs = tuple(
            layer.excitatory_synapse_config for layer in layers
        )
        self.inhibitory_synapse_configs = tuple(
            layer.inhibitory_synapse_config for layer in layers
        )
        self.excitatory_branch_configs = tuple(
            tuple(layer.excitatory_branch_configs) for layer in layers
        )
        self.inhibitory_branch_configs = tuple(
            tuple(layer.inhibitory_branch_configs) for layer in layers
        )
        self.branch_configs = tuple(
            branch_config
            for layer in layers
            for branch_configs in (
                layer.excitatory_branch_configs,
                layer.inhibitory_branch_configs,
            )
            for branch_config in branch_configs
        )

    def setup_network_layers(self, **kwargs):
        """Set up the excitatory and inhibitory network layers"""
        # Create network layers using ExcitationInhibitionLayer
        layers = []

        check_inhibitory_wiring(
            {
                "excitatory_layer_sizes": self.excitatory_layer_sizes,
                "inhibitory_layer_sizes": self.inhibitory_layer_sizes,
            },
            {
                "ee_synapses_per_branch_per_layer": self.ee_synapses_per_branch_per_layer,
                "ei_synapses_per_branch_per_layer": self.ei_synapses_per_branch_per_layer,
                "ie_synapses_per_branch_per_layer": self.ie_synapses_per_branch_per_layer,
                "ii_synapses_per_branch_per_layer": self.ii_synapses_per_branch_per_layer,
            },
            has_transfer_stream=True,
        )
        excitatory_layer_sizes, inhibitory_layer_sizes = (
            _prepare_ei_layer_size_sequences(
                input_dim=self.input_dim,
                excitatory_layer_sizes=self.excitatory_layer_sizes,
                inhibitory_layer_sizes=self.inhibitory_layer_sizes,
            )
        )

        # Extend lists if needed - use exactly what's specified, no smart defaulting
        num_layers = len(excitatory_layer_sizes) - 1

        # Check if we have inhibitory neurons in the network
        has_inhibitory_neurons = any(
            size is not None and size > 0 for size in inhibitory_layer_sizes
        )
        ee_syn, ei_syn, ie_syn, ii_syn = _prepare_ei_synapse_schedules(
            ee_synapses_per_branch_per_layer=list(
                self.ee_synapses_per_branch_per_layer
            ),
            ei_synapses_per_branch_per_layer=list(
                self.ei_synapses_per_branch_per_layer
            ),
            ie_synapses_per_branch_per_layer=list(
                self.ie_synapses_per_branch_per_layer
            ),
            ii_synapses_per_branch_per_layer=list(
                self.ii_synapses_per_branch_per_layer
            ),
            num_layers=num_layers,
            has_inhibitory_neurons=has_inhibitory_neurons,
        )

        excitatory_input_dim = self.transfer_fn.excitatory_dim
        inhibitory_input_dim = self.transfer_fn.inhibitory_dim

        # Build each layer
        for i in range(num_layers):
            layer_spec = _resolve_ei_layer_build_spec(
                layer_idx=i,
                excitatory_layer_sizes=excitatory_layer_sizes,
                inhibitory_layer_sizes=inhibitory_layer_sizes,
                excitatory_input_dim=excitatory_input_dim,
                inhibitory_input_dim=inhibitory_input_dim,
                ei_synapses_per_branch=ei_syn[i],
                ii_synapses_per_branch=ii_syn[i],
            )

            layer = self._make_ei_layer(
                layer_idx=i,
                layer_spec=layer_spec,
                ee_synapses_per_branch=ee_syn[i],
                ei_synapses_per_branch=ei_syn[i],
                ie_synapses_per_branch=ie_syn[i],
                ii_synapses_per_branch=ii_syn[i],
                extra_kwargs=kwargs,
            )
            layers.append(layer)

        self._register_ei_layers(layers)

        # Optional torch.compile acceleration. Keep this opt-in because
        # Inductor caches can be fragile on shared filesystems and in tests.
        # Module.compile leaves dynamo's process-wide settings alone, and
        # copies and pickles of the module drop the compiled call.
        if self.compile_forward:
            self._reject_uncompilable_gradient_scaling()
            self.compile(dynamic=True)

    def _reject_uncompilable_gradient_scaling(self) -> None:
        """Refuse what ``compile_forward`` cannot compile before training starts.

        Inductor fails at the first training step on the conductance-scaled
        BlockLinear gradient hooks of shunting layers. The Top-K conductance
        scaling of shunting layers and the BlockLinear rule of additive layers
        compile and match the uncompiled gradients.
        """
        if any(
            isinstance(layer, DendriticBranchLayer)
            and layer.use_shunting
            and layer.blocklinear_strategy == "conductance_dynamic"
            for layer in self.modules()
        ):
            raise ValueError(
                "compile_forward does not support blocklinear gradient_scaling: "
                "conductance_dynamic in shunting layers (torch.compile fails on "
                "its gradient hooks at the first training step); set "
                "compile_forward: false"
            )

    def decay_weights(self, weight_decay, weight_boosting=False):
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            layer.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            layer.apply_rewiring()

    @_dynamo_disable
    def _check_nonneg_network_input(self, excitatory_x, inhibitory_x=None) -> None:
        """Runtime sanity check on the conductance-driving inputs.

        Positive-conductance shunting requires the *transfer-layer outputs* that
        feed the first dendritic layer to be non-negative. This allows configs
        to clamp a signed raw input via ``transfer.output_activation='relu'``
        while still rejecting silent signed conductance drive on every batch at the
        shunting boundary.
        """
        if not self.use_shunting:
            self._nonneg_input_checked = True
            return
        if self.weight_transform.lower() not in POSITIVE_WEIGHT_TRANSFORMS:
            self._nonneg_input_checked = True
            return

        # One host read for both streams (often the same tensor).
        streams = {
            id(tensor): tensor
            for tensor in (excitatory_x, inhibitory_x)
            if torch.is_tensor(tensor)
        }
        x_min = (
            float(torch.stack([t.detach().min() for t in streams.values()]).min())
            if streams
            else 0.0
        )
        if x_min < -1e-5:  # small tolerance for float-precision negatives
            raise ValueError(
                "ExcitationInhibitionNetwork received signed conductance-driving "
                "inputs after the transfer layer "
                f"(min={x_min:.4g}) while configured for shunting with "
                f"weight_transform={self.weight_transform!r}. A positive-"
                "conductance shunting network requires non-negative inputs "
                "(firing rates / conductances) so the shunting denominator "
                "(1 + g_exc + g_inh) stays bounded away from zero. Common "
                "cause: `data.processing.normalize: true` on an image dataset "
                "or a signed transfer output feeding the first dendritic "
                "layer.\n"
                "Fix options:\n"
                "  1. data.processing.normalize: false   "
                "(keep pixels in [0, 1])\n"
                '  2. model.core.transfer.output_activation: "relu"   '
                "(clamp transfer outputs before shunting)\n"
                '  3. model.core.type: "dendritic_additive"   '
                "(use additive morphology, which handles signed inputs)"
            )
        self._nonneg_input_checked = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transfer layer, then every E/I layer in order.

        The transfer layer returns the excitatory and inhibitory input streams.
        Each layer returns its excitatory output and its inhibitory cells'
        output, which feed the next layer (see ``config.inhibitory_wiring``).
        """
        excitatory_x, inhibitory_x = self.transfer_fn(x)
        self._check_nonneg_network_input(excitatory_x, inhibitory_x)

        # Optimized: Use direct iteration over ModuleList
        for layer in self.layers:
            excitatory_x, inhibitory_x = layer(excitatory_x, inhibitory_x)

        return excitatory_x

    def forward_until(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Excitatory output of layer ``layer_index``, computed as ``forward`` does.

        The transfer layer and the layers before it run exactly as in
        ``forward``, passing the excitatory and inhibitory streams; later
        layers do not run. ``forward_until(x, len(layers) - 1)`` equals
        ``forward(x)``.
        """
        if not 0 <= int(layer_index) < len(self.layers):
            raise IndexError(
                f"layer_index {layer_index} outside the {len(self.layers)} layers"
            )
        excitatory_x, inhibitory_x = self.transfer_fn(x)
        self._check_nonneg_network_input(excitatory_x, inhibitory_x)
        for layer in self.layers[: int(layer_index) + 1]:
            excitatory_x, inhibitory_x = layer(excitatory_x, inhibitory_x)
        return excitatory_x

    @property
    def branch_layers(self):
        """Get all layers that have branch structures.

        Inhibitory cells built as an MLP (``inhibitory_network_type: mlp``)
        have no branches and contribute none.
        """
        branch_layers = []
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            if hasattr(layer, "excitatory_cells"):
                branch_layers.extend(layer.excitatory_cells.branch_layers)
            inhibitory_cells = getattr(layer, "inhibitory_cells", None)
            if inhibitory_cells is not None:
                branch_layers.extend(getattr(inhibitory_cells, "branch_layers", ()))
        return branch_layers

    @property
    def n_branch_layers(self):
        """Get the number of branch layers"""
        return len(self.branch_layers)

    def get_effective_params(self) -> int:
        """Calculate the effective number of parameters after topk sparsity."""
        effective = 0

        for layer in self.layers:
            # Process excitatory cells
            if (
                hasattr(layer, "excitatory_cells")
                and layer.excitatory_cells is not None
            ):
                excitatory_cells = layer.excitatory_cells
                for branch_layer in excitatory_cells.branch_layers:
                    # Excitatory branch (TopKLinear)
                    if (
                        hasattr(branch_layer, "branch_excitation")
                        and branch_layer.branch_excitation is not None
                    ):
                        if hasattr(branch_layer.branch_excitation, "K"):
                            effective += (
                                branch_layer.branch_excitation.out_features
                                * branch_layer.branch_excitation.K
                            )

                    # Inhibitory branch (if present)
                    if (
                        hasattr(branch_layer, "branch_inhibition")
                        and branch_layer.branch_inhibition is not None
                    ):
                        if hasattr(branch_layer.branch_inhibition, "K"):
                            effective += (
                                branch_layer.branch_inhibition.out_features
                                * branch_layer.branch_inhibition.K
                            )

                    # Branches to output (BlockLinear - dense, so full params)
                    if (
                        hasattr(branch_layer, "branches_to_output")
                        and branch_layer.branches_to_output is not None
                    ):
                        effective += sum(
                            p.numel()
                            for p in branch_layer.branches_to_output.parameters()
                        )

                    # Reactivation (if present)
                    if hasattr(branch_layer, "reactivation") and not isinstance(
                        branch_layer.reactivation, nn.Identity
                    ):
                        effective += sum(
                            p.numel() for p in branch_layer.reactivation.parameters()
                        )

            # Process inhibitory cells
            if (
                hasattr(layer, "inhibitory_cells")
                and layer.inhibitory_cells is not None
            ):
                inhibitory_cells = layer.inhibitory_cells
                if not hasattr(inhibitory_cells, "branch_layers"):
                    # An MLP (inhibitory_network_type: mlp) is dense, so every
                    # parameter acts.
                    effective += sum(p.numel() for p in inhibitory_cells.parameters())
                    continue
                for branch_layer in inhibitory_cells.branch_layers:
                    # Excitatory branch (TopKLinear)
                    if (
                        hasattr(branch_layer, "branch_excitation")
                        and branch_layer.branch_excitation is not None
                    ):
                        if hasattr(branch_layer.branch_excitation, "K"):
                            effective += (
                                branch_layer.branch_excitation.out_features
                                * branch_layer.branch_excitation.K
                            )

                    # Inhibitory branch (if present)
                    if (
                        hasattr(branch_layer, "branch_inhibition")
                        and branch_layer.branch_inhibition is not None
                    ):
                        if hasattr(branch_layer.branch_inhibition, "K"):
                            effective += (
                                branch_layer.branch_inhibition.out_features
                                * branch_layer.branch_inhibition.K
                            )

                    # Branches to output (BlockLinear - dense, so full params)
                    if (
                        hasattr(branch_layer, "branches_to_output")
                        and branch_layer.branches_to_output is not None
                    ):
                        effective += sum(
                            p.numel()
                            for p in branch_layer.branches_to_output.parameters()
                        )

                    # Reactivation (if present)
                    if hasattr(branch_layer, "reactivation") and not isinstance(
                        branch_layer.reactivation, nn.Identity
                    ):
                        effective += sum(
                            p.numel() for p in branch_layer.reactivation.parameters()
                        )

        return effective


class ConfigurableEINetwork(ExcitationInhibitionNetwork):
    """
    ExcitationInhibitionNetwork that is initialized from structured config.

    This provides a clean way to create EINet instances from structured configuration,
    separating configuration structure from implementation details.
    """

    def __init__(
        self,
        config: Union[dict, DictConfig],
        input_dim: int,
        synapse_mode: Optional[str] = None,
        use_shunting: Optional[bool] = None,
        weight_transform: Optional[str] = None,
        flatten_dendrites: bool = False,
    ):
        """
        Initialize from structured config.

        Args:
            config: Structured configuration (e.g., core_config from YAML)
                   Must contain 'architecture', 'connectivity', 'transfer', etc. sections
            input_dim: Input dimension from encoder
            synapse_mode: Override synapse mode (e.g., "ei", "mlp").
            use_shunting: Override shunting inhibition flag.
            weight_transform: Override weight transform (e.g., "exp", "identity").
        """
        from dendritic_modeling.scripts.script_utils.config_utils import (
            canonicalize_model_core_flags,
            prepare_ei_network_params,
        )

        check_retired_inhibitory_routing(config, recurrent=False)
        # A plain-dict copy: callers pass typed CoreConfig objects or mappings.
        core_config = canonicalize_model_core_flags(config)
        params = prepare_ei_network_params(core_config, input_dim)

        if synapse_mode is not None:
            params["synapse_mode"] = synapse_mode
        if use_shunting is not None:
            params["use_shunting"] = use_shunting
        if weight_transform is not None:
            params["weight_transform"] = weight_transform

        if flatten_dendrites:
            exc_branch_factors = deepcopy(params["excitatory_branch_factors"])
            for i in range(1, len(exc_branch_factors)):
                exc_branch_factors[i] = (
                    exc_branch_factors[i] * exc_branch_factors[i - 1]
                )
            branch_compartments_per_soma = sum(exc_branch_factors)
            params["excitatory_branch_factors"] = [branch_compartments_per_soma]

        super().__init__(**params)

        from dendritic_modeling.config.biological import resolve_biological_neuron

        core_type = core_config.get("type") or "einet"
        if resolve_biological_neuron(core_type, core_config.get("biological_neuron")):
            from dendritic_modeling.networks.architectures.biological_constraints import (
                enforce_biological_neuron,
            )

            enforce_biological_neuron(core_type, self)


__all__ = ["ConfigurableEINetwork", "ExcitationInhibitionNetwork"]
