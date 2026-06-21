"""Replacement modules for transformer feed-forward blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNet,
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.replacement import (
    LayerwiseTokenReplacementStack,
    TokenReplacementCell,
    make_linear_output_projection,
    transformed_feature_dim,
)
from dendritic_modeling.networks.architectures.transformer.estimates import (
    estimate_dendritic_ffn_params,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _as_list,
    _infer_mlp_dims,
)


class DendriticFFNReplacement(TokenReplacementCell):
    """Token-wise DendriNet replacement for transformer MLP/FFN modules.

    The module preserves the transformer contract ``[..., hidden_size] ->
    [..., hidden_size]``.  It is intended to replace the MLP submodule inside a
    residual transformer block; the residual connection itself remains outside
    this module.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        dendritic_units: int,
        branch_factors: Sequence[int] = (3, 3),
        synapses_per_branch: int = 40,
        input_transform: str = "signed_split",
        topk_type: str = "indexed_rewire",
        topk_noise_level: float = 0.0,
        topk_temperature: float = 0.5,
        topk_ultrafast: bool = False,
        topk_weight_norm_order: int | None = None,
        topk_gamma: float = 1.0,
        dense_to_sparse_initial_density: float = 1.0,
        dense_to_sparse_initial_k: int | None = None,
        dense_to_sparse_start_step: int = 0,
        dense_to_sparse_end_step: int = 1000,
        dense_to_sparse_update_interval: int = 1,
        dense_to_sparse_schedule: str = "cubic",
        dense_to_sparse_freeze_on_end: bool = False,
        dense_to_sparse_advance_on_forward: bool = True,
        dense_to_sparse_prune_metric: str = "weight",
        indexed_seed: int | None = None,
        indexed_candidate_size: int | None = None,
        indexed_selection: str = "standard",
        indexed_output_chunk_size: int = 2048,
        indexed_rewire_frequency: int = 100,
        indexed_rewire_quantile: float = 0.05,
        indexed_rewire_until_step: int | None = None,
        use_shunting: bool = True,
        use_additive_normalization: bool = False,
        reactivate: bool = True,
        reactivation_type: str = "param_tanh",
        somatic_synapses: bool = True,
        reactivation_init_m: float = 1.0,
        reactivation_init_b: float = 0.5,
        reactivation_init_policy: str = "analytical",
        reactivation_occupancy_quantile_low: float | None = None,
        reactivation_occupancy_quantile_high: float | None = None,
        reactivation_occupancy_target_low: float | None = None,
        reactivation_occupancy_target_high: float | None = None,
        reactivation_calibration_min_quantile_width: float = 1e-3,
        reactivation_calibration_max_m: float = 50.0,
        reactivation_calibration_revert_on_invalid: bool = True,
        reactivation_sigma_aware_k: float = 0.25,
        weight_transform: str = "softplus",
        topk_init_method: str = "xavier_normal",
        efficient_blocklinear: bool = False,
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
    ):
        if dendritic_units < 1:
            raise ValueError("dendritic_units must be >= 1")
        if synapses_per_branch < 1:
            raise ValueError("synapses_per_branch must be >= 1")

        hidden_size = int(hidden_size)
        dendritic_units = int(dendritic_units)
        branch_factors = _as_list(branch_factors)
        synapses_per_branch = int(synapses_per_branch)
        input_dim = transformed_feature_dim(hidden_size, input_transform)
        synapse_config = DendriticSynapseConfig.from_kwargs(
            {
                "topk_init_method": topk_init_method,
                "topk_noise_level": topk_noise_level,
                "topk_type": topk_type,
                "topk_temperature": topk_temperature,
                "topk_ultrafast": topk_ultrafast,
                "topk_weight_norm_order": topk_weight_norm_order,
                "topk_gamma": topk_gamma,
                "dense_to_sparse_initial_density": (dense_to_sparse_initial_density),
                "dense_to_sparse_initial_k": dense_to_sparse_initial_k,
                "dense_to_sparse_start_step": dense_to_sparse_start_step,
                "dense_to_sparse_end_step": dense_to_sparse_end_step,
                "dense_to_sparse_update_interval": (dense_to_sparse_update_interval),
                "dense_to_sparse_schedule": dense_to_sparse_schedule,
                "dense_to_sparse_freeze_on_end": dense_to_sparse_freeze_on_end,
                "dense_to_sparse_advance_on_forward": (
                    dense_to_sparse_advance_on_forward
                ),
                "dense_to_sparse_prune_metric": dense_to_sparse_prune_metric,
                "indexed_seed": indexed_seed,
                "indexed_candidate_size": indexed_candidate_size,
                "indexed_selection": indexed_selection,
                "indexed_output_chunk_size": indexed_output_chunk_size,
                "indexed_rewire_frequency": indexed_rewire_frequency,
                "indexed_rewire_quantile": indexed_rewire_quantile,
                "indexed_rewire_until_step": indexed_rewire_until_step,
                "use_shunting": use_shunting,
                "use_additive_normalization": use_additive_normalization,
                "reactivate": reactivate,
                "reactivation_type": reactivation_type,
                "reactivation_init_m": reactivation_init_m,
                "reactivation_init_b": reactivation_init_b,
                "reactivation_init_policy": reactivation_init_policy,
                "reactivation_occupancy_quantile_low": (
                    reactivation_occupancy_quantile_low
                ),
                "reactivation_occupancy_quantile_high": (
                    reactivation_occupancy_quantile_high
                ),
                "reactivation_occupancy_target_low": (
                    reactivation_occupancy_target_low
                ),
                "reactivation_occupancy_target_high": (
                    reactivation_occupancy_target_high
                ),
                "reactivation_calibration_min_quantile_width": (
                    reactivation_calibration_min_quantile_width
                ),
                "reactivation_calibration_max_m": reactivation_calibration_max_m,
                "reactivation_calibration_revert_on_invalid": (
                    reactivation_calibration_revert_on_invalid
                ),
                "reactivation_sigma_aware_k": reactivation_sigma_aware_k,
                "weight_transform": weight_transform,
                "efficient_blocklinear": bool(efficient_blocklinear),
            }
        )
        core = DendriNet(
            n_soma=dendritic_units,
            branch_factors=branch_factors,
            excitatory_input_dim=input_dim,
            excitatory_synapses_per_branch=synapses_per_branch,
            inhibitory_input_dim=None,
            inhibitory_synapses_per_branch=None,
            somatic_synapses=bool(somatic_synapses),
            synapse_config=synapse_config,
        )

        super().__init__(
            input_dim=hidden_size,
            output_dim=hidden_size,
            input_transform=input_transform,
            core=core,
            core_output_dim=dendritic_units,
            output_bias=output_bias,
            output_init_std=output_init_std,
            output_scale=output_scale,
            pre_norm=pre_norm,
        )

        self.hidden_size = hidden_size
        self.dendritic_units = dendritic_units
        self.branch_factors = branch_factors
        self.synapses_per_branch = synapses_per_branch
        self.indexed_candidate_size = indexed_candidate_size
        self.indexed_selection = indexed_selection
        self.indexed_rewire_frequency = int(indexed_rewire_frequency)
        self.indexed_rewire_quantile = float(indexed_rewire_quantile)
        self.indexed_rewire_until_step = indexed_rewire_until_step
        self.topk_type = topk_type
        self.topk_noise_level = float(topk_noise_level)
        self.topk_temperature = float(topk_temperature)
        self.topk_ultrafast = bool(topk_ultrafast)
        self.somatic_synapses = bool(somatic_synapses)
        self.efficient_blocklinear = bool(efficient_blocklinear)

    @classmethod
    def from_mlp(cls, mlp: nn.Module, **kwargs: object) -> DendriticFFNReplacement:
        """Construct a replacement by inferring ``hidden_size`` from an MLP."""
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        replacement = cls(hidden_size=hidden_size, **kwargs)
        replacement.reference_intermediate_size = intermediate_size
        return replacement

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        """Run the dendritic FFN on all leading token/batch dimensions."""
        return super().forward(hidden_states, *args, **kwargs)

    def parameter_estimate(self) -> dict[str, int]:
        """Return a parameter estimate matching this module's configuration."""
        return estimate_dendritic_ffn_params(
            hidden_size=self.hidden_size,
            dendritic_units=self.dendritic_units,
            branch_factors=self.branch_factors,
            synapses_per_branch=self.synapses_per_branch,
            candidate_size=self.indexed_candidate_size,
            input_transform=self.input_transform,
            topk_type=self.topk_type,
            output_bias=self.output_projection.bias is not None,
            somatic_synapses=self.somatic_synapses,
        )


def _effective_params_for_dendrinet(module: nn.Module | None) -> int:
    if module is None or not isinstance(module, DendriNet):
        return 0
    effective = 0
    for branch_layer in module.branch_layers:
        for attr in ("branch_excitation", "branch_inhibition"):
            synapse = getattr(branch_layer, attr, None)
            if synapse is None:
                continue
            if hasattr(synapse, "K") and hasattr(synapse, "out_features"):
                effective += int(synapse.out_features) * int(synapse.K)
            else:
                effective += sum(param.numel() for param in synapse.parameters())
        block = getattr(branch_layer, "branches_to_output", None)
        if block is not None:
            effective += sum(param.numel() for param in block.parameters())
        reactivation = getattr(branch_layer, "reactivation", None)
        if reactivation is not None and not isinstance(reactivation, nn.Identity):
            effective += sum(param.numel() for param in reactivation.parameters())
    return effective


class LayerwiseEIStackReplacement(LayerwiseTokenReplacementStack):
    """A shared EI stack whose layers replace transformer MLP slots.

    OLMo/Llama-style MLPs are interleaved with attention and residual blocks.
    This module therefore exposes one callable slot per transformer depth:
    layer ``k`` receives the hidden state at transformer block ``k``, runs the
    corresponding EI layer with its own E/I populations, and projects the
    excitatory population back to the model hidden size for the residual path.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        excitatory_cells: int = 200,
        inhibitory_cells: int = 50,
        excitatory_branch_factors: Sequence[int] = (2, 2),
        inhibitory_branch_factors: Sequence[int] = (2,),
        ee_synapses_per_branch: int = 32,
        ei_synapses_per_branch: int = 16,
        ie_synapses_per_branch: int = 16,
        ii_synapses_per_branch: int = 8,
        input_transform: str = "signed_split",
        use_shunting: bool = True,
        synapse_mode: str = "ei",
        reactivation_type: str = "param_tanh",
        reactivate: bool = True,
        topk_init_method: str = "xavier_normal",
        topk_noise_level: float = 0.0,
        topk_type: str = "indexed_rewire",
        topk_temperature: float = 0.25,
        topk_weight_norm_order: int | None = None,
        topk_gamma: float = 1.0,
        weight_transform: str = "softplus",
        indexed_seed: int | None = None,
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
        efficient_blocklinear: bool = False,
        use_noise: bool = False,
        compile_forward: bool = False,
        **kwargs: object,
    ):
        super().__init__(
            input_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=num_layers,
            input_transform=input_transform,
            output_scale=output_scale,
            pre_norm=pre_norm,
        )
        if int(excitatory_cells) < 1:
            raise ValueError("excitatory_cells must be >= 1")
        if int(inhibitory_cells) < 1:
            raise ValueError("inhibitory_cells must be >= 1")

        self.hidden_size = int(hidden_size)
        self.excitatory_cells = int(excitatory_cells)
        self.inhibitory_cells = int(inhibitory_cells)
        self.efficient_blocklinear = bool(efficient_blocklinear)
        transformed_dim = self.adapted_input_dim
        self.layers = nn.ModuleList()
        self.output_projections = nn.ModuleList()
        extra_kwargs = dict(kwargs)
        for layer_idx in range(self.num_layers):
            layer_seed = (
                None if indexed_seed is None else int(indexed_seed) + int(layer_idx)
            )
            self.layers.append(
                ExcitationInhibitionLayer(
                    n_excitatory_cells=self.excitatory_cells,
                    n_inhibitory_cells=self.inhibitory_cells,
                    excitatory_branch_factors=list(excitatory_branch_factors),
                    inhibitory_branch_factors=list(inhibitory_branch_factors),
                    excitatory_input_dim=transformed_dim,
                    inhibitory_input_dim=transformed_dim,
                    ee_synapses_per_branch=int(ee_synapses_per_branch),
                    ei_synapses_per_branch=int(ei_synapses_per_branch),
                    ie_synapses_per_branch=int(ie_synapses_per_branch),
                    ii_synapses_per_branch=int(ii_synapses_per_branch),
                    build_inhibitory_cells=True,
                    inhibitory_network_type="dendritic",
                    reactivate=bool(reactivate),
                    topk_init_method=str(topk_init_method),
                    use_shunting=bool(use_shunting),
                    synapse_mode=str(synapse_mode),
                    reactivation_type=str(reactivation_type),
                    topk_noise_level=float(topk_noise_level),
                    topk_type=str(topk_type),
                    topk_temperature=float(topk_temperature),
                    topk_weight_norm_order=topk_weight_norm_order,
                    topk_gamma=float(topk_gamma),
                    weight_transform=str(weight_transform),
                    use_noise=bool(use_noise),
                    indexed_seed=layer_seed,
                    efficient_blocklinear=self.efficient_blocklinear,
                    compile_forward=bool(compile_forward),
                    **extra_kwargs,
                )
            )
            projection = make_linear_output_projection(
                self.excitatory_cells,
                self.hidden_size,
                bias=bool(output_bias),
                init_std=float(output_init_std),
            )
            self.output_projections.append(projection)

    def forward_layer(
        self, layer_index: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        idx = self._validate_layer_index(layer_index)
        original_shape = hidden_states.shape
        x = self.encode_layer_input(idx, hidden_states)
        x = x.reshape(-1, self.adapted_input_dim)
        excitatory, _ = self.layers[idx](x, x)
        output = self.decode_layer_output(idx, excitatory)
        return output.reshape(original_shape)

    def layer_parameter_estimate(self, layer_index: int) -> dict[str, int]:
        idx = int(layer_index)
        layer = self.layers[idx]
        projection = self.output_projections[idx]
        norm = self.pre_norm[idx]
        stored = (
            sum(param.numel() for param in layer.parameters())
            + sum(param.numel() for param in projection.parameters())
            + sum(param.numel() for param in norm.parameters())
        )
        active = (
            _effective_params_for_dendrinet(getattr(layer, "excitatory_cells", None))
            + _effective_params_for_dendrinet(getattr(layer, "inhibitory_cells", None))
            + sum(param.numel() for param in projection.parameters())
            + sum(param.numel() for param in norm.parameters())
        )
        return {
            "stored_total": int(stored),
            "active_total": int(active),
        }


class EIStackMLPSlot(nn.Module):
    """One OLMo MLP replacement slot backed by a shared EI stack."""

    def __init__(self, stack: LayerwiseEIStackReplacement, layer_index: int):
        super().__init__()
        self.stack = stack
        self.layer_index = int(layer_index)

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        del args, kwargs
        return self.stack.forward_layer(self.layer_index, hidden_states)

    def parameter_estimate(self) -> dict[str, int]:
        return self.stack.layer_parameter_estimate(self.layer_index)
