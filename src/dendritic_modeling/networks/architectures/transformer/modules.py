"""Replacement modules for transformer feed-forward blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNet,
    DendriNetConfig,
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.replacement import (
    LayerwiseTokenReplacementStack,
    TokenReplacementCell,
    make_linear_output_projection,
    transformed_feature_dim,
)
from dendritic_modeling.networks.architectures.transformer.estimates import (
    estimate_dendritic_ffn_params,
    estimate_gated_dendritic_ffn_params,
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
        indexed_use_fused_gather: bool = False,
        indexed_use_triton_gather: bool = False,
        indexed_use_cuda_gather: bool = False,
        indexed_use_auto_gather: bool = False,
        indexed_recompute_backward: bool = False,
        indexed_comparison_backend: str | None = None,
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
                "indexed_use_fused_gather": indexed_use_fused_gather,
                "indexed_use_triton_gather": indexed_use_triton_gather,
                "indexed_use_cuda_gather": indexed_use_cuda_gather,
                "indexed_use_auto_gather": indexed_use_auto_gather,
                "indexed_recompute_backward": indexed_recompute_backward,
                "indexed_comparison_backend": indexed_comparison_backend,
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
            config=DendriNetConfig(
                n_soma=dendritic_units,
                branch_factors=tuple(branch_factors),
                excitatory_input_dim=input_dim,
                excitatory_synapses_per_branch=synapses_per_branch,
                inhibitory_input_dim=None,
                inhibitory_synapses_per_branch=None,
                somatic_synapses=bool(somatic_synapses),
                synapse_config=synapse_config,
            ),
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
        self.reactivate = bool(reactivate)
        self.reactivation_type = str(reactivation_type)
        self.uses_pre_norm = not isinstance(self.pre_norm, nn.Identity)

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
        reactivation_params = sum(
            param.numel()
            for branch_layer in self.core.branch_layers
            for param in branch_layer.reactivation.parameters()
        )
        reactivation_units = sum(
            int(branch_layer.branch_config.output_dim)
            for branch_layer in self.core.branch_layers
        )
        params_per_unit = (
            reactivation_params // reactivation_units
            if reactivation_units and reactivation_params % reactivation_units == 0
            else 0
        )
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
            reactivation_parameters_per_unit=params_per_unit,
            pre_norm=self.uses_pre_norm,
        )


class GatedDendriticFFNReplacement(nn.Module):
    """Two-path dendritic FFN with a SwiGLU-style multiplicative bottleneck.

    A conventional Llama/OLMo MLP learns separate gate and value projections,
    multiplies them after the gate activation, and projects the result back to
    the residual stream.  The original :class:`DendriticFFNReplacement` has a
    single dendritic path, so it cannot express that factorization directly.
    This variant uses two independently wired DendriNet cores while sharing the
    input adapter and the final output projection::

        y = output_projection(activation(gate_core(x)) * value_core(x))

    It intentionally reuses the same dendritic configuration for both paths so
    TopK, reactivation calibration, and rewiring infrastructure work unchanged.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        dendritic_units: int,
        gate_activation: str = "silu",
        value_seed_offset: int = 1,
        teacher_topk_init: bool = False,
        output_topk: int | None = None,
        **dendritic_kwargs: object,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        dendritic_units = int(dendritic_units)
        gate_activation = str(gate_activation).lower()
        if gate_activation not in {
            "silu",
            "swish",
            "gelu",
            "relu",
            "identity",
            "linear",
            "none",
        }:
            raise ValueError(
                f"Unsupported gated dendritic activation {gate_activation!r}"
            )

        gate_kwargs = dict(dendritic_kwargs)
        value_kwargs = dict(dendritic_kwargs)
        indexed_seed = value_kwargs.get("indexed_seed")
        if indexed_seed is not None:
            value_kwargs["indexed_seed"] = int(indexed_seed) + int(value_seed_offset)

        gate_path = DendriticFFNReplacement(
            hidden_size=hidden_size,
            dendritic_units=dendritic_units,
            **gate_kwargs,
        )
        value_path = DendriticFFNReplacement(
            hidden_size=hidden_size,
            dendritic_units=dendritic_units,
            **value_kwargs,
        )
        # Keep only the modules used by this factorization.  The temporary path
        # projections are deliberately discarded rather than left as unused
        # parameters (which would also break strict DDP configurations).
        self.pre_norm = gate_path.pre_norm
        self.input_adapter = gate_path.input_adapter
        self.gate_core = gate_path.core
        self.value_core = value_path.core
        output_bias = bool(dendritic_kwargs.get("output_bias", False))
        if output_topk is None:
            self.output_projection = make_linear_output_projection(
                dendritic_units,
                hidden_size,
                bias=output_bias,
                init_std=float(dendritic_kwargs.get("output_init_std", 0.02)),
            )
        else:
            output_topk = int(output_topk)
            if output_topk < 1 or output_topk > dendritic_units:
                raise ValueError("output_topk must be in [1, dendritic_units]")
            if output_bias:
                raise ValueError("output_topk currently requires output_bias=false")
            output_seed = dendritic_kwargs.get("indexed_seed")
            if output_seed is not None:
                output_seed = int(output_seed) + 2 * int(value_seed_offset)
            self.output_projection = IndexedSparseLinear(
                in_features=dendritic_units,
                out_features=hidden_size,
                K=output_topk,
                init_method=str(
                    dendritic_kwargs.get("topk_init_method", "xavier_normal")
                ),
                weight_transform="identity",
                seed=output_seed,
                output_chunk_size=int(
                    dendritic_kwargs.get("indexed_output_chunk_size", 2048)
                ),
                use_fused_gather=bool(
                    dendritic_kwargs.get("indexed_use_fused_gather", False)
                ),
                use_triton_gather=bool(
                    dendritic_kwargs.get("indexed_use_triton_gather", False)
                ),
                use_cuda_gather=bool(
                    dendritic_kwargs.get("indexed_use_cuda_gather", False)
                ),
                use_auto_gather=bool(
                    dendritic_kwargs.get("indexed_use_auto_gather", False)
                ),
                comparison_backend=dendritic_kwargs.get(
                    "indexed_comparison_backend"
                ),
            )

        self.hidden_size = hidden_size
        self.dendritic_units = dendritic_units
        self.input_transform = gate_path.input_transform
        self.output_scale = float(dendritic_kwargs.get("output_scale", 1.0))
        self.gate_activation = gate_activation
        self.value_seed_offset = int(value_seed_offset)
        self.teacher_topk_init = bool(teacher_topk_init)
        self.output_topk = output_topk
        self.teacher_topk_diagnostics: dict[str, float | int] = {}
        self.uses_pre_norm = not isinstance(self.pre_norm, nn.Identity)
        self.reference_intermediate_size: int | None = None
        self._estimate_kwargs = {
            "hidden_size": hidden_size,
            "dendritic_units": dendritic_units,
            "branch_factors": list(gate_path.branch_factors),
            "synapses_per_branch": gate_path.synapses_per_branch,
            "candidate_size": gate_path.indexed_candidate_size,
            "input_transform": gate_path.input_transform,
            "topk_type": gate_path.topk_type,
            "output_bias": (
                isinstance(self.output_projection, nn.Linear)
                and self.output_projection.bias is not None
            ),
            "output_topk": self.output_topk,
            "somatic_synapses": gate_path.somatic_synapses,
            "reactivation_parameters_per_unit": self._reactivation_params_per_unit(),
            "pre_norm": self.uses_pre_norm,
        }

    @classmethod
    def from_mlp(
        cls,
        mlp: nn.Module,
        **kwargs: object,
    ) -> GatedDendriticFFNReplacement:
        """Construct a gated replacement by inferring the source hidden size."""
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        replacement = cls(hidden_size=hidden_size, **kwargs)
        replacement.reference_intermediate_size = intermediate_size
        if replacement.teacher_topk_init:
            replacement.initialize_from_mlp(mlp)
        return replacement

    def _single_synapse_layer(self, core: DendriNet, *, path: str) -> nn.Module:
        if len(core.branch_layers) != 1:
            raise ValueError(
                "teacher_topk_init requires branch_factors=[] so each gated "
                f"{path} path has exactly one somatic synapse layer"
            )
        branch = core.branch_layers[0]
        synapse = branch.branch_excitation
        if synapse is None or not all(
            hasattr(synapse, name)
            for name in ("connection_indices", "pre_w", "K", "weight_transform")
        ):
            raise TypeError(
                "teacher_topk_init requires an indexed or indexed_rewire "
                f"synapse in the {path} path"
            )
        if not isinstance(branch.reactivation, nn.Identity):
            raise ValueError("teacher_topk_init requires reactivate=false")
        if str(synapse.weight_transform).lower() != "identity":
            raise ValueError("teacher_topk_init requires weight_transform='identity'")
        return synapse

    @staticmethod
    def _copy_topk_projection(
        source: nn.Linear,
        target: nn.Module,
        *,
        path: str,
    ) -> float:
        if source.bias is not None:
            raise ValueError(
                f"teacher_topk_init cannot represent a bias in teacher {path}"
            )
        weight = source.weight.detach()
        expected = (int(target.out_features), int(target.in_features))
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"Teacher {path} weight has shape {tuple(weight.shape)}, "
                f"expected {expected}"
            )
        k = int(target.K)
        # Canonical input-index order makes fixed-topology bitmask export
        # numerically stable: serialization no longer changes BF16 reduction
        # order when reconstructing the same sparse row.
        indices = torch.topk(weight.abs(), k=k, dim=1, sorted=False).indices
        indices = indices.sort(dim=1).values
        selected = weight.gather(1, indices)
        with torch.no_grad():
            target.connection_indices.copy_(
                indices.to(device=target.connection_indices.device)
            )
            target.pre_w.copy_(
                selected.to(device=target.pre_w.device, dtype=target.pre_w.dtype)
            )
        total_energy = weight.float().square().sum().clamp_min(1e-30)
        retained_energy = selected.float().square().sum() / total_energy
        return float(retained_energy.item())

    def initialize_from_mlp(self, mlp: nn.Module) -> dict[str, float | int]:
        """Magnitude-TopK initialize a one-level gated cell from a dense MLP.

        This is exact when ``K == hidden_size`` and otherwise initializes each
        gate/value row from its largest-magnitude teacher weights.  The dense
        down projection is copied without pruning.
        """
        if self.input_transform != "identity":
            raise ValueError("teacher_topk_init requires input_transform='identity'")
        if self.output_scale != 1.0:
            raise ValueError("teacher_topk_init requires output_scale=1.0")
        required = ("gate_proj", "up_proj", "down_proj")
        if not all(
            isinstance(getattr(mlp, name, None), nn.Linear) for name in required
        ):
            raise TypeError(
                "teacher_topk_init expects gate_proj, up_proj, and down_proj "
                "nn.Linear modules"
            )
        gate_synapse = self._single_synapse_layer(self.gate_core, path="gate")
        value_synapse = self._single_synapse_layer(self.value_core, path="value")
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        if self.dendritic_units != int(gate_proj.out_features):
            raise ValueError(
                "teacher_topk_init requires dendritic_units to equal the "
                f"teacher intermediate size ({gate_proj.out_features})"
            )
        gate_energy = self._copy_topk_projection(
            gate_proj,
            gate_synapse,
            path="gate_proj",
        )
        value_energy = self._copy_topk_projection(
            up_proj,
            value_synapse,
            path="up_proj",
        )
        if isinstance(self.output_projection, IndexedSparseLinear):
            down_energy = self._copy_topk_projection(
                down_proj,
                self.output_projection,
                path="down_proj",
            )
        else:
            if tuple(down_proj.weight.shape) != tuple(
                self.output_projection.weight.shape
            ):
                raise ValueError(
                    "Teacher down_proj weight shape does not match the gated "
                    "dendritic output projection"
                )
            with torch.no_grad():
                self.output_projection.weight.copy_(
                    down_proj.weight.to(
                        device=self.output_projection.weight.device,
                        dtype=self.output_projection.weight.dtype,
                    )
                )
                if down_proj.bias is not None:
                    if self.output_projection.bias is None:
                        raise ValueError(
                            "Teacher down_proj has a bias but output_bias is disabled"
                        )
                    self.output_projection.bias.copy_(
                        down_proj.bias.to(
                            device=self.output_projection.bias.device,
                            dtype=self.output_projection.bias.dtype,
                        )
                    )
                elif self.output_projection.bias is not None:
                    self.output_projection.bias.zero_()
            down_energy = 1.0
        diagnostics: dict[str, float | int] = {
            "k": int(gate_synapse.K),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.dendritic_units,
            "gate_weight_energy_retained": gate_energy,
            "value_weight_energy_retained": value_energy,
            "down_weight_energy_retained": down_energy,
        }
        self.teacher_topk_diagnostics = diagnostics
        return diagnostics

    def _activate_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_activation in {"silu", "swish"}:
            return F.silu(x)
        if self.gate_activation == "gelu":
            return F.gelu(x)
        if self.gate_activation == "relu":
            return F.relu(x)
        return x

    def _reactivation_params_per_unit(self) -> int:
        parameters = sum(
            param.numel()
            for branch_layer in self.gate_core.branch_layers
            for param in branch_layer.reactivation.parameters()
        )
        units = sum(
            int(branch_layer.branch_config.output_dim)
            for branch_layer in self.gate_core.branch_layers
        )
        return parameters // units if units and parameters % units == 0 else 0

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        """Run both dendritic paths on every leading token/batch dimension."""
        del args, kwargs
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        adapted = self.input_adapter(self.pre_norm(hidden_states))
        gated = self._activate_gate(self.gate_core(adapted))
        value = self.value_core(adapted)
        return self.output_projection(gated * value) * self.output_scale

    def parameter_estimate(self) -> dict[str, int]:
        """Estimate stored/active parameters across both dendritic paths."""
        return estimate_gated_dendritic_ffn_params(**self._estimate_kwargs)


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
