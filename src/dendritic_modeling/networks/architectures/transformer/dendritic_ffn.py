"""Dendritic replacements for transformer feed-forward blocks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config.conversion import normalize_sparsity_type, to_plain_dict
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    warn_alias_conflicts,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)


def _first_parameter(module: nn.Module) -> nn.Parameter | None:
    return next(module.parameters(), None)


def _get_attr_path(root: object, path: str) -> object:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(path)
        current = getattr(current, part)
    return current


def _set_attr_path(root: object, path: str, value: object) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = _get_attr_path(root, parent_path) if parent_path else root
    setattr(parent, name, value)


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    """Convert config-like objects into plain dictionaries."""
    return to_plain_dict(obj)


def resolve_transformer_layers(
    model: nn.Module,
    *,
    layers_attr: str | None = None,
) -> nn.ModuleList | list[nn.Module]:
    """Resolve a decoder layer list from common Hugging Face model layouts.

    DeepSeek/Llama-style causal language models usually expose decoder layers
    as ``model.model.layers``.  Passing ``layers_attr`` makes the lookup
    explicit for custom wrappers.
    """
    if layers_attr is not None:
        layers = _get_attr_path(model, layers_attr)
        if not isinstance(layers, (nn.ModuleList, list)):
            raise TypeError(f"{layers_attr!r} does not resolve to a layer list")
        return layers

    candidates = (
        "model.layers",
        "model.model.layers",
        "layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for path in candidates:
        try:
            layers = _get_attr_path(model, path)
        except AttributeError:
            continue
        if isinstance(layers, (nn.ModuleList, list)):
            return layers
    raise ValueError(
        "Could not find transformer decoder layers. Pass layers_attr explicitly."
    )


def _infer_mlp_dims(mlp: nn.Module) -> tuple[int, int | None]:
    hidden_candidates = []
    intermediate_candidates = []

    for attr in ("gate_proj", "up_proj", "fc1", "dense_h_to_4h"):
        layer = getattr(mlp, attr, None)
        if isinstance(layer, nn.Linear):
            hidden_candidates.append(layer.in_features)
            intermediate_candidates.append(layer.out_features)

    for attr in ("down_proj", "fc2", "dense_4h_to_h"):
        layer = getattr(mlp, attr, None)
        if isinstance(layer, nn.Linear):
            hidden_candidates.append(layer.out_features)
            intermediate_candidates.append(layer.in_features)

    if not hidden_candidates:
        linears = [mod for mod in mlp.modules() if isinstance(mod, nn.Linear)]
        if not linears:
            raise ValueError("Could not infer MLP dimensions; no Linear modules found")
        hidden_candidates.append(linears[0].in_features)
        hidden_candidates.append(linears[-1].out_features)
        if len(linears) > 1:
            intermediate_candidates.append(linears[0].out_features)

    hidden_size = max(set(hidden_candidates), key=hidden_candidates.count)
    intermediate_size = (
        max(set(intermediate_candidates), key=intermediate_candidates.count)
        if intermediate_candidates
        else None
    )
    return int(hidden_size), (
        None if intermediate_size is None else int(intermediate_size)
    )


def _as_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _first_list_value(value: Any, default: int) -> int:
    if isinstance(value, list):
        if not value:
            return int(default)
        return int(value[0])
    if value is None:
        return int(default)
    return int(value)


def _last_list_value(value: Any, default: int) -> int:
    if isinstance(value, list):
        if not value:
            return int(default)
        return int(value[-1])
    if value is None:
        return int(default)
    return int(value)


def _normalize_topk_type(value: Any) -> str:
    return normalize_sparsity_type(value)


def estimate_deepseek_dense_ffn_params(
    hidden_size: int = 7168,
    intermediate_size: int = 18432,
    *,
    gated: bool = True,
    bias: bool = False,
) -> int:
    """Estimate parameters in a DeepSeek/Llama-style dense FFN block."""
    projections = 3 if gated else 2
    params = projections * int(hidden_size) * int(intermediate_size)
    if bias:
        params += (2 if gated else 1) * int(intermediate_size) + int(hidden_size)
    return params


def estimate_dendritic_ffn_params(
    *,
    hidden_size: int,
    dendritic_units: int,
    branch_factors: Sequence[int],
    synapses_per_branch: int,
    candidate_size: int | None = None,
    input_transform: str = "signed_split",
    topk_type: str = "indexed",
    output_bias: bool = False,
    somatic_synapses: bool = True,
) -> dict[str, int]:
    """Estimate stored and active parameters for a dendritic FFN replacement."""
    input_dim = int(hidden_size) * (2 if input_transform == "signed_split" else 1)
    layer_sizes = [int(dendritic_units)]
    for branch_factor in branch_factors:
        layer_sizes.append(layer_sizes[-1] * int(branch_factor))
    dendritic_layer_sizes = list(reversed(layer_sizes))
    if not somatic_synapses:
        dendritic_layer_sizes = dendritic_layer_sizes[:-1]
    branch_units = sum(dendritic_layer_sizes)

    active_synapses = branch_units * int(synapses_per_branch)
    if topk_type == "indexed":
        stored_synapses = active_synapses
    elif topk_type == "indexed_dynamic":
        if candidate_size is None:
            candidate_size = min(
                input_dim,
                max(int(synapses_per_branch), 4 * int(synapses_per_branch)),
            )
        stored_synapses = branch_units * int(candidate_size)
    else:
        stored_synapses = branch_units * input_dim

    blocklinear = sum(
        int(parent) * int(branch_factor)
        for parent, branch_factor in zip(layer_sizes, branch_factors)
    )
    projection = int(dendritic_units) * int(hidden_size)
    if output_bias:
        projection += int(hidden_size)
    return {
        "input_dim": input_dim,
        "branch_units": branch_units,
        "active_synapses": active_synapses,
        "stored_synapses": stored_synapses,
        "blocklinear": blocklinear,
        "output_projection": projection,
        "stored_total": stored_synapses + blocklinear + projection,
        "active_total": active_synapses + blocklinear + projection,
    }


class DendriticFFNReplacement(nn.Module):
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
        topk_type: str = "indexed",
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
        use_shunting: bool = True,
        use_additive_normalization: bool = False,
        reactivate: bool = True,
        reactivation_type: str = "param_tanh",
        weight_transform: str = "softplus",
        topk_init_method: str = "xavier_normal",
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
    ):
        super().__init__()

        if input_transform not in {"identity", "relu", "signed_split"}:
            raise ValueError("input_transform must be identity, relu, or signed_split")
        if dendritic_units < 1:
            raise ValueError("dendritic_units must be >= 1")
        if synapses_per_branch < 1:
            raise ValueError("synapses_per_branch must be >= 1")

        self.hidden_size = int(hidden_size)
        self.dendritic_units = int(dendritic_units)
        self.branch_factors = _as_list(branch_factors)
        self.synapses_per_branch = int(synapses_per_branch)
        self.indexed_candidate_size = indexed_candidate_size
        self.indexed_selection = indexed_selection
        self.input_transform = input_transform
        self.output_scale = float(output_scale)
        self.topk_type = topk_type
        self.topk_noise_level = float(topk_noise_level)
        self.topk_temperature = float(topk_temperature)
        self.topk_ultrafast = bool(topk_ultrafast)

        input_dim = self.hidden_size * (2 if input_transform == "signed_split" else 1)
        self.pre_norm = nn.LayerNorm(self.hidden_size) if pre_norm else nn.Identity()
        self.core = DendriNet(
            n_soma=self.dendritic_units,
            branch_factors=self.branch_factors,
            excitatory_input_dim=input_dim,
            excitatory_synapses_per_branch=self.synapses_per_branch,
            inhibitory_input_dim=None,
            inhibitory_synapses_per_branch=None,
            reactivate=reactivate,
            somatic_synapses=True,
            topk_init_method=topk_init_method,
            topk_noise_level=topk_noise_level,
            topk_type=topk_type,
            topk_temperature=topk_temperature,
            topk_ultrafast=topk_ultrafast,
            topk_weight_norm_order=topk_weight_norm_order,
            topk_gamma=topk_gamma,
            dense_to_sparse_initial_density=dense_to_sparse_initial_density,
            dense_to_sparse_initial_k=dense_to_sparse_initial_k,
            dense_to_sparse_start_step=dense_to_sparse_start_step,
            dense_to_sparse_end_step=dense_to_sparse_end_step,
            dense_to_sparse_update_interval=dense_to_sparse_update_interval,
            dense_to_sparse_schedule=dense_to_sparse_schedule,
            dense_to_sparse_freeze_on_end=dense_to_sparse_freeze_on_end,
            dense_to_sparse_advance_on_forward=dense_to_sparse_advance_on_forward,
            dense_to_sparse_prune_metric=dense_to_sparse_prune_metric,
            weight_transform=weight_transform,
            use_shunting=use_shunting,
            use_additive_normalization=use_additive_normalization,
            reactivation_type=reactivation_type,
            indexed_seed=indexed_seed,
            indexed_candidate_size=indexed_candidate_size,
            indexed_selection=indexed_selection,
            indexed_output_chunk_size=indexed_output_chunk_size,
        )
        self.output_projection = nn.Linear(
            self.dendritic_units, self.hidden_size, bias=output_bias
        )
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=output_init_std)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    @classmethod
    def from_mlp(cls, mlp: nn.Module, **kwargs: object) -> DendriticFFNReplacement:
        """Construct a replacement by inferring ``hidden_size`` from an MLP."""
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        replacement = cls(hidden_size=hidden_size, **kwargs)
        replacement.reference_intermediate_size = intermediate_size
        return replacement

    def _transform_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.input_transform == "identity":
            return hidden_states
        if self.input_transform == "relu":
            return torch.relu(hidden_states)
        return torch.cat(
            [torch.relu(hidden_states), torch.relu(-hidden_states)], dim=-1
        )

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        """Run the dendritic FFN on all leading token/batch dimensions."""
        del args, kwargs
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got "
                f"{hidden_states.shape[-1]}"
            )
        x = self.pre_norm(hidden_states)
        x = self._transform_input(x)
        dendritic_state = self.core(x)
        return self.output_projection(dendritic_state) * self.output_scale

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


class LayerwiseEIStackReplacement(nn.Module):
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
        topk_type: str = "indexed_dynamic",
        topk_temperature: float = 0.25,
        topk_weight_norm_order: int | None = None,
        topk_gamma: float = 1.0,
        weight_transform: str = "softplus",
        indexed_seed: int | None = None,
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
        use_noise: bool = False,
        compile_forward: bool = False,
        **kwargs: object,
    ):
        super().__init__()
        if input_transform not in {"identity", "relu", "signed_split"}:
            raise ValueError("input_transform must be identity, relu, or signed_split")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        if int(excitatory_cells) < 1:
            raise ValueError("excitatory_cells must be >= 1")
        if int(inhibitory_cells) < 1:
            raise ValueError("inhibitory_cells must be >= 1")

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.excitatory_cells = int(excitatory_cells)
        self.inhibitory_cells = int(inhibitory_cells)
        self.input_transform = str(input_transform)
        self.output_scale = float(output_scale)
        self.pre_norm = (
            nn.ModuleList(
                [nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)]
            )
            if pre_norm
            else nn.ModuleList([nn.Identity() for _ in range(self.num_layers)])
        )

        transformed_dim = self.hidden_size * (
            2 if self.input_transform == "signed_split" else 1
        )
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
                    compile_forward=bool(compile_forward),
                    **extra_kwargs,
                )
            )
            projection = nn.Linear(
                self.excitatory_cells,
                self.hidden_size,
                bias=bool(output_bias),
            )
            nn.init.normal_(projection.weight, mean=0.0, std=float(output_init_std))
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
            self.output_projections.append(projection)

    def _transform_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.input_transform == "identity":
            return hidden_states
        if self.input_transform == "relu":
            return torch.relu(hidden_states)
        return torch.cat(
            [torch.relu(hidden_states), torch.relu(-hidden_states)],
            dim=-1,
        )

    def forward_layer(
        self, layer_index: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got "
                f"{hidden_states.shape[-1]}"
            )
        idx = int(layer_index)
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"EI stack layer {idx} out of range for {self.num_layers}")
        original_shape = hidden_states.shape
        x = self.pre_norm[idx](hidden_states)
        x = x.reshape(-1, self.hidden_size)
        x = self._transform_input(x)
        excitatory, _ = self.layers[idx](x, x)
        output = self.output_projections[idx](excitatory) * self.output_scale
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


@dataclass
class ReplacementRecord:
    """Metadata for one patched transformer MLP."""

    layer_index: int
    mlp_attr: str
    original_mlp: nn.Module
    replacement: nn.Module


def build_dendritic_ffn_kwargs_from_core_config(
    core_config: Any,
    *,
    transformer_replacement: Any = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a structured ``model.core`` config into FFN replacement kwargs.

    This mirrors the AlexNet replacement convention: the replacement target is
    described separately, while the dendritic module itself is still described
    by the shared ``model.core`` fields.
    """
    core = _to_plain_mapping(core_config)
    tr = _to_plain_mapping(transformer_replacement)
    architecture = _to_plain_mapping(core.get("architecture", {}))
    connectivity = _to_plain_mapping(core.get("connectivity", {}))
    transfer = _to_plain_mapping(core.get("transfer", {}))
    morphology = _to_plain_mapping(core.get("morphology", {}))
    sparsity = _to_plain_mapping(core.get("sparsity", {}))
    indexed = _to_plain_mapping(sparsity.get("indexed", {}))
    dense_to_sparse = _to_plain_mapping(
        sparsity.get("dense_to_sparse", sparsity.get("annealed_topk", {}))
    )
    reactivation = _to_plain_mapping(core.get("reactivation", {}))
    core_type = str(core.get("type", "")).lower()
    morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        morphology,
        morphology_overrides,
        context="model.core.morphology",
    )

    dendritic_units = int(
        architecture.get(
            "dendritic_units",
            _last_list_value(architecture.get("excitatory_layer_sizes"), 512),
        )
    )
    branch_factors = architecture.get(
        "branch_factors",
        architecture.get("excitatory_branch_factors", [3, 3]),
    )
    synapses_per_branch = int(
        connectivity.get(
            "synapses_per_branch",
            _first_list_value(
                connectivity.get("ee_synapses_per_branch_per_layer"),
                40,
            ),
        )
    )

    kwargs: dict[str, Any] = {
        "dendritic_units": dendritic_units,
        "branch_factors": list(branch_factors),
        "synapses_per_branch": synapses_per_branch,
        "input_transform": tr.get(
            "input_transform",
            transfer.get("input_transform", "signed_split"),
        ),
        "topk_type": _normalize_topk_type(sparsity.get("type", "indexed_dynamic")),
        "topk_init_method": sparsity.get("init_method", "xavier_normal"),
        "topk_noise_level": float(sparsity.get("noise_level", 0.0)),
        "topk_temperature": float(sparsity.get("temperature", 0.5)),
        "topk_ultrafast": bool(sparsity.get("ultrafast", False)),
        "topk_weight_norm_order": sparsity.get("weight_norm_order"),
        "topk_gamma": float(sparsity.get("gamma", 1.0)),
        "dense_to_sparse_initial_density": float(
            dense_to_sparse.get("initial_density", 1.0)
        ),
        "dense_to_sparse_initial_k": dense_to_sparse.get("initial_k"),
        "dense_to_sparse_start_step": int(dense_to_sparse.get("start_step", 0)),
        "dense_to_sparse_end_step": int(dense_to_sparse.get("end_step", 1000)),
        "dense_to_sparse_update_interval": int(
            dense_to_sparse.get("update_interval", 1)
        ),
        "dense_to_sparse_schedule": str(dense_to_sparse.get("schedule", "cubic")),
        "dense_to_sparse_freeze_on_end": bool(
            dense_to_sparse.get("freeze_on_end", False)
        ),
        "dense_to_sparse_advance_on_forward": bool(
            dense_to_sparse.get("advance_on_forward", True)
        ),
        "dense_to_sparse_prune_metric": str(
            dense_to_sparse.get("prune_metric", "weight")
        ),
        "indexed_seed": indexed.get("seed", sparsity.get("seed")),
        "indexed_candidate_size": indexed.get("candidate_size"),
        "indexed_selection": indexed.get("selection", "standard"),
        "indexed_output_chunk_size": int(indexed.get("output_chunk_size", 2048)),
        "use_shunting": bool(
            morphology_overrides.get(
                "use_shunting", morphology.get("use_shunting", True)
            )
        ),
        "use_additive_normalization": bool(
            morphology_overrides.get(
                "use_additive_normalization",
                morphology.get("use_additive_normalization", False),
            )
        ),
        "weight_transform": morphology_overrides.get(
            "weight_transform", morphology.get("weight_transform", "softplus")
        ),
        "reactivate": bool(reactivation.get("enabled", True)),
        "reactivation_type": reactivation.get("type", "param_tanh"),
        "output_bias": bool(tr.get("output_bias", False)),
        "output_init_std": float(tr.get("output_init_std", 0.02)),
        "output_scale": float(tr.get("output_scale", 1.0)),
        "pre_norm": bool(tr.get("pre_norm", False)),
    }
    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_overrides.pop("kind", None)
    replacement_overrides.pop("type", None)
    kwargs.update(replacement_overrides)
    if overrides:
        kwargs.update(overrides)
    return kwargs


def build_ei_stack_kwargs_from_core_config(
    core_config: Any,
    *,
    transformer_replacement: Any = None,
    num_layers: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a shared ``model.core`` config into EI-stack kwargs."""
    core = _to_plain_mapping(core_config)
    tr = _to_plain_mapping(transformer_replacement)
    architecture = _to_plain_mapping(core.get("architecture", {}))
    connectivity = _to_plain_mapping(core.get("connectivity", {}))
    transfer = _to_plain_mapping(core.get("transfer", {}))
    morphology = _to_plain_mapping(core.get("morphology", {}))
    sparsity = _to_plain_mapping(core.get("sparsity", {}))
    indexed = _to_plain_mapping(sparsity.get("indexed", {}))
    dense_to_sparse = _to_plain_mapping(
        sparsity.get("dense_to_sparse", sparsity.get("annealed_topk", {}))
    )
    reactivation = _to_plain_mapping(core.get("reactivation", {}))
    core_type = str(core.get("type", "")).lower()
    morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        morphology,
        morphology_overrides,
        context="model.core.morphology",
    )

    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_overrides.pop("kind", None)
    replacement_overrides.pop("type", None)

    kwargs: dict[str, Any] = {
        "num_layers": int(num_layers),
        "excitatory_cells": _last_list_value(
            architecture.get("excitatory_layer_sizes"),
            200,
        ),
        "inhibitory_cells": _last_list_value(
            architecture.get("inhibitory_layer_sizes"),
            50,
        ),
        "excitatory_branch_factors": list(
            architecture.get("excitatory_branch_factors", [2, 2])
        ),
        "inhibitory_branch_factors": list(
            architecture.get("inhibitory_branch_factors", [2])
        ),
        "ee_synapses_per_branch": _first_list_value(
            connectivity.get("ee_synapses_per_branch_per_layer"),
            32,
        ),
        "ei_synapses_per_branch": _first_list_value(
            connectivity.get("ei_synapses_per_branch_per_layer"),
            16,
        ),
        "ie_synapses_per_branch": _first_list_value(
            connectivity.get("ie_synapses_per_branch_per_layer"),
            16,
        ),
        "ii_synapses_per_branch": _first_list_value(
            connectivity.get("ii_synapses_per_branch_per_layer"),
            8,
        ),
        "input_transform": tr.get(
            "input_transform",
            transfer.get("input_transform", "signed_split"),
        ),
        "use_shunting": bool(
            morphology_overrides.get(
                "use_shunting", morphology.get("use_shunting", True)
            )
        ),
        "use_additive_normalization": bool(
            morphology_overrides.get(
                "use_additive_normalization",
                morphology.get("use_additive_normalization", False),
            )
        ),
        "weight_transform": morphology_overrides.get(
            "weight_transform", morphology.get("weight_transform", "softplus")
        ),
        "topk_type": _normalize_topk_type(sparsity.get("type", "indexed_dynamic")),
        "topk_init_method": sparsity.get("init_method", "xavier_normal"),
        "topk_noise_level": float(sparsity.get("noise_level", 0.0)),
        "topk_temperature": float(sparsity.get("temperature", 0.25)),
        "topk_weight_norm_order": sparsity.get("weight_norm_order"),
        "topk_gamma": float(sparsity.get("gamma", 1.0)),
        "dense_to_sparse_initial_density": float(
            dense_to_sparse.get("initial_density", 1.0)
        ),
        "dense_to_sparse_initial_k": dense_to_sparse.get("initial_k"),
        "dense_to_sparse_start_step": int(dense_to_sparse.get("start_step", 0)),
        "dense_to_sparse_end_step": int(dense_to_sparse.get("end_step", 1000)),
        "dense_to_sparse_update_interval": int(
            dense_to_sparse.get("update_interval", 1)
        ),
        "dense_to_sparse_schedule": str(dense_to_sparse.get("schedule", "cubic")),
        "dense_to_sparse_freeze_on_end": bool(
            dense_to_sparse.get("freeze_on_end", False)
        ),
        "dense_to_sparse_advance_on_forward": bool(
            dense_to_sparse.get("advance_on_forward", True)
        ),
        "dense_to_sparse_prune_metric": str(
            dense_to_sparse.get("prune_metric", "weight")
        ),
        "indexed_seed": indexed.get("seed", sparsity.get("seed")),
        "indexed_candidate_size": indexed.get("candidate_size"),
        "indexed_selection": indexed.get("selection", "standard"),
        "indexed_output_chunk_size": int(indexed.get("output_chunk_size", 2048)),
        "reactivate": bool(reactivation.get("enabled", True)),
        "reactivation_type": reactivation.get("type", "param_tanh"),
        "output_bias": bool(tr.get("output_bias", False)),
        "output_init_std": float(tr.get("output_init_std", 0.02)),
        "output_scale": float(tr.get("output_scale", 1.0)),
        "pre_norm": bool(tr.get("pre_norm", False)),
    }
    kwargs.update(replacement_overrides)
    if overrides:
        kwargs.update(overrides)
    return kwargs


def _validate_layer_indices(layer_indices: Sequence[int]) -> list[int]:
    """Return the layer indices in order, or raise before a model is changed."""
    ordered = [int(index) for index in layer_indices]
    if not ordered:
        raise ValueError("layer_indices must contain at least one layer")
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"layer_indices must not repeat a layer, got {ordered}")
    return ordered


def _resolve_layer_target(
    layers: nn.ModuleList | list[nn.Module],
    layer_index: int,
    mlp_attr: str,
) -> tuple[nn.Module, nn.Module]:
    """Return one transformer layer and the module at ``mlp_attr`` in it."""
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(
            f"layer index {layer_index} out of range for {len(layers)} layers"
        )
    layer = layers[layer_index]
    try:
        target = _get_attr_path(layer, mlp_attr)
    except AttributeError as exc:
        raise AttributeError(
            f"layer {layer_index} has no attribute {mlp_attr!r}"
        ) from exc
    if not isinstance(target, nn.Module):
        raise TypeError(f"layer {layer_index} attribute {mlp_attr!r} is not a module")
    return layer, target


def _move_like(replacement: nn.Module, source: nn.Module) -> nn.Module:
    """Move ``replacement`` to the device and floating dtype of ``source``."""
    param = _first_parameter(source)
    if param is None:
        return replacement
    dtype = param.dtype if param.dtype.is_floating_point else None
    return replacement.to(device=param.device, dtype=dtype)


def replace_transformer_mlp_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
    *,
    replacement_kwargs: dict[str, object] | None = None,
    layers_attr: str | None = None,
    mlp_attr: str = "mlp",
    preserve_device_dtype: bool = True,
    replacement_class: type[nn.Module] = DendriticFFNReplacement,
) -> list[ReplacementRecord]:
    """Replace selected transformer layer MLPs with dendritic FFNs.

    This is intentionally model-object based rather than config-file based so
    it can work with Hugging Face DeepSeek/Llama-like modules after loading.
    ``mlp_attr`` may be a dotted path inside each layer, and
    ``replacement_class`` any module class with a ``from_mlp`` constructor.
    Every target is resolved and every replacement built before the model is
    changed, so an invalid index, path or setting leaves it untouched; a
    repeated layer index raises.
    """
    replacement_kwargs = dict(replacement_kwargs or {})
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    ordered_indices = _validate_layer_indices(layer_indices)
    from_mlp = getattr(replacement_class, "from_mlp", None)
    if not callable(from_mlp):
        raise TypeError("replacement_class must provide a callable from_mlp")
    targets = [
        (*_resolve_layer_target(layers, layer_index, mlp_attr), layer_index)
        for layer_index in ordered_indices
    ]
    prepared = []
    for layer, original_mlp, layer_index in targets:
        replacement = from_mlp(original_mlp, **replacement_kwargs)
        if preserve_device_dtype:
            replacement = _move_like(replacement, original_mlp)
        prepared.append((layer, original_mlp, replacement, layer_index))

    records: list[ReplacementRecord] = []
    for layer, original_mlp, replacement, layer_index in prepared:
        _set_attr_path(layer, mlp_attr, replacement)
        records.append(
            ReplacementRecord(
                layer_index=int(layer_index),
                mlp_attr=mlp_attr,
                original_mlp=original_mlp,
                replacement=replacement,
            )
        )
    return records


def replace_transformer_mlp_layers_with_ei_stack(
    model: nn.Module,
    layer_indices: Sequence[int],
    *,
    replacement_kwargs: dict[str, object] | None = None,
    layers_attr: str | None = None,
    mlp_attr: str = "mlp",
    preserve_device_dtype: bool = True,
) -> list[ReplacementRecord]:
    """Replace selected MLPs with slots from one shared EI stack.

    As :func:`replace_transformer_mlp_layers`, every target is checked and the
    stack built before the model is changed, and a repeated layer raises.
    """
    replacement_kwargs = dict(replacement_kwargs or {})
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    ordered_indices = _validate_layer_indices(layer_indices)
    targets = [
        (*_resolve_layer_target(layers, layer_index, mlp_attr), layer_index)
        for layer_index in ordered_indices
    ]

    first_mlp = targets[0][1]
    hidden_size, _ = _infer_mlp_dims(first_mlp)
    stack_num_layers = int(replacement_kwargs.pop("num_layers", len(ordered_indices)))
    if stack_num_layers != len(ordered_indices):
        raise ValueError(
            "EI stack num_layers must match the number of replaced transformer "
            f"layers ({stack_num_layers} != {len(ordered_indices)})"
        )
    stack = LayerwiseEIStackReplacement(
        hidden_size=hidden_size,
        num_layers=stack_num_layers,
        **replacement_kwargs,
    )
    if preserve_device_dtype:
        stack = _move_like(stack, first_mlp)

    records: list[ReplacementRecord] = []
    for stack_pos, (layer, original_mlp, layer_index) in enumerate(targets):
        replacement = EIStackMLPSlot(stack, stack_pos)
        _set_attr_path(layer, mlp_attr, replacement)
        records.append(
            ReplacementRecord(
                layer_index=int(layer_index),
                mlp_attr=mlp_attr,
                original_mlp=original_mlp,
                replacement=replacement,
            )
        )
    return records


def apply_transformer_replacement_config(
    model: nn.Module,
    transformer_replacement: Any,
    core_config: Any,
    *,
    replacement_kwargs: dict[str, Any] | None = None,
) -> list[ReplacementRecord]:
    """Apply a YAML-style transformer replacement config to a loaded model."""
    tr = _to_plain_mapping(transformer_replacement)
    if not bool(tr.get("enabled", True)):
        return []

    layer_indices = tr.get("layers", tr.get("layer_indices", []))
    if not layer_indices:
        raise ValueError(
            "transformer_replacement.layers must contain at least one layer"
        )

    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_kind = str(
        replacement_overrides.get(
            "kind", replacement_overrides.get("type", "dendritic_ffn")
        )
    ).lower()
    if replacement_kind in {"ei_stack", "layerwise_ei_stack", "einet_stack"}:
        kwargs = build_ei_stack_kwargs_from_core_config(
            core_config,
            transformer_replacement=tr,
            num_layers=len(layer_indices),
            overrides=replacement_kwargs,
        )
        return replace_transformer_mlp_layers_with_ei_stack(
            model,
            [int(idx) for idx in layer_indices],
            replacement_kwargs=kwargs,
            layers_attr=tr.get("layers_attr"),
            mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
            preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
        )

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=tr,
        overrides=replacement_kwargs,
    )
    return replace_transformer_mlp_layers(
        model,
        [int(idx) for idx in layer_indices],
        replacement_kwargs=kwargs,
        layers_attr=tr.get("layers_attr"),
        mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
        preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
    )


def load_transformer_with_dendritic_replacements(
    transformer_replacement: Any,
    core_config: Any,
) -> tuple[nn.Module, list[ReplacementRecord]]:
    """Load a Hugging Face causal LM and apply dendritic MLP replacements."""
    tr = _to_plain_mapping(transformer_replacement)
    model_name = tr.get("model_name", tr.get("backbone", ""))
    if not model_name:
        raise ValueError("transformer_replacement.model_name is required")

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load transformer_replacement models"
        ) from exc

    model_kwargs = _to_plain_mapping(tr.get("model_kwargs", {}))
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    records = apply_transformer_replacement_config(model, tr, core_config)
    return model, records


__all__ = [
    "DendriticFFNReplacement",
    "EIStackMLPSlot",
    "LayerwiseEIStackReplacement",
    "ReplacementRecord",
    "apply_transformer_replacement_config",
    "build_dendritic_ffn_kwargs_from_core_config",
    "build_ei_stack_kwargs_from_core_config",
    "estimate_deepseek_dense_ffn_params",
    "estimate_dendritic_ffn_params",
    "load_transformer_with_dendritic_replacements",
    "replace_transformer_mlp_layers",
    "replace_transformer_mlp_layers_with_ei_stack",
    "resolve_transformer_layers",
]
