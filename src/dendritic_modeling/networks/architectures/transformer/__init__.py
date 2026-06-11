"""Transformer replacement utilities built around dendritic modules."""

from dendritic_modeling.networks.architectures.transformer.dendritic_ffn import (
    DendriticFFNReplacement,
    EIStackMLPSlot,
    LayerwiseEIStackReplacement,
    ReplacementRecord,
    apply_transformer_replacement_config,
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
    estimate_deepseek_dense_ffn_params,
    estimate_dendritic_ffn_params,
    load_transformer_with_dendritic_replacements,
    replace_transformer_mlp_layers,
    replace_transformer_mlp_layers_with_ei_stack,
    resolve_transformer_layers,
)

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
