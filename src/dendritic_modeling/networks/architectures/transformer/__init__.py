"""Transformer replacement utilities built around dendritic modules."""

from dendritic_modeling.networks.architectures.transformer.config_translation import (
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
)
from dendritic_modeling.networks.architectures.transformer.estimates import (
    estimate_deepseek_dense_ffn_params,
    estimate_dendritic_ffn_params,
    estimate_gated_dendritic_ffn_params,
)
from dendritic_modeling.networks.architectures.transformer.modules import (
    DendriticFFNReplacement,
    EIStackMLPSlot,
    GatedDendriticFFNReplacement,
    LayerwiseEIStackReplacement,
)
from dendritic_modeling.networks.architectures.transformer.patching import (
    ReplacementRecord,
    apply_transformer_replacement_config,
    load_transformer_with_dendritic_replacements,
    replace_transformer_mlp_layers,
    replace_transformer_mlp_layers_with_ei_stack,
    resolve_transformer_layers,
)
from dendritic_modeling.networks.architectures.transformer.scaling import (
    TransformerParameterSpec,
    estimate_dendritic_replacement_from_config,
    estimate_factorized_lm_head_params,
    estimate_factorized_token_embedding_params,
    estimate_indexed_attention_params,
    profile_dendritic_lm,
    profile_dendritic_lm_config,
    profile_transformer_component_options,
)

__all__ = [
    "DendriticFFNReplacement",
    "EIStackMLPSlot",
    "GatedDendriticFFNReplacement",
    "LayerwiseEIStackReplacement",
    "ReplacementRecord",
    "TransformerParameterSpec",
    "apply_transformer_replacement_config",
    "build_dendritic_ffn_kwargs_from_core_config",
    "build_ei_stack_kwargs_from_core_config",
    "estimate_deepseek_dense_ffn_params",
    "estimate_dendritic_ffn_params",
    "estimate_dendritic_replacement_from_config",
    "estimate_factorized_lm_head_params",
    "estimate_factorized_token_embedding_params",
    "estimate_gated_dendritic_ffn_params",
    "estimate_indexed_attention_params",
    "load_transformer_with_dendritic_replacements",
    "profile_dendritic_lm",
    "profile_dendritic_lm_config",
    "profile_transformer_component_options",
    "replace_transformer_mlp_layers",
    "replace_transformer_mlp_layers_with_ei_stack",
    "resolve_transformer_layers",
]
