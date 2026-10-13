"""Config normalization for local learning strategies."""

from __future__ import annotations

import logging
from typing import Any

from dendritic_modeling.config.keys import build_section
from dendritic_modeling.config.training import (
    LOCAL_RULE_VH_ALIASES,
    VOLTAGE_HOMEOSTASIS_ALIAS_TARGET,
    VOLTAGE_HOMEOSTASIS_ALIAS_WEIGHT,
    LocalRuleConfig,
)

logger = logging.getLogger(__name__)


def section_to_dict(section: Any) -> dict[str, Any]:
    """Convert a config section object to a plain dict."""

    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "asdict") and callable(section.asdict):
        return section.asdict()
    if hasattr(section, "__dict__"):
        return {
            key: value
            for key, value in vars(section).items()
            if not key.startswith("_")
        }
    return {}


def coerce_legacy_local_rule_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy flat local learning config to nested LocalRuleConfig layout."""

    cfg = dict(config_dict)
    alias_mode = cfg.pop("inhibitory_cell_update_mode", None)
    if alias_mode is not None:
        existing_mode = cfg.get("explicit_inhibitory_update_mode")
        if (
            existing_mode is not None
            and LocalRuleConfig._normalize_explicit_inhibitory_update_mode(
                existing_mode
            )
            != LocalRuleConfig._normalize_explicit_inhibitory_update_mode(alias_mode)
        ):
            raise ValueError(
                "Conflicting inhibitory-cell update aliases in LocalRuleConfig"
            )
        cfg["explicit_inhibitory_update_mode"] = alias_mode

    def ensure_section(section_name: str) -> dict[str, Any]:
        section = section_to_dict(cfg.get(section_name))
        cfg[section_name] = section
        return section

    legacy_group_fields = {
        "three_factor": [
            "dynamics_mode",
            "use_conductance_scaling",
            "use_driving_force",
            "theta",
            "e_rev_exc",
            "e_rev_inh",
            "additive_gain_mode",
            "additive_stats_ema_alpha",
        ],
        "four_factor": [
            "rho_mode",
            "rho_estimator",
            "ema_alpha",
            "layer_wise_rho_scale",
            "augment_k",
            "augment_noise_sigma",
        ],
        "five_factor": [
            "phi_mode",
            "phi_estimator",
            "phi_ridge_lambda",
            "layer_wise_phi_scale",
        ],
        "morphology_aware": [
            "use_path_propagation",
            "path_factor_mode",
            "morphology_modulator_mode",
            "morphology_depth_offset",
            "use_dendritic_normalization",
            "use_branch_type_rules",
            "apical_branch_scale",
            "basal_branch_scale",
            "use_branch_length_modulation",
            "use_branch_role_rules",
            "branch_role_source",
            "specialized_branch_scale",
            "mixed_branch_scale",
            "branch_role_power",
            "branch_role_alignment_weight",
        ],
        "inhibitory_homeostasis": [
            "mode",
            "weight",
            "target_r_tot",
            "target_voltage",
        ],
        "voltage_homeostasis": [
            "weight",
            "target_voltage",
        ],
    }

    moved: list[str] = []

    def move(legacy_name: str, section: dict[str, Any], nested_name: str) -> None:
        value = cfg.pop(legacy_name)
        if nested_name in section and section[nested_name] != value:
            raise ValueError(
                f"Local-rule option {legacy_name}={value!r} conflicts with the "
                f"nested value {section[nested_name]!r}; keep only the nested key"
            )
        section[nested_name] = value
        moved.append(legacy_name)

    for section_name, field_names in legacy_group_fields.items():
        section = ensure_section(section_name)
        for field_name in field_names:
            if field_name in cfg:
                move(field_name, section, field_name)

    hsic_section = ensure_section("hsic")
    hsic_field_map = {
        "hsic_enabled": "enabled",
        "hsic_weight": "weight",
        "hsic_self_weight": "self_weight",
        "hsic_target_weight": "target_weight",
        "hsic_target_source": "target_source",
        "hsic_kernel": "kernel",
        "hsic_sigma": "sigma",
        "hsic_degree": "degree",
        "hsic_coef0": "coef0",
        "hsic_grad_clip_value": "grad_clip_value",
        "hsic_warmup_epochs": "warmup_epochs",
        "hsic_apply_last_layer_only": "apply_last_layer_only",
    }
    for legacy_name, nested_name in hsic_field_map.items():
        if legacy_name in cfg:
            move(legacy_name, hsic_section, nested_name)

    stdp_section = ensure_section("stdp")
    stdp_field_map = {
        "stdp_enabled": "enabled",
        "stdp_apply_to": "apply_to",
        "stdp_activity_mode": "activity_mode",
        "stdp_pre_threshold": "pre_threshold",
        "stdp_post_threshold": "post_threshold",
        "stdp_tau_pre": "tau_pre",
        "stdp_tau_post": "tau_post",
        "stdp_a_plus": "a_plus",
        "stdp_a_minus": "a_minus",
        "stdp_learning_rate_scale": "learning_rate_scale",
        "stdp_inhibitory_update_sign": "inhibitory_update_sign",
        "stdp_use_error_modulation": "use_error_modulation",
        "stdp_error_modulation_mode": "error_modulation_mode",
        "stdp_clamp_update": "clamp_update",
        "stdp_detach_traces": "detach_traces",
    }
    for legacy_name, nested_name in stdp_field_map.items():
        if legacy_name in cfg:
            move(legacy_name, stdp_section, nested_name)

    if moved:
        logger.warning(
            "Flat local-rule options %s are deprecated; write them in their "
            "nested sections (three_factor, four_factor, five_factor, "
            "morphology_aware, hsic, stdp, inhibitory_homeostasis, "
            "voltage_homeostasis).",
            ", ".join(sorted(moved)),
        )
    return cfg


def build_local_rule_config(
    config_dict: dict[str, Any],
    *,
    section: str = "training.main.learning_strategy_config",
) -> LocalRuleConfig:
    """Build LocalRuleConfig from a mapping, rejecting keys it does not read."""

    cfg_kwargs = coerce_legacy_local_rule_config(dict(config_dict))

    # "<rule>_vh" is ordinary LocalCA plus strictly local voltage homeostasis.
    # Explicitly written homeostasis settings take precedence over the alias.
    raw_rule_variant = str(cfg_kwargs.get("rule_variant", "3f")).strip().lower()
    base_rule = LOCAL_RULE_VH_ALIASES.get(raw_rule_variant.replace("-", "_"))
    if base_rule is not None:
        cfg_kwargs["rule_variant"] = base_rule
        vh_cfg = dict(cfg_kwargs.get("voltage_homeostasis") or {})
        vh_cfg.setdefault("enabled", True)
        vh_cfg.setdefault("weight", VOLTAGE_HOMEOSTASIS_ALIAS_WEIGHT)
        vh_cfg.setdefault("target_voltage", VOLTAGE_HOMEOSTASIS_ALIAS_TARGET)
        cfg_kwargs["voltage_homeostasis"] = vh_cfg

    return build_section(LocalRuleConfig, cfg_kwargs, section=section)


__all__ = [
    "build_local_rule_config",
    "coerce_legacy_local_rule_config",
    "section_to_dict",
]
