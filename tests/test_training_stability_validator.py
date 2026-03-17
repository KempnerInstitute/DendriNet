"""Tests for training stability safeguards and auto-fixes."""

from types import SimpleNamespace

import dendritic_modeling.utils.training_stability_validator as tsv
from dendritic_modeling.utils.training_stability_validator import (
    apply_automatic_safeguards,
    suggest_optimal_config_for_model_size,
    validate_training_stability,
)


class ConfigNode(SimpleNamespace):
    """Simple attribute-based config object with dict-like get support."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _build_config(
    *,
    use_amp=False,
    enable_hooks=True,
    distributed_mode="none",
    fsdp_config=None,
    learning_rate=0.001,
    grad_clip_value=None,
    topk_ultrafast=False,
    efficient_blocklinear=False,
    reactivation_strategy="none",
    topk_strategy="none",
    blocklinear_strategy="none",
):
    architecture = ConfigNode(
        topk_ultrafast=topk_ultrafast,
        efficient_blocklinear=efficient_blocklinear,
        reactivation_strategy=reactivation_strategy,
        topk_strategy=topk_strategy,
        blocklinear_strategy=blocklinear_strategy,
    )
    core_network_params = ConfigNode(
        reactivation_strategy=reactivation_strategy,
        topk_strategy=topk_strategy,
        blocklinear_strategy=blocklinear_strategy,
    )

    return ConfigNode(
        experiment=ConfigNode(
            enable_hooks=enable_hooks,
        ),
        training=ConfigNode(
            main=ConfigNode(
                common=ConfigNode(
                    use_amp=use_amp,
                    grad_clip_value=grad_clip_value,
                ),
                param_groups=ConfigNode(lr=learning_rate),
            ),
        ),
        distributed=ConfigNode(
            mode=distributed_mode,
            fsdp=fsdp_config or ConfigNode(),
        ),
        model=ConfigNode(
            core=ConfigNode(architecture=architecture),
            core_network=ConfigNode(parameters=core_network_params),
        ),
    )


def test_validate_training_stability_logs_expected_warnings(monkeypatch):
    config = _build_config(
        use_amp=True,
        enable_hooks=True,
        distributed_mode="fsdp",
        fsdp_config=ConfigNode(
            sharding_strategy="full",
            reduce_communication_overhead=False,
        ),
        learning_rate=0.02,
        grad_clip_value=20.0,
        topk_ultrafast=True,
        reactivation_strategy="scaled",
    )

    warnings = []
    monkeypatch.setattr(
        tsv.logger,
        "warning",
        lambda message: warnings.append(message),
    )

    assert validate_training_stability(config) is True

    messages = "\n".join(warnings)
    assert "AMP + Gradient Hooks" in messages
    assert "High NaN Risk" in messages
    assert "FSDP Performance" in messages
    assert "High Gradient Clipping" in messages


def test_validate_training_stability_warns_on_memory_optimization_without_amp(
    monkeypatch,
):
    config = _build_config(efficient_blocklinear=True, use_amp=False)

    warnings = []
    monkeypatch.setattr(
        tsv.logger,
        "warning",
        lambda message: warnings.append(message),
    )

    assert validate_training_stability(config) is True
    assert "Memory Optimization" in "\n".join(warnings)


def test_validate_training_stability_handles_missing_sections_gracefully():
    config = ConfigNode()
    assert validate_training_stability(config) is True


def test_suggest_optimal_config_for_model_size_ranges():
    small = suggest_optimal_config_for_model_size(9_999_999)
    assert small["training.main.common.use_amp"] is False
    assert small["distributed.mode"] == "none"
    assert small["experiment.enable_profiling"] is True

    medium = suggest_optimal_config_for_model_size(50_000_000)
    assert medium["training.main.common.use_amp"] is True
    assert medium["distributed.mode"] == "ddp"
    assert medium["experiment.enable_hooks"] is True

    large = suggest_optimal_config_for_model_size(100_000_000)
    assert large["training.main.common.use_amp"] is True
    assert large["distributed.mode"] == "fsdp"
    assert large["experiment.enable_hooks"] is False
    assert large["distributed.fsdp.sharding_strategy"] == "HYBRID_SHARD"


def test_apply_automatic_safeguards_sets_flags_for_amp_with_gradient_scaling(
    monkeypatch,
):
    config = _build_config(
        use_amp=True,
        topk_strategy="scaled",
    )
    info_messages = []
    monkeypatch.setattr(
        tsv.logger,
        "info",
        lambda message: info_messages.append(message),
    )

    apply_automatic_safeguards(config)

    common = config.training.main.common
    assert common.enable_adaptive_clipping is True
    assert common.disable_hooks_with_amp is True
    assert "Auto-enabled adaptive gradient clipping" in "\n".join(info_messages)


def test_apply_automatic_safeguards_does_not_override_existing_values():
    config = _build_config(use_amp=True, topk_strategy="scaled")
    config.training.main.common.enable_adaptive_clipping = False
    config.training.main.common.disable_hooks_with_amp = False

    apply_automatic_safeguards(config)

    assert config.training.main.common.enable_adaptive_clipping is False
    assert config.training.main.common.disable_hooks_with_amp is False


def test_apply_automatic_safeguards_handles_missing_nested_config():
    config = ConfigNode(training=ConfigNode())
    apply_automatic_safeguards(config)
