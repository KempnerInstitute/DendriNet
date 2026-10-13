"""Tests for the training stability validator."""

from types import SimpleNamespace

import dendritic_modeling.utils.training_stability_validator as tsv
from dendritic_modeling.utils.training_stability_validator import (
    validate_training_stability,
)


class ConfigNode(SimpleNamespace):
    """Simple attribute-based config object with dict-like get support."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _build_config(
    *,
    use_amp=False,
    distributed_mode="none",
    fsdp_config=None,
    learning_rate=0.001,
    grad_clip_value=None,
    sparsity_type="standard",
    ultrafast=True,
    efficient_blocklinear=False,
    reactivation_strategy="none",
    topk_strategy="none",
    blocklinear_strategy="none",
    core_type="einet",
    morphology_use_shunting=True,
    weight_transform="softplus",
    normalize_inputs=False,
    transfer_output_activation=None,
    dataset_name="synthetic",
    dataset_params=None,
):
    return ConfigNode(
        experiment=ConfigNode(),
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
            core=ConfigNode(
                type=core_type,
                sparsity=ConfigNode(
                    type=sparsity_type,
                    ultrafast=ultrafast,
                    gradient_scaling=topk_strategy,
                ),
                reactivation=ConfigNode(gradient_scaling=reactivation_strategy),
                blocklinear=ConfigNode(
                    gradient_scaling=blocklinear_strategy,
                    efficient=efficient_blocklinear,
                ),
                morphology=ConfigNode(
                    use_shunting=morphology_use_shunting,
                    weight_transform=weight_transform,
                ),
                transfer=ConfigNode(output_activation=transfer_output_activation),
            ),
        ),
        data=ConfigNode(
            dataset_name=dataset_name,
            processing=ConfigNode(normalize=normalize_inputs),
            dataset_params=dataset_params or ConfigNode(),
        ),
    )


def test_training_stability_context_reads_the_typed_paths():
    fsdp_config = ConfigNode(sharding_strategy="full", forward_prefetch=False)
    config = _build_config(
        use_amp=True,
        distributed_mode="fsdp",
        fsdp_config=fsdp_config,
        learning_rate=0.02,
        grad_clip_value=20.0,
        sparsity_type="stochastic",
        reactivation_strategy="straight_through",
        topk_strategy="conductance_dynamic",
        blocklinear_strategy="distal_upweight_by_idx",
        core_type="dendritic_shunting",
        morphology_use_shunting=True,
        weight_transform="softplus",
        normalize_inputs=False,
        transfer_output_activation="relu",
        dataset_name="stringer_v1",
        dataset_params=ConfigNode(stringer_v1=ConfigNode(normalize=True)),
    )

    context = tsv._collect_training_stability_context(config)

    assert context.enable_amp is True
    assert context.topk_ultrafast is True
    assert context.reactivation_strategy == "straight_through"
    assert context.topk_strategy == "conductance_dynamic"
    assert context.blocklinear_strategy == "distal_upweight_by_idx"
    assert context.effective_use_shunting is True
    assert context.weight_transform == "softplus"
    assert context.transfer_output_activation == "relu"
    assert context.normalize_inputs is True
    assert context.learning_rate == 0.02
    assert context.dist_mode == "fsdp"
    assert context.fsdp_config is fsdp_config
    assert context.trainer_params.grad_clip_value == 20.0


def test_validate_training_stability_logs_expected_warnings(monkeypatch):
    config = _build_config(
        use_amp=True,
        distributed_mode="fsdp",
        fsdp_config=ConfigNode(sharding_strategy="full", limit_all_gathers=False),
        learning_rate=0.02,
        grad_clip_value=20.0,
        sparsity_type="stochastic",
        reactivation_strategy="straight_through",
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
    assert "distributed.fsdp.limit_all_gathers off" in messages
    assert "adaptive" not in messages  # no such clipping mode
    assert "High Gradient Clipping" in messages


def test_ultrafast_counts_only_for_stochastic_topk():
    standard = tsv._collect_training_stability_context(_build_config())
    assert standard.topk_ultrafast is False
    stochastic = tsv._collect_training_stability_context(
        _build_config(sparsity_type="stochastic")
    )
    assert stochastic.topk_ultrafast is True


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


def test_validate_training_stability_rejects_signed_shunting_input(monkeypatch):
    config = _build_config(
        core_type="dendritic_shunting",
        normalize_inputs=True,
        weight_transform="softplus",
        transfer_output_activation=None,
    )

    errors = []
    monkeypatch.setattr(tsv.logger, "error", lambda message: errors.append(message))

    assert validate_training_stability(config) is False
    assert "Signed shunting input" in "\n".join(errors)


def test_validate_training_stability_rejects_nested_stringer_normalize(monkeypatch):
    config = _build_config(
        core_type="dendritic_shunting",
        normalize_inputs=False,
        weight_transform="softplus",
        transfer_output_activation=None,
        dataset_name="stringer_v1",
        dataset_params=ConfigNode(stringer_v1=ConfigNode(normalize=True)),
    )

    errors = []
    monkeypatch.setattr(tsv.logger, "error", lambda message: errors.append(message))

    assert validate_training_stability(config) is False
    assert "Signed shunting input" in "\n".join(errors)


def test_validate_training_stability_unwraps_sweep_base_config(monkeypatch):
    config = ConfigNode(
        sweep_config=ConfigNode(),
        base_config=_build_config(
            core_type="dendritic_shunting",
            normalize_inputs=True,
            weight_transform="softplus",
            transfer_output_activation=None,
        ),
    )

    errors = []
    monkeypatch.setattr(tsv.logger, "error", lambda message: errors.append(message))

    assert validate_training_stability(config) is False
    assert "Signed shunting input" in "\n".join(errors)


def test_validate_training_stability_allows_relu_clamped_shunting_input(monkeypatch):
    config = _build_config(
        core_type="dendritic_shunting",
        normalize_inputs=True,
        weight_transform="softplus",
        transfer_output_activation="relu",
    )

    errors = []
    monkeypatch.setattr(tsv.logger, "error", lambda message: errors.append(message))

    assert validate_training_stability(config) is True
    assert errors == []
