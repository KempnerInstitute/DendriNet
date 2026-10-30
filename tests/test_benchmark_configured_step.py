"""The configuration benchmark must run actual updates and reject omissions."""

from pathlib import Path

import pytest
import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.config.regularization import RegularizationConfig
from dendritic_modeling.scripts.benchmarks.benchmark_configured_step import (
    build_step,
    measure_step,
)


def small_config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        str(root / "configs/examples/contextual_remapping_affine_mlp.yaml")
    )
    config.model.core.affine_mlp.dimensions = [8, 6, 3]
    config.model.core.affine_mlp.dropout = 0
    config.training.main.common.batch_size = 4
    return config


def test_updates_are_finite_change_parameters_and_repeat():
    config = small_config()
    kwargs = {
        "input_dim": 8,
        "classes": 3,
        "device": torch.device("cpu"),
        "steps_per_epoch": 4,
    }
    model, _, step, _ = build_step(config, **kwargs)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = measure_step(step, device=torch.device("cpu"), warmup=1, steps=2)
    after = {k: v.clone() for k, v in model.state_dict().items()}
    assert any(not torch.equal(before[k], v) for k, v in after.items())
    assert result["step_ms_mean"] > 0
    assert result["peak_allocated_mb"] is None
    model2, _, step2, _ = build_step(small_config(), **kwargs)
    for _ in range(3):
        step2()
    for k, v in model2.state_dict().items():
        torch.testing.assert_close(v, after[k], rtol=0, atol=0)


@pytest.mark.parametrize("option", ["regularization", "amp", "strategy"])
def test_reject_unsupported_training_semantics(option):
    config = small_config()
    if option == "regularization":
        config.training.main.regularization.l1_weight = 0.1
    elif option == "amp":
        config.training.main.common.use_amp = True
    else:
        config.training.main.strategy = "multi_stage"
    with pytest.raises(ValueError):
        build_step(
            config,
            input_dim=8,
            classes=3,
            device=torch.device("cpu"),
            steps_per_epoch=4,
        )


def test_legacy_mapping_regularization_and_pruning_are_supported():
    config = small_config()
    config.training.main.regularization = {"l1_weight": 0, "l2_weight": 0}
    config.training.main.pruning = {"enabled": False}
    _, _, step, _ = build_step(
        config, input_dim=8, classes=3, device=torch.device("cpu"), steps_per_epoch=4
    )
    assert torch.isfinite(torch.tensor(step()))
    config.training.main.regularization["l1_weight"] = 1
    with pytest.raises(ValueError, match="regularization"):
        build_step(
            config,
            input_dim=8,
            classes=3,
            device=torch.device("cpu"),
            steps_per_epoch=4,
        )


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize(
    "penalty",
    [
        {"l1_weight": 0.1},
        {"l2_weight": 0.1},
        {"routing_entropy_weight": 0.1},
        {"routing_balance_weight": 0.1},
        {"enforce_ei_weight_ratio": True},
        {"split_params": True, "param_group_weights": {"decoder": {"l2_weight": 0.1}}},
    ],
)
def test_all_enabled_penalties_rejected_before_initialization(penalty, as_mapping):
    config = small_config()
    config.training.main.regularization = (
        penalty if as_mapping else RegularizationConfig(**penalty)
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="additional regularization"):
        build_step(
            config,
            input_dim=8,
            classes=3,
            device=torch.device("cpu"),
            steps_per_epoch=4,
        )
    assert torch.equal(rng, torch.get_rng_state())
