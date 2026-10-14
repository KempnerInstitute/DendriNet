"""Causal isolation and TRAIN-only accounting for affine logit rescaling."""

import pytest
import torch

from dendritic_modeling.networks.benchmarks.parameter_budget import (
    build_model,
    model_report,
)
from dendritic_modeling.training.readout_initialization import initialize_readout


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "family",
    [
        "dense",
        "sparse",
        "dendritic_signed",
        "dendritic_additive",
        "dendritic_shunting",
        "grouped_gated",
    ],
)
def test_rescaling_preserves_hidden_state_support_count_and_rng(device, family):
    torch.set_num_threads(1)
    model = build_model(
        {
            "family": family,
            "input_dim": 8,
            "output_dim": 3,
            "width": 8,
            "contacts_e": 4,
            "contacts_i": 0 if family == "dendritic_signed" else 2,
            "branch_factors": [2, 2],
            "network_depth": 2,
            "seed": 111,
        }
    ).to(device)
    x = torch.randn(31, 8, generator=torch.Generator().manual_seed(17))
    before = model(x.to(device)).detach()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    report = model_report(model)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    audit = initialize_readout(
        model, x, {"mode": "train_batch_rms", "target_rms": 0.1, "examples": 16}
    )
    after = model(x.to(device)).detach()
    assert audit["achieved_rms"] == pytest.approx(0.1, rel=2e-5)
    torch.testing.assert_close(after, before * audit["scale"], rtol=2e-5, atol=1e-7)
    for name, value in model.state_dict().items():
        if not name.startswith("readout."):
            assert torch.equal(value, state[name]), name
    assert model_report(model)["total_parameters"] == report["total_parameters"]
    assert model_report(model)["topology_sha256"] == report["topology_sha256"]
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert model.training and audit["training_examples_exposed"] == 16
    assert not audit["labels_used"] and not audit["validation_used"]


@pytest.mark.parametrize(
    "spec",
    [
        {"mode": "train_batch_rms", "target_rms": 0, "examples": 2},
        {"mode": "train_batch_rms", "target_rms": 0.1, "examples": 99},
        {"mode": "train_batch_rms", "target_rms": 0.1, "examples": True},
        {"mode": "preserve", "target_rms": 0.1},
    ],
)
def test_invalid_interventions_are_rejected_before_mutation(spec):
    model = build_model({"family": "dense", "input_dim": 4, "output_dim": 2})
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        initialize_readout(model, torch.ones(3, 4), spec)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


def test_zero_logits_fail_without_changing_parameters_or_mode():
    model = build_model({"family": "dense", "input_dim": 4, "output_dim": 2})
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
    model.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="Zero logits"):
        initialize_readout(
            model,
            torch.ones(3, 4),
            {"mode": "train_batch_rms", "target_rms": 0.1, "examples": 2},
        )
    assert not model.training
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "family", ["dense", "dendritic_additive", "dendritic_shunting"]
)
def test_cold_initialization_without_prior_forward(device, family):
    torch.set_num_threads(1)
    model = build_model(
        {
            "family": family,
            "input_dim": 8,
            "output_dim": 3,
            "width": 8,
            "contacts_e": 4,
            "contacts_i": 2,
            "branch_factors": [2, 2],
            "network_depth": 2,
        }
    ).to(device)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    result = initialize_readout(
        model,
        torch.ones(16, 8),
        {"mode": "train_batch_rms", "target_rms": 0.1, "examples": 16},
    )
    assert result["hidden_state_sha256_before"] == result["hidden_state_sha256_after"]
    assert result["initialization_examples_processed"] == 32
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


def test_individual_module_modes_are_preserved():
    model = build_model({"family": "dense", "input_dim": 4, "output_dim": 2})
    model.train()
    model.hidden.eval()
    modes = {name: module.training for name, module in model.named_modules()}
    initialize_readout(
        model,
        torch.ones(3, 4),
        {"mode": "train_batch_rms", "target_rms": 0.1, "examples": 2},
    )
    assert {name: module.training for name, module in model.named_modules()} == modes


def test_guard_rolls_back_unexpected_hidden_state_mutation():
    model = build_model({"family": "dense", "input_dim": 4, "output_dim": 2})
    model.hidden.register_buffer("unexpected_counter", torch.zeros(()))
    state = {name: value.clone() for name, value in model.state_dict().items()}

    def mutate(module, _inputs, _output):
        module.unexpected_counter.add_(1)

    handle = model.hidden.register_forward_hook(mutate)
    try:
        with pytest.raises(RuntimeError, match="hidden state"):
            initialize_readout(
                model,
                torch.ones(3, 4),
                {"mode": "train_batch_rms", "target_rms": 0.1, "examples": 2},
            )
    finally:
        handle.remove()
    assert all(
        torch.equal(value, state[name]) for name, value in model.state_dict().items()
    )
