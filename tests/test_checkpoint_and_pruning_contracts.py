"""Checkpoint precision, optimizer groups and configured pruning behavior."""

import pytest
import torch
from torch import nn

from dendritic_modeling.config.regularization import PruningConfig, SelectiveConfig
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_reactivation import (
    _analytical_reactivation_prior,
    _apply_reactivation_init,
)
from dendritic_modeling.networks.utils.sparse_maintenance import restrict_maintenance_to
from dendritic_modeling.training.optimizers.muonh import MuonH
from dendritic_modeling.training.strategies.standard import Trainer


def test_explicit_ell_backend_rejects_overflow_before_launch(monkeypatch):
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
        backend,
    )

    monkeypatch.setattr(
        backend, "indexed_projection_backend_available", lambda *a, **k: True
    )
    with pytest.raises(ValueError, match="int32"):
        backend.resolve_indexed_projection_backend(
            "triton_ell",
            device="cuda",
            batch_size=16384,
            in_features=64,
            out_features=140000,
            synapses_per_output=4,
        )


def test_analytical_calibration_prior_survives_checkpoint_reload():
    source, restored = [
        DendriticBranchLayer(
            output_dim=3,
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=2,
            reactivate=True,
            reactivation_type="param_tanh",
        )
        for _ in range(2)
    ]
    _apply_reactivation_init(source, 4.23456789123, 0.173456789123)
    _apply_reactivation_init(restored, 2.0, 0.5)
    # The learned gate no longer determines the immutable calibration prior.
    source.reactivation.initialize(6.0, 0.9)
    restored.load_state_dict(source.half().state_dict(), strict=True)
    assert _analytical_reactivation_prior(restored) == _analytical_reactivation_prior(
        source
    )


def test_muon_parameter_groups_keep_rates_and_match_separate_optimizers():
    torch.manual_seed(17)
    params = [nn.Parameter(torch.randn(4, 3)), nn.Parameter(torch.randn(3))]
    reference = [nn.Parameter(p.detach().clone()) for p in params]
    rates = [0.02, 0.003]
    grouped = MuonH(
        [
            {"params": [p], "lr": lr, "name": f"group{i}"}
            for i, (p, lr) in enumerate(zip(params, rates))
        ],
        lr=0.01,
    )
    separate = [MuonH([p], lr=lr) for p, lr in zip(reference, rates)]
    for step in range(3):
        for p, q in zip(params, reference):
            grad = torch.full_like(p, (step + 1) * 0.1)
            p.grad, q.grad = grad.clone(), grad.clone()
        grouped.step()
        for opt in separate:
            opt.step()
    for p, q in zip(params, reference):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert [group["lr"] for group in grouped.param_groups] == rates
    assert [group["name"] for group in grouped.param_groups] == ["group0", "group1"]


def test_legacy_prior_checkpoint_load_and_corrupt_current_checkpoint():
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=8,
        excitatory_synapses_per_branch=2,
        reactivate=True,
        reactivation_type="param_tanh",
    )
    _apply_reactivation_init(layer, 2.0, 0.5)
    original_prior = _analytical_reactivation_prior(layer)
    state = layer.state_dict()
    state.pop("reactivation_analytical_init_m")
    state.pop("reactivation_analytical_init_b")
    with pytest.raises(RuntimeError, match="reactivation_analytical_init"):
        layer.load_state_dict(state, strict=True)
    state._metadata[""]["version"] = 1
    layer.load_state_dict(state, strict=True)
    assert _analytical_reactivation_prior(layer) == original_prior


def test_maintenance_rejects_flattened_optimizer_ownership():
    parameter = nn.Parameter(torch.ones(4))
    parameter._is_flat_param = True
    with pytest.raises(ValueError, match="flattened"):
        with restrict_maintenance_to([parameter]):
            pass


def _pruning_trainer(tmp_path, **options):
    model = nn.Sequential(nn.Linear(2, 2))
    trainer = Trainer(
        torch.optim.SGD(model.parameters(), lr=0.1),
        pruning_config=PruningConfig(
            enabled=True,
            threshold=1e5,
            selective=SelectiveConfig(weight_types=["all"]),
            **options,
        ),
        save_path=str(tmp_path),
        num_workers=0,
    )
    trainer.device = "cpu"
    data = torch.utils.data.TensorDataset(torch.ones(4, 2), torch.zeros(4, 2))
    return model, trainer, data


def test_pruning_evaluation_flag_is_honored(tmp_path, monkeypatch):
    model, trainer, data = _pruning_trainer(tmp_path, re_evaluate_after=False)
    calls = []
    monkeypatch.setattr(
        trainer, "_evaluate_loader_loss", lambda *a: calls.append(1) or 0.0
    )
    result = trainer._apply_pruning_and_evaluate(model, data, data)
    assert calls == []
    assert "post_pruning_performance" not in result
    assert result["pruning_stats"]["pruned_params"] > 0


def test_pruning_report_and_file_are_produced_when_requested(tmp_path, capsys):
    model, trainer, data = _pruning_trainer(
        tmp_path,
        re_evaluate_after=False,
        report_non_pruned=True,
        save_pruning_stats=True,
    )
    trainer._apply_pruning_and_evaluate(model, data, data)
    assert (tmp_path / "pruning_stats.json").is_file()
    assert "WEIGHT PRUNING REPORT" in capsys.readouterr().out


def test_pruning_reanalysis_fails_explicitly_before_training(tmp_path):
    with pytest.raises(ValueError, match="redo_analysis_after_pruning"):
        _pruning_trainer(tmp_path, redo_analysis_after_pruning=True)
