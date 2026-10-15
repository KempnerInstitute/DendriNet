"""Level lesions and OOD-delay evaluation rebuild the run and report soma depths."""

from __future__ import annotations

import json
import sys

import pytest
import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.analysis import (
    analyze_routing_by_condition,
    evaluate_ood_delay,
    lesion_analysis,
)
from dendritic_modeling.scripts.analysis.analyze_routing_by_condition import (
    load_model_and_config,
)
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment


def _population(n_neurons, branch_factors):
    return {
        "n_neurons": n_neurons,
        "branch_factors": branch_factors,
        "ff_excitatory_synapses": 2,
        "rec_excitatory_synapses": 2,
        "topk_type": "standard",
    }


@pytest.fixture
def trained_run(tmp_path):
    payload = {
        "experiment": {"seed": 3, "train_valid_split": 0.75},
        "data": {
            "dataset_name": "variable_delay_dms",
            "base_dir": "",
            "processing": {"flatten": False, "normalize": False},
            "dataset_params": {
                "variable_delay_dms": {
                    "n_samples": 48,
                    "stimulus_dim": 4,
                    "stimulus_duration": 2,
                    "min_delay": 2,
                    "max_delay": 6,
                    "probe_duration": 2,
                }
            },
        },
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": 4}},
            "core": {
                "biological_neuron": False,
                "type": "unified_ei",
                "unified_ei": {
                    "output_mode": "last",
                    "layers": [
                        {
                            "recurrent": True,
                            "dt": 1.0,
                            "excitatory": _population(4, [2]),
                            "inhibitory": _population(2, []),
                        }
                    ],
                },
            },
            "decoder": {"type": "MLP", "params": {"hidden_dims": [], "output_dim": 2}},
        },
        "training": {
            "main": {
                "strategy": "recurrent",
                "optimizer": {"name": "adam", "lr": 0.001},
                "common": {"epochs": 1, "batch_size": 8, "num_workers": 0},
            }
        },
        "analysis": {},
        "outputs": {"run_name": "run", "results_dir": str(tmp_path / "train")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(str(config_path))
    _run, _train, _valid, _test, model, _encoder = setup_environment(
        model_config=config.model,
        training_config=config.training,
        data_config=config.data,
        outputs_config=config.outputs,
        experiment_config=config.experiment,
    )
    model_path = tmp_path / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    return model_path, config_path


def test_the_loader_rebuilds_the_runs_test_set(trained_run, tmp_path):
    model_path, config_path = trained_run
    first = load_model_and_config(
        str(model_path), str(config_path), output_dir=str(tmp_path / "a")
    )
    second = load_model_and_config(
        str(model_path), str(config_path), output_dir=str(tmp_path / "b")
    )
    assert torch.equal(first[2][0][0], second[2][0][0])
    assert (tmp_path / "a" / "resolved_seeds.json").exists()


def test_a_level_lesion_silences_and_restores_by_soma_depth(trained_run, tmp_path):
    model_path, config_path = trained_run
    model, _config, test_ds = load_model_and_config(
        str(model_path), str(config_path), output_dir=str(tmp_path / "out")
    )
    depths = [depth for depth, _tau in lesion_analysis.level_depths_and_taus(model)]
    assert sorted(depths) == [0, 1]
    x = test_ds[0][0].unsqueeze(0)
    with torch.no_grad():
        intact = model(x)
        with lesion_analysis.lesion_level(model, depth=0):
            silenced = model(x)
        restored = model(x)
    assert not torch.equal(intact, silenced)
    assert torch.equal(intact, restored)
    with pytest.raises(ValueError, match="no dendritic level at depth 5"):
        with lesion_analysis.lesion_level(model, depth=5):
            pass


def test_the_lesion_script_writes_results_and_both_figures(
    trained_run, tmp_path, monkeypatch
):
    model_path, config_path = trained_run
    out = tmp_path / "lesion"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lesion_analysis",
            "--model_path",
            str(model_path),
            "--config_path",
            str(config_path),
            "--output_dir",
            str(out),
            "--n_bins",
            "4",
        ],
    )
    lesion_analysis.main()
    results = json.loads((out / "lesion_results.json").read_text())
    assert {"intact", *(f for f in results if f.startswith("Lesion depth"))} == set(
        results
    )
    assert len(results) == 3
    assert (out / "lesion_accuracy_vs_delay.png").exists()


def test_ood_delay_trials_are_seeded_and_shaped_like_training(
    trained_run, tmp_path, monkeypatch
):
    model_path, config_path = trained_run
    outputs = []
    for name in ("first", "second"):
        out = tmp_path / name
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_ood_delay",
                "--model_path",
                str(model_path),
                "--config_path",
                str(config_path),
                "--output_dir",
                str(out),
                "--min_delay",
                "8",
                "--max_delay",
                "12",
                "--n_samples",
                "32",
                "--n_bins",
                "2",
            ],
        )
        evaluate_ood_delay.main()
        outputs.append(json.loads((out / "ood_delay_metrics.json").read_text()))
    assert outputs[0] == outputs[1]
    assert outputs[0]["trial_shape"] == {
        "stimulus_dim": 4,
        "stimulus_duration": 2,
        "probe_duration": 2,
    }


def test_the_routing_script_writes_results_and_labels_levels_by_depth(
    trained_run, tmp_path, monkeypatch
):
    model_path, config_path = trained_run
    out = tmp_path / "routing"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_routing_by_condition",
            "--model_path",
            str(model_path),
            "--config_path",
            str(config_path),
            "--output_dir",
            str(out),
            "--n_samples",
            "24",
            "--n_bins",
            "3",
        ],
    )
    analyze_routing_by_condition.main()
    results = json.loads((out / "routing_by_condition.json").read_text())
    assert results["n_samples"] == 24
    assert results["binned_results"]
    for bin_result in results["binned_results"].values():
        assert len(bin_result["mean_beta"]) == 2
    assert (out / "routing_heatmap.png").exists()
    assert (out / "beta_vs_condition.png").exists()

    model, _config, _test = load_model_and_config(
        str(model_path), str(config_path), output_dir=str(tmp_path / "labels")
    )
    labels = analyze_routing_by_condition.routing_level_labels(model)
    # Level 0 is the most distal level; the last level is the soma.
    assert [label.split(",")[0] for label in labels] == ["L0 (depth 1", "L1 (depth 0"]
    assert analyze_routing_by_condition.routing_level_labels(model, "layer_5") is None


def test_the_level_plot_names_each_level_without_colour(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    binned = {
        float(delay): {"mean_beta": [0.5, 0.3, 0.2], "temporal_beta": [[0.5, 0.3, 0.2]]}
        for delay in (2, 4, 6)
    }
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    analyze_routing_by_condition.plot_beta_vs_condition(
        binned, {}, str(tmp_path / "beta.png")
    )
    lines = plt.gcf().axes[0].get_lines()
    shapes = {(line.get_marker(), line.get_linestyle()) for line in lines}
    assert len(lines) == 3 and len(shapes) == 3
    monkeypatch.undo()
    plt.close("all")
