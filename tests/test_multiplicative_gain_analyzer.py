import contextlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools import multiplicative_gain as mg
from dendritic_modeling.config import MultiplicativeGainAnalysisParams
from dendritic_modeling.datasets.standard_datasets import (
    ContextualGainDataset,
    PoissonGeneratorDataset,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity


@pytest.fixture(autouse=True)
def _seed_rngs():
    torch.manual_seed(0)


def _make_classifier(input_dim: int = 2) -> Classifier:
    return Classifier(
        encoder_network=Identity(input_dim=input_dim),
        core_network=Identity(input_dim=input_dim),
        decoder_network=Identity(input_dim=input_dim),
    )


def test_apply_gain_preprocess_modes_and_random_gain(monkeypatch):
    base_inputs = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float32)
    dataset = TensorDataset(base_inputs, torch.tensor([0, 1]))

    expectations = {
        "none": base_inputs * 2.0,
        "identity": base_inputs * 2.0,
        "relu": base_inputs.clamp_min(0.0) * 2.0,
        "softplus": torch.nn.functional.softplus(base_inputs) * 2.0,
        "shift_min": (base_inputs - base_inputs.amin(dim=0, keepdim=True)) * 2.0,
        "global_shift_min": (base_inputs - base_inputs.amin()) * 2.0,
    }

    for preprocess, expected in expectations.items():
        analyzer = mg.MultiplicativeGainAnalyzer(
            MultiplicativeGainAnalysisParams(
                input_preprocess=preprocess,
                noise_model="none",
            )
        )
        scaled, _labels = analyzer._apply_gain_to_inputs(
            dataset=dataset,
            gain_factor=2.0,
            base_inputs=base_inputs,
        )
        assert torch.allclose(scaled, expected)

    analyzer = mg.MultiplicativeGainAnalyzer(
        MultiplicativeGainAnalysisParams(
            uniform_gain=False,
            input_preprocess="relu",
            noise_model="none",
        )
    )
    monkeypatch.setattr(mg.torch, "rand_like", lambda x: torch.full_like(x, 0.25))
    scaled, _labels = analyzer._apply_gain_to_inputs(
        dataset=dataset,
        gain_factor=5.0,
        base_inputs=base_inputs,
    )
    expected_gain = (0.25 * (5.0 - 1.0)) + 1.0
    assert torch.allclose(scaled, base_inputs.clamp_min(0.0) * expected_gain)


def test_apply_gain_noise_validation_and_poisson_dataset_branch(monkeypatch):
    dataset = TensorDataset(
        torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float32),
        torch.tensor([0, 1]),
    )
    analyzer = mg.MultiplicativeGainAnalyzer(
        MultiplicativeGainAnalysisParams(noise_model="poisson", poisson_eps=0.125)
    )
    seen = {}

    def fake_poisson(x):
        seen["arg"] = x.clone()
        return x + 1.0

    monkeypatch.setattr(mg.torch, "poisson", fake_poisson)
    noisy, _labels = analyzer._apply_gain_to_inputs(
        dataset=dataset,
        gain_factor=1.0,
        base_inputs=dataset.tensors[0],
    )
    assert torch.equal(noisy, seen["arg"] + 1.0)
    assert float(seen["arg"].min()) >= 0.125

    with pytest.raises(ValueError, match="input_preprocess"):
        mg.MultiplicativeGainAnalyzer(
            MultiplicativeGainAnalysisParams(input_preprocess="bad-mode")
        )._apply_gain_to_inputs(dataset, 1.0, base_inputs=dataset.tensors[0])

    with pytest.raises(ValueError, match="noise_model"):
        mg.MultiplicativeGainAnalyzer(
            MultiplicativeGainAnalysisParams(noise_model="bad-noise")
        )._apply_gain_to_inputs(dataset, 1.0, base_inputs=dataset.tensors[0])

    poisson_dataset = PoissonGeneratorDataset(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
        multiplicative_gain=False,
        poisson_sampling=False,
        stimulus_duration=1.0,
    )
    materialized = (torch.full((2, 2), 7.0), torch.tensor([0, 1]))
    monkeypatch.setattr(mg, "materialize_dataset", lambda *args, **kwargs: materialized)

    fixed_calls = []
    monkeypatch.setattr(
        poisson_dataset,
        "_set_gain_factor",
        lambda gain_factor: fixed_calls.append(("fixed", gain_factor)),
    )
    scaled, scaled_labels = analyzer._apply_gain_to_inputs(
        poisson_dataset, gain_factor=3.0
    )
    assert fixed_calls == [("fixed", 3.0)]
    assert torch.equal(scaled, materialized[0])
    assert torch.equal(scaled_labels, materialized[1])

    max_calls = []
    analyzer.uniform_gain = False
    monkeypatch.setattr(
        poisson_dataset,
        "_set_max_gain_factor",
        lambda gain_factor: max_calls.append(("max", gain_factor)),
    )
    scaled, scaled_labels = analyzer._apply_gain_to_inputs(
        poisson_dataset, gain_factor=5.0
    )
    assert max_calls == [("max", 5.0)]
    assert torch.equal(scaled, materialized[0])
    assert torch.equal(scaled_labels, materialized[1])


def test_analyze_collects_metrics_and_respects_model_type(tmp_path, monkeypatch):
    base_inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    dataset = TensorDataset(base_inputs, labels)
    model = _make_classifier(input_dim=2)

    params = MultiplicativeGainAnalysisParams(
        gain_min=0.5,
        gain_max=2.0,
        n_steps=3,
        logspace=True,
        accuracy=True,
        auc=True,
        categorical_loglikelihood=True,
        mse=True,
        cosine_similarity=True,
    )
    analyzer = mg.MultiplicativeGainAnalyzer(params)

    monkeypatch.setattr(
        mg,
        "analysis_device_context",
        lambda model, device: contextlib.nullcontext("cpu"),
    )
    monkeypatch.setattr(
        mg,
        "materialize_dataset",
        lambda *args, **kwargs: (base_inputs.clone(), labels.clone()),
    )
    monkeypatch.setattr(mg, "evaluation_kwargs_from_runtime", lambda runtime: {})
    monkeypatch.setattr(mg, "evaluate_accuracy", lambda *args, **kwargs: ("ok", 0.7))
    monkeypatch.setattr(mg, "evaluate_auc", lambda *args, **kwargs: ("ok", 0.8))
    monkeypatch.setattr(
        mg,
        "evaluate_categorical_loglikelihood",
        lambda *args, **kwargs: ("ok", -1.2),
    )
    monkeypatch.setattr(mg, "evaluate_mse", lambda *args, **kwargs: ("ok", 1.5))
    monkeypatch.setattr(
        mg,
        "evaluate_cosine_similarity",
        lambda *args, **kwargs: ("ok", 0.3),
    )

    plotted = {"final": [], "training": []}
    monkeypatch.setattr(
        mg,
        "plot_performance_vs_gain_final",
        lambda data_dict, metric, logspace, save_path: plotted["final"].append(
            (metric, logspace, Path(save_path).name)
        ),
    )
    monkeypatch.setattr(
        mg,
        "plot_performance_vs_gain_training",
        lambda epochs_data, epoch_numbers, metric, logspace, save_path: plotted[
            "training"
        ].append((metric, tuple(epoch_numbers), logspace)),
    )

    epochs_dir = tmp_path / "epochs"
    epochs_dir.mkdir()
    for epoch in (5, 2):
        (epochs_dir / f"epoch{epoch}.json").write_text(
            json.dumps(
                {
                    "gain_factors": [0.5, 1.0],
                    "accuracy": [0.1 * epoch, 0.2 * epoch],
                    "auc": [0.3 * epoch, 0.4 * epoch],
                    "categorical_loglikelihood": [-0.1 * epoch, -0.2 * epoch],
                }
            )
        )

    result = analyzer.analyze(
        model=model, test_dataset=dataset, save_path=str(tmp_path)
    )
    assert result["raw_scores"]["accuracy"] == [0.7, 0.7, 0.7]
    assert result["raw_scores"]["auc"] == [0.8, 0.8, 0.8]
    assert result["raw_scores"]["categorical_loglikelihood"] == [-1.2, -1.2, -1.2]
    assert "accuracy" not in result
    assert "auc" not in result
    assert "categorical_loglikelihood" not in result
    assert result["ID"]["accuracy"] == 0.7
    assert result["OOD_lower"]["accuracy"] == pytest.approx(0.7)
    assert result["OOD_upper"]["accuracy"] == pytest.approx(0.7)
    assert "mse" not in result
    assert "cosine_similarity" not in result
    assert result["gain_factors"] == pytest.approx([0.5, 1.0, 2.0])
    assert plotted["final"] == [
        ("accuracy", True, tmp_path.name),
        ("auc", True, tmp_path.name),
        ("categorical_loglikelihood", True, tmp_path.name),
    ]
    assert plotted["training"] == [
        ("accuracy", (2, 5), True),
        ("auc", (2, 5), True),
        ("categorical_loglikelihood", (2, 5), True),
    ]
    assert (tmp_path / "final.json").exists()

    generic_model = torch.nn.Linear(2, 2)
    generic_analyzer = mg.MultiplicativeGainAnalyzer(
        MultiplicativeGainAnalysisParams(
            gain_min=1.0,
            gain_max=3.0,
            n_steps=2,
            logspace=False,
            accuracy=True,
            auc=True,
            categorical_loglikelihood=True,
            mse=True,
            cosine_similarity=True,
        )
    )
    empty_result = generic_analyzer.analyze(
        model=generic_model,
        test_dataset=dataset,
        save_path=None,
    )
    assert empty_result == {"gain_factors": [1.0, 3.0]}


def test_contextual_gain_sets_multiscale_flag_and_training_bounds(monkeypatch):
    dataset = ContextualGainDataset(
        torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1.0,
        torch.arange(3, dtype=torch.long),
        n_regions=2,
        min_gain_factor=0.01,
        max_gain_factor=1.0,
        region_poisson=False,
    )
    assert dataset.multiplicative_gain is True
    assert dataset.multiscale_gain is False
    assert mg.MultiplicativeGainAnalyzer._training_gain_bounds(dataset) == (0.01, 1.0)

    analyzer = mg.MultiplicativeGainAnalyzer(
        MultiplicativeGainAnalysisParams(accuracy=True, n_steps=2)
    )
    model = _make_classifier(input_dim=8)
    monkeypatch.setattr(
        mg, "analysis_device_context", lambda model, device: contextlib.nullcontext("cpu")
    )
    monkeypatch.setattr(
        mg,
        "materialize_dataset",
        lambda *args, **kwargs: (torch.ones(3, 8), torch.zeros(3, dtype=torch.long)),
    )
    monkeypatch.setattr(mg, "evaluation_kwargs_from_runtime", lambda runtime: {})
    monkeypatch.setattr(mg, "evaluate_accuracy", lambda *args, **kwargs: ("ok", 0.5))
    analyzer.analyze(model=model, test_dataset=dataset, save_path=None)
    assert analyzer.multiscale_gain is True


def test_scaled_dataset_uses_labels_from_same_materialize(monkeypatch):
    dataset = PoissonGeneratorDataset(
        torch.ones(2, 2),
        torch.tensor([0, 1]),
        multiplicative_gain=True,
        min_gain_factor=0.1,
        max_gain_factor=1.0,
        poisson_sampling=False,
    )
    model = _make_classifier(input_dim=2)
    analyzer = mg.MultiplicativeGainAnalyzer(
        MultiplicativeGainAnalysisParams(
            gain_min=1.0,
            gain_max=1.0,
            n_steps=1,
            logspace=False,
            accuracy=True,
        )
    )

    call_count = {"n": 0}
    seen_labels = []

    def fake_materialize(*args, **kwargs):
        call_count["n"] += 1
        labels = torch.full((2,), call_count["n"], dtype=torch.long)
        return torch.ones(2, 2) * call_count["n"], labels

    def fake_accuracy(model, test_ds=None, **kwargs):
        seen_labels.append(test_ds.tensors[1].clone())
        return ("ok", 0.5)

    monkeypatch.setattr(
        mg, "analysis_device_context", lambda model, device: contextlib.nullcontext("cpu")
    )
    monkeypatch.setattr(mg, "materialize_dataset", fake_materialize)
    monkeypatch.setattr(mg, "evaluation_kwargs_from_runtime", lambda runtime: {})
    monkeypatch.setattr(mg, "evaluate_accuracy", fake_accuracy)

    analyzer.analyze(model=model, test_dataset=dataset, save_path=None)

    assert seen_labels, "expected evaluate_accuracy to run"
    # First materialize is the unused base snapshot (labels==1). Gain evals must
    # score labels from their own materialize call, not that snapshot.
    for labels in seen_labels:
        assert not torch.equal(labels, torch.ones(2, dtype=torch.long))
        assert int(labels[0]) >= 2


def test_epoch_loading_and_plot_helpers(tmp_path, monkeypatch):
    analyzer = mg.MultiplicativeGainAnalyzer(MultiplicativeGainAnalysisParams())
    assert analyzer.load_epochs_data(str(tmp_path)) == (None, None)

    seed_data = {
        "gain_factors": [1.0, 2.0],
        "metrics": {"accuracy": [0.2, 0.3]},
        "flat_metric": [0.9, 1.1],
    }
    agg = analyzer._initialize_agg_structure(seed_data)
    assert agg == {
        "gain_factors": [1.0, 2.0],
        "metrics": {"accuracy": []},
        "flat_metric": [],
    }

    analyzer._populate_agg_dict(
        agg,
        {
            "gain_factors": [99.0],
            "metrics": {"accuracy": [0.4, 0.5]},
            "flat_metric": [1.2, 1.3],
        },
    )
    assert agg["gain_factors"] == [1.0, 2.0]
    assert agg["metrics"]["accuracy"] == [[0.4, 0.5]]
    assert agg["flat_metric"] == [[1.2, 1.3]]

    epochs_dir = tmp_path / "epochs"
    epochs_dir.mkdir()
    (epochs_dir / "epoch10.json").write_text(
        json.dumps({"gain_factors": [1.0], "accuracy": [0.9]})
    )
    (epochs_dir / "epoch2.json").write_text(
        json.dumps({"gain_factors": [1.0], "accuracy": [0.5]})
    )
    (epochs_dir / "epoch_bad.json").write_text("{}")
    loaded, epoch_numbers = analyzer.load_epochs_data(str(tmp_path))
    assert epoch_numbers == [2, 10]
    assert loaded == {"gain_factors": [1.0], "accuracy": [[0.5], [0.9]]}

    mg.plot_performance_vs_gain_final(
        data_dict={"gain_factors": [1.0, 2.0], "accuracy": [0.4, 0.6]},
        metric="accuracy",
        logspace=True,
        save_path=str(tmp_path),
    )
    assert (tmp_path / "accuracy_vs_gain_final.png").exists()

    class DummyAnimation:
        def __init__(self, fig, animate, frames, interval, repeat, blit):
            animate(0)

        def save(self, path, writer):
            Path(path).touch()

    monkeypatch.setattr(mg.animation, "FuncAnimation", DummyAnimation)
    monkeypatch.setattr(mg.animation.writers, "is_available", lambda writer_name: False)
    monkeypatch.setattr(mg.animation, "PillowWriter", lambda fps: object())

    mg.plot_performance_vs_gain_training(
        epochs_data={
            "gain_factors": [1.0, 2.0],
            "accuracy": [[0.2, 0.4], [0.3, 0.5]],
        },
        epoch_numbers=[1, 2],
        metric="accuracy",
        logspace=False,
        save_path=str(tmp_path),
    )
    assert (tmp_path / "accuracy_vs_gain_training.gif").exists()
