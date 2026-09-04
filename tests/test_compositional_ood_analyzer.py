import json

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools import compositional_ood as cood
from dendritic_modeling.config import CompositionalOODAnalysisParams
from dendritic_modeling.datasets.contextual_gain import ContextualGainDataset
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity


def _make_classifier(input_dim: int = 2) -> Classifier:
    return Classifier(
        encoder_network=Identity(input_dim=input_dim),
        core_network=Identity(input_dim=input_dim),
        decoder_network=Identity(input_dim=input_dim),
    )


def _make_dataset(
    *,
    n_samples: int = 8,
    n_features: int = 4,
    n_regions: int = 2,
    n_context_latents: int = 2,
    context_dim: int = 4,
    compositional_ood: bool = False,
    region_poisson: bool = False,
) -> ContextualGainDataset:
    base_inputs = (
        torch.arange(n_samples * n_features, dtype=torch.float32).reshape(
            n_samples, n_features
        )
        + 1.0
    )
    labels = torch.arange(n_samples, dtype=torch.long)
    return ContextualGainDataset(
        base_inputs,
        labels,
        n_regions=n_regions,
        min_gain_factor=0.01,
        max_gain_factor=1.0,
        region_poisson=region_poisson,
        n_context_latents=n_context_latents,
        context_dim=context_dim,
        compositional_ood=compositional_ood,
    )


def _make_ood_dataset() -> ContextualGainDataset:
    return _make_dataset(
        n_regions=2,
        n_context_latents=2,
        compositional_ood=True,
    )


def test_noop_on_tensor_dataset(tmp_path):
    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy"])
    )
    dataset = TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
    result = analyzer.analyze(
        model=_make_classifier(),
        test_dataset=dataset,
        save_path=str(tmp_path),
    )
    assert result is None
    assert not (tmp_path / "final.json").exists()


def test_noop_when_ood_disabled(tmp_path):
    dataset = _make_dataset(
        n_regions=2,
        n_context_latents=2,
        compositional_ood=False,
    )
    assert dataset._ood_enabled is False
    dataset._ood_sampling = False
    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy"])
    )
    result = analyzer.analyze(
        model=_make_classifier(),
        test_dataset=dataset,
        save_path=str(tmp_path),
    )
    assert result is None
    assert dataset._ood_sampling is False
    assert not (tmp_path / "final.json").exists()


def test_happy_path_saves_json_and_restores_ood_flag(tmp_path, monkeypatch):
    dataset = _make_ood_dataset()
    assert dataset._ood_enabled is True
    dataset._set_OOD_sampling(False)

    seen = {"ood_during_materialize": None}
    original_materialize = cood.materialize_dataset

    def tracking_materialize(ds, runtime, **kwargs):
        seen["ood_during_materialize"] = bool(ds._ood_sampling)
        return original_materialize(ds, runtime, **kwargs)

    monkeypatch.setattr(cood, "materialize_dataset", tracking_materialize)
    monkeypatch.setattr(
        cood,
        "evaluate_accuracy",
        lambda *args, **kwargs: (None, None, 0.81),
    )

    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy"])
    )
    result = analyzer.analyze(
        model=_make_classifier(),
        test_dataset=dataset,
        save_path=str(tmp_path),
    )
    assert result is None
    assert seen["ood_during_materialize"] is True
    assert dataset._ood_sampling is False
    payload = json.loads((tmp_path / "final.json").read_text())
    assert payload == {"accuracy": 0.81}


def test_metrics_share_same_tensor_dataset(monkeypatch):
    dataset = _make_ood_dataset()
    frozen_inputs = torch.ones(4, 8)
    frozen_labels = torch.zeros(4, dtype=torch.long)
    seen_datasets: list[object] = []

    monkeypatch.setattr(
        cood,
        "materialize_dataset",
        lambda *args, **kwargs: (frozen_inputs, frozen_labels),
    )

    def _record(score):
        def _eval(model, test_ds=None, **kwargs):
            del model, kwargs
            seen_datasets.append(test_ds)
            return (None, None, score)

        return _eval

    monkeypatch.setattr(cood, "evaluate_accuracy", _record(0.5))
    monkeypatch.setattr(cood, "evaluate_auc", _record(0.6))

    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy", "auc"])
    )
    result = analyzer.analyze(
        model=_make_classifier(),
        test_dataset=dataset,
        save_path=None,
    )
    assert result == {"accuracy": 0.5, "auc": 0.6}
    assert len(seen_datasets) == 2
    assert seen_datasets[0] is seen_datasets[1]
    assert isinstance(seen_datasets[0], TensorDataset)


def test_incompatible_metric_omitted_from_payload(monkeypatch):
    dataset = _make_ood_dataset()
    monkeypatch.setattr(
        cood,
        "materialize_dataset",
        lambda *args, **kwargs: (torch.ones(4, 8), torch.zeros(4, dtype=torch.long)),
    )
    monkeypatch.setattr(
        cood,
        "evaluate_accuracy",
        lambda *args, **kwargs: (None, None, 0.9),
    )
    called = {"mse": False}

    def _mse(*args, **kwargs):
        called["mse"] = True
        return (None, None, 0.1)

    monkeypatch.setattr(cood, "evaluate_mse", _mse)

    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy", "mse"])
    )
    result = analyzer.analyze(
        model=_make_classifier(),
        test_dataset=dataset,
        save_path=None,
    )
    assert result == {"accuracy": 0.9}
    assert called["mse"] is False


def _write_epoch_json(epochs_dir, epoch: int, payload: dict) -> None:
    epochs_dir.mkdir(parents=True, exist_ok=True)
    (epochs_dir / f"epoch{epoch}.json").write_text(json.dumps(payload))


def test_load_epochs_data_sorts_and_aggregates(tmp_path):
    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy"])
    )
    epochs_dir = tmp_path / "epochs"
    _write_epoch_json(epochs_dir, 3, {"accuracy": 0.7, "auc": 0.6})
    _write_epoch_json(epochs_dir, 1, {"accuracy": 0.4, "auc": 0.5})

    epochs_data, epoch_numbers = analyzer.load_epochs_data(str(tmp_path))
    assert epoch_numbers == [1, 3]
    assert epochs_data["accuracy"] == [0.4, 0.7]
    assert epochs_data["auc"] == [0.5, 0.6]


def test_load_epochs_data_missing_dir_returns_none(tmp_path):
    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy"])
    )
    epochs_data, epoch_numbers = analyzer.load_epochs_data(str(tmp_path))
    assert epochs_data is None
    assert epoch_numbers is None


def test_plot_performance_vs_epoch_writes_png(tmp_path):
    cood.plot_performance_vs_epoch(
        {"accuracy": [0.2, 0.5, 0.8]},
        [1, 2, 3],
        "accuracy",
        str(tmp_path),
    )
    assert (tmp_path / "accuracy_vs_epoch.png").is_file()


def test_plot_performance_vs_epoch_skips_mismatch(tmp_path):
    cood.plot_performance_vs_epoch(
        {"accuracy": [0.2, 0.5]},
        [1, 2, 3],
        "accuracy",
        str(tmp_path),
    )
    assert not (tmp_path / "accuracy_vs_epoch.png").exists()
    cood.plot_performance_vs_epoch(
        {"auc": [0.2, 0.5]},
        [1, 2],
        "accuracy",
        str(tmp_path),
    )
    assert not (tmp_path / "accuracy_vs_epoch.png").exists()


def test_analyze_final_writes_epoch_plots(tmp_path, monkeypatch):
    dataset = _make_ood_dataset()
    _write_epoch_json(tmp_path / "epochs", 1, {"accuracy": 0.4, "auc": 0.5})
    _write_epoch_json(tmp_path / "epochs", 2, {"accuracy": 0.6, "auc": 0.7})
    monkeypatch.setattr(
        cood,
        "materialize_dataset",
        lambda *args, **kwargs: (torch.ones(4, 8), torch.zeros(4, dtype=torch.long)),
    )
    monkeypatch.setattr(
        cood,
        "evaluate_accuracy",
        lambda *args, **kwargs: (None, None, 0.8),
    )
    monkeypatch.setattr(
        cood,
        "evaluate_auc",
        lambda *args, **kwargs: (None, None, 0.9),
    )

    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy", "auc"])
    )
    analyzer.analyze(
        model=_make_classifier(),
        test_dataset=dataset,
        save_path=str(tmp_path),
        training=False,
    )
    assert (tmp_path / "final.json").is_file()
    assert (tmp_path / "accuracy_vs_epoch.png").is_file()
    assert (tmp_path / "auc_vs_epoch.png").is_file()


def test_analyze_training_does_not_write_plots(tmp_path, monkeypatch):
    dataset = _make_ood_dataset()
    monkeypatch.setattr(
        cood,
        "materialize_dataset",
        lambda *args, **kwargs: (torch.ones(4, 8), torch.zeros(4, dtype=torch.long)),
    )
    monkeypatch.setattr(
        cood,
        "evaluate_accuracy",
        lambda *args, **kwargs: (None, None, 0.5),
    )

    analyzer = cood.CompositionalOODAnalyzer(
        CompositionalOODAnalysisParams(metrics=["accuracy"])
    )
    analyzer.analyze(
        model=_make_classifier(),
        test_dataset=dataset,
        save_path=str(tmp_path),
        filename="epoch1",
        training=True,
    )
    assert (tmp_path / "epochs" / "epoch1.json").is_file()
    assert not (tmp_path / "accuracy_vs_epoch.png").exists()
    payload = json.loads((tmp_path / "epochs" / "epoch1.json").read_text())
    assert payload == {"accuracy": 0.5}
