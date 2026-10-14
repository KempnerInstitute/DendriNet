"""Tests for manifest-driven collection of completed sweep metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dendritic_modeling.scripts.sweeps.collect_sweep_metrics import (
    collect_sweep_metrics,
    main,
)


def _write_sweep(root: Path, *, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for config_id in range(2):
        config = root / "configs" / f"unified_config_{config_id}.yaml"
        performance = (
            root / "results" / f"config_{config_id}" / "performance" / "final.json"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        performance.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"seed: {config_id}\n", encoding="utf-8")
        performance.write_text(
            f'{{"test": {{"accuracy": {0.5 + offset + config_id / 100}}}}}\n',
            encoding="utf-8",
        )
        (performance.parents[1] / "training_summary.json").write_text(
            json.dumps(
                {
                    "train_losses": [1.0 + offset, 0.5 + offset],
                    "valid_losses": [1.1 + offset, 0.6 + offset],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        epoch_dir = performance.parent / "epochs"
        epoch_dir.mkdir()
        for epoch, accuracy in ((1, 0.4), (2, 0.6)):
            (epoch_dir / f"epoch{epoch}.json").write_text(
                json.dumps({"accuracy": {"valid": accuracy + offset + config_id / 100}})
                + "\n",
                encoding="utf-8",
            )
        rows.append(
            {
                "config_id": config_id,
                "seed": config_id,
                "test_accuracy": 0.5 + offset + config_id / 100,
            }
        )
    return pd.DataFrame(rows)


def _manifest(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "sweep_metric_collection_v1",
        "required_columns": ["config_id", "seed", "test_accuracy"],
        "training_history": {
            "fields": {"train_losses": "train_loss", "valid_losses": "valid_loss"},
            "require_equal_length_within_checkpoint": True,
        },
        "sweeps": [
            {
                "name": "additive",
                "root": str(tmp_path / "additive"),
                "expected_config_count": 2,
                "labels": {"mechanism": "additive"},
            },
            {
                "name": "shunting",
                "root": str(tmp_path / "shunting"),
                "expected_config_count": 2,
                "labels": {"mechanism": "shunting"},
            },
        ],
        "output_dir": str(tmp_path / "collected"),
    }
    path = tmp_path / "collection.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_collect_sweep_metrics_is_complete_labeled_and_hash_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = {
        str((tmp_path / "additive").resolve()): _write_sweep(tmp_path / "additive"),
        str((tmp_path / "shunting").resolve()): _write_sweep(
            tmp_path / "shunting", offset=0.1
        ),
    }
    monkeypatch.setattr(
        "dendritic_modeling.scripts.sweeps.collect_sweep_metrics.DataCollector.collect_all",
        lambda collector: frames[str(Path(collector.results_dir).resolve())].copy(),
    )
    manifest = _manifest(tmp_path)

    release = collect_sweep_metrics(manifest_path=manifest)

    assert release["collected_checkpoint_count"] == 4
    units = pd.read_csv(tmp_path / "collected/checkpoint_units.csv")
    assert set(units["mechanism"]) == {"additive", "shunting"}
    assert not units.duplicated(["sweep_name", "config_id"]).any()
    assert all(len(record["artifacts"]) == 6 for record in release["sweeps"])
    history = pd.read_csv(tmp_path / "collected/training_history.csv")
    assert len(history) == 16
    assert set(history["metric"]) == {"train_loss", "valid_loss"}
    assert (tmp_path / "collected/collection_provenance.json").is_file()


def test_collect_sweep_metrics_rejects_incomplete_or_changed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _write_sweep(tmp_path / "additive")
    _write_sweep(tmp_path / "shunting")
    monkeypatch.setattr(
        "dendritic_modeling.scripts.sweeps.collect_sweep_metrics.DataCollector.collect_all",
        lambda collector: frame.iloc[:1].copy(),
    )
    manifest = _manifest(tmp_path)
    with pytest.raises(RuntimeError, match="incomplete sweep"):
        collect_sweep_metrics(manifest_path=manifest)

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["sweeps"] = [payload["sweeps"][0]]
    payload["sweeps"][0]["artifact_set_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "dendritic_modeling.scripts.sweeps.collect_sweep_metrics.DataCollector.collect_all",
        lambda collector: frame.copy(),
    )
    with pytest.raises(ValueError, match="artifact set hash differs"):
        collect_sweep_metrics(manifest_path=manifest)


def test_collect_sweep_metrics_reads_declared_performance_epoch_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = {
        str((tmp_path / "additive").resolve()): _write_sweep(tmp_path / "additive"),
        str((tmp_path / "shunting").resolve()): _write_sweep(
            tmp_path / "shunting", offset=0.1
        ),
    }
    monkeypatch.setattr(
        "dendritic_modeling.scripts.sweeps.collect_sweep_metrics.DataCollector.collect_all",
        lambda collector: frames[str(Path(collector.results_dir).resolve())].copy(),
    )
    manifest = _manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["training_history"]["performance_epoch_fields"] = {
        "accuracy.valid": "valid_accuracy"
    }
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    release = collect_sweep_metrics(manifest_path=manifest)

    history = pd.read_csv(tmp_path / "collected/training_history.csv")
    accuracy = history.loc[history["metric"] == "valid_accuracy"]
    assert len(accuracy) == 8
    assert set(accuracy["epoch_number"]) == {1, 2}
    assert set(accuracy["value"].round(2)) == {
        0.4,
        0.41,
        0.5,
        0.51,
        0.6,
        0.61,
        0.7,
        0.71,
    }
    assert all(len(record["artifacts"]) == 10 for record in release["sweeps"])


def test_collect_sweep_metrics_cli_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    frames = {
        str((tmp_path / "additive").resolve()): _write_sweep(tmp_path / "additive"),
        str((tmp_path / "shunting").resolve()): _write_sweep(
            tmp_path / "shunting", offset=0.1
        ),
    }
    monkeypatch.setattr(
        "dendritic_modeling.scripts.sweeps.collect_sweep_metrics.DataCollector.collect_all",
        lambda collector: frames[str(Path(collector.results_dir).resolve())].copy(),
    )
    manifest = _manifest(tmp_path)

    assert main(["--manifest", str(manifest)]) == 0
    assert "Collected 4 checkpoints" in capsys.readouterr().out


def test_collect_sweep_metrics_rejects_runs_recorded_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _write_sweep(tmp_path / "additive")
    _write_sweep(tmp_path / "shunting")
    monkeypatch.setattr(
        "dendritic_modeling.scripts.sweeps.collect_sweep_metrics.DataCollector.collect_all",
        lambda collector: frame.copy(),
    )
    manifest = _manifest(tmp_path)
    run_dir = tmp_path / "additive" / "results" / "config_1"

    def write_status(passes):
        (run_dir / "analysis_status.json").write_text(
            json.dumps({"schema_version": "analysis_status_v1", "passes": passes})
        )

    write_status(
        {
            "final": {
                "finished": True,
                "analyses": {"performance": {"status": "empty"}},
            },
        }
    )
    with pytest.raises(ValueError, match=r"config_id=1 .*performance: empty"):
        collect_sweep_metrics(manifest_path=manifest)

    write_status(
        {
            "final": {
                "finished": True,
                "analyses": {"performance": {"status": "completed"}},
            },
            "epoch2": {
                "training": True,
                "finished": True,
                "analyses": {"performance": {"status": "skipped", "reason": "x"}},
            },
        }
    )
    with pytest.raises(ValueError, match=r"performance \(epoch2\): skipped"):
        collect_sweep_metrics(manifest_path=manifest)

    (run_dir / "analysis_status.json").write_text("{not json")
    with pytest.raises(ValueError, match="is not valid JSON"):
        collect_sweep_metrics(manifest_path=manifest)

    (run_dir / "analysis_status.json").unlink()  # runs without records pass
    assert collect_sweep_metrics(manifest_path=manifest)["collected_checkpoint_count"]
