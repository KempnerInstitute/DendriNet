from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dendritic_modeling.scripts.analysis.select_validation_epochs import (
    select_validation_epochs,
)


def _write_inputs(tmp_path: Path) -> Path:
    units = pd.DataFrame(
        [
            {
                "sweep_name": "audit",
                "config_id": 0,
                "seed": 10,
                "mechanism": "shunting",
            },
            {
                "sweep_name": "audit",
                "config_id": 1,
                "seed": 11,
                "mechanism": "additive",
            },
        ]
    )
    history_rows = []
    values = {
        0: {
            "valid_accuracy": [0.4, 0.6, 0.6],
            "test_accuracy": [0.3, 0.5, 0.7],
            "valid_categorical_loglikelihood": [-2.0, -1.0, -1.5],
        },
        1: {
            "valid_accuracy": [0.2, 0.5, 0.7],
            "test_accuracy": [0.1, 0.4, 0.6],
            "valid_categorical_loglikelihood": [-2.0, -1.2, -1.3],
        },
    }
    for config_id, metrics in values.items():
        for metric, trajectory in metrics.items():
            for epoch, value in enumerate(trajectory, start=1):
                history_rows.append(
                    {
                        "sweep_name": "audit",
                        "config_id": config_id,
                        "seed": 10 + config_id,
                        "metric": metric,
                        "epoch_number": epoch,
                        "value": value,
                    }
                )
    units.to_csv(tmp_path / "units.csv", index=False)
    pd.DataFrame(history_rows).to_csv(tmp_path / "history.csv", index=False)
    config = {
        "schema_version": "validation_epoch_selection_v1",
        "history": {"path": "history.csv"},
        "units": {"path": "units.csv"},
        "join_columns": ["sweep_name", "config_id", "seed"],
        "metadata_columns": ["mechanism"],
        "expected_unit_count": 2,
        "expected_epochs_per_unit": 3,
        "selectors": [
            {
                "name": "validation_nll",
                "metric": "valid_categorical_loglikelihood",
                "objective": "max",
                "tie_break": "earliest",
            },
            {
                "name": "validation_accuracy",
                "metric": "valid_accuracy",
                "objective": "max",
                "tie_break": "earliest",
            },
        ],
        "report_metrics": ["valid_accuracy", "test_accuracy"],
        "output_dir": "selected",
    }
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_select_validation_epochs_uses_validation_and_reports_test(
    tmp_path: Path,
) -> None:
    config_path = _write_inputs(tmp_path)

    release = select_validation_epochs(config_path=config_path)

    selected = pd.read_csv(tmp_path / "selected/selected_epochs.csv")
    first = selected.loc[
        (selected["config_id"] == 0) & (selected["selection"] == "validation_accuracy")
    ].iloc[0]
    assert first["selected_epoch_number"] == 2
    assert first["test_accuracy"] == pytest.approx(0.5)
    nll = selected.loc[
        (selected["config_id"] == 0) & (selected["selection"] == "validation_nll")
    ].iloc[0]
    assert nll["selected_epoch_number"] == 2
    second_accuracy = selected.loc[
        (selected["config_id"] == 1) & (selected["selection"] == "validation_accuracy")
    ].iloc[0]
    assert second_accuracy["selected_epoch_number"] == 3
    assert second_accuracy["test_accuracy"] == pytest.approx(0.6)
    assert release["selection_data_role"] == "validation_only"
    provenance = json.loads(
        (tmp_path / "selected/selection_provenance.json").read_text()
    )
    assert provenance["trained_checkpoint_count"] == 2
    assert provenance["selected_epoch_count"] == 4
    assert provenance["selected_metric_row_count"] == 8


def test_select_validation_epochs_rejects_test_selection(tmp_path: Path) -> None:
    config_path = _write_inputs(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["selectors"][0]["metric"] = "test_accuracy"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="validation metric named valid_\\*"):
        select_validation_epochs(config_path=config_path)


def test_select_validation_epochs_rejects_incomplete_epoch_grid(
    tmp_path: Path,
) -> None:
    config_path = _write_inputs(tmp_path)
    history = pd.read_csv(tmp_path / "history.csv")
    history = history.loc[
        ~(
            (history["config_id"] == 1)
            & (history["metric"] == "test_accuracy")
            & (history["epoch_number"] == 3)
        )
    ]
    history.to_csv(tmp_path / "history.csv", index=False)

    with pytest.raises(ValueError, match="incomplete metric-by-epoch grid"):
        select_validation_epochs(config_path=config_path)
