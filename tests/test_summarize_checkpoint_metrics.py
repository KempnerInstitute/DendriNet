"""Tests for config-driven trained-checkpoint summaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dendritic_modeling.scripts.analysis.summarize_checkpoint_metrics import (
    summarize_checkpoint_metrics,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(tmp_path: Path, rows: list[dict], *, expected: int = 3) -> Path:
    source = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    config = {
        "schema_version": "checkpoint_metric_summary_v1",
        "input": {"path": str(source), "sha256": _sha256(source)},
        "filters": {"analyzer": "information"},
        "regex_filters": {"metric_path": r"layer\.[01]\.I\(E;C\)"},
        "group_by": ["policy", "mechanism", "metric_path"],
        "unit_columns": ["seed"],
        "value_column": "value",
        "expected_units_per_group": expected,
        "bootstrap": {"draws": 200, "seed": 17, "confidence": 0.95},
        "paired_comparisons": [
            {
                "name": "shunting_minus_additive",
                "arm_column": "mechanism",
                "reference": "additive",
                "comparison": "shunting",
                "pair_columns": ["seed"],
                "stratum_columns": ["policy", "metric_path"],
                "expected_pairs_per_stratum": expected,
            }
        ],
        "output_dir": str(tmp_path / "summary"),
    }
    path = tmp_path / "summary.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _rows() -> list[dict]:
    rows = []
    for metric_index in range(2):
        for mechanism, offset in (("additive", 0.0), ("shunting", 1.0)):
            for seed in range(3):
                rows.append(
                    {
                        "analyzer": "information",
                        "policy": "occupancy",
                        "mechanism": mechanism,
                        "seed": seed,
                        "metric_path": f"layer.{metric_index}.I(E;C)",
                        "value": float(seed + offset + metric_index),
                    }
                )
    rows.append(
        {
            "analyzer": "ablation",
            "policy": "occupancy",
            "mechanism": "additive",
            "seed": 0,
            "metric_path": "ignored",
            "value": 100.0,
        }
    )
    return rows


def test_checkpoint_metric_summary_is_grouped_paired_and_hash_bound(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, _rows())
    release = summarize_checkpoint_metrics(config_path=config)

    assert release["selected_row_count"] == 12
    summary = pd.read_csv(tmp_path / "summary/group_summary.csv")
    assert len(summary) == 4
    assert set(summary["n_units"]) == {3}
    contrasts = pd.read_csv(tmp_path / "summary/paired_summary.csv")
    assert len(contrasts) == 2
    assert contrasts["mean"].tolist() == pytest.approx([1.0, 1.0])
    assert (tmp_path / "summary/summary_provenance.json").is_file()


def test_checkpoint_metric_summary_rejects_duplicate_or_missing_units(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    config = _write_config(tmp_path, rows)
    with pytest.raises(ValueError, match="duplicate experimental units"):
        summarize_checkpoint_metrics(config_path=config)

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    config = _write_config(missing_root, _rows()[:-2])
    with pytest.raises(ValueError, match="Expected 3 experimental units"):
        summarize_checkpoint_metrics(config_path=config)


def test_checkpoint_metric_summary_rejects_modified_input(tmp_path: Path) -> None:
    config = _write_config(tmp_path, _rows())
    with (tmp_path / "metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("modified\n")
    with pytest.raises(ValueError, match="hash differs"):
        summarize_checkpoint_metrics(config_path=config)
