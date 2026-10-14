"""Tests for config-driven trained-checkpoint summaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dendritic_modeling.scripts.analysis.summarize_checkpoint_metrics import (
    main,
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


def test_paired_comparison_inherits_top_level_value_column(tmp_path: Path) -> None:
    rows = []
    for mechanism, offset in (("additive", 0.0), ("shunting", 0.1)):
        for seed in range(3):
            rows.append(
                {
                    "mechanism": mechanism,
                    "seed": seed,
                    "excitatory_contacts": 40,
                    "inhibitory_contacts": 10,
                    "test_accuracy": 0.5 + offset + seed / 100,
                }
            )
    source = tmp_path / "accuracy.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    payload = {
        "schema_version": "checkpoint_metric_summary_v1",
        "input": {"path": str(source)},
        "derived_columns": {
            "total_contacts": {
                "operation": "sum",
                "columns": ["excitatory_contacts", "inhibitory_contacts"],
            },
            "balanced": {
                "operation": "equal",
                "columns": ["excitatory_contacts", "inhibitory_contacts"],
            },
        },
        "filters": {"total_contacts": 50, "balanced": False},
        "group_by": ["mechanism"],
        "unit_columns": ["seed"],
        "value_column": "test_accuracy",
        "expected_units_per_group": 3,
        "bootstrap": {"draws": 200, "seed": 17},
        "paired_comparisons": [
            {
                "name": "shunting_minus_additive",
                "arm_column": "mechanism",
                "reference": "additive",
                "comparison": "shunting",
                "pair_columns": ["seed"],
                "expected_pairs_per_stratum": 3,
            }
        ],
        "output_dir": str(tmp_path / "accuracy_summary"),
    }
    config = tmp_path / "accuracy.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    summarize_checkpoint_metrics(config_path=config)

    paired = pd.read_csv(tmp_path / "accuracy_summary/paired_summary.csv")
    assert paired.loc[0, "metric"] == "test_accuracy"
    assert paired.loc[0, "mean"] == pytest.approx(0.1)


def test_paired_comparison_reduces_declared_arm_sets(tmp_path: Path) -> None:
    rows = []
    for seed in range(3):
        for inhibition, accuracy in ((1, 0.40), (5, 0.44), (45, 0.52), (50, 0.56)):
            rows.append(
                {
                    "seed": seed,
                    "inhibition": inhibition,
                    "test_accuracy": accuracy + seed / 100,
                }
            )
    source = tmp_path / "accuracy.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    payload = {
        "schema_version": "checkpoint_metric_summary_v1",
        "input": {"path": str(source)},
        "group_by": ["inhibition"],
        "unit_columns": ["seed"],
        "value_column": "test_accuracy",
        "expected_units_per_group": 3,
        "bootstrap": {"draws": 200, "seed": 17},
        "paired_comparisons": [
            {
                "name": "upper_minus_lower",
                "arm_column": "inhibition",
                "reference": "lower",
                "comparison": "upper",
                "arm_sets": {"lower": [1, 5], "upper": [45, 50]},
                "arm_set_reduction": "mean",
                "pair_columns": ["seed"],
                "expected_pairs_per_stratum": 3,
            }
        ],
        "output_dir": str(tmp_path / "summary"),
    }
    config = tmp_path / "summary.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    summarize_checkpoint_metrics(config_path=config)

    units = pd.read_csv(tmp_path / "summary/paired_units.csv")
    summary = pd.read_csv(tmp_path / "summary/paired_summary.csv")
    assert units["test_accuracy__lower"].tolist() == pytest.approx([0.42, 0.43, 0.44])
    assert units["test_accuracy__upper"].tolist() == pytest.approx([0.54, 0.55, 0.56])
    assert summary.loc[0, "mean"] == pytest.approx(0.12)


def test_any_of_filters_define_mixed_architecture_reference(tmp_path: Path) -> None:
    rows = []
    for lineage in ("additive", "shunting"):
        for seed in range(3):
            rows.extend(
                [
                    {
                        "lineage": lineage,
                        "architecture": "point",
                        "ee": 3000,
                        "ie": 3000,
                        "seed": seed,
                        "test_accuracy": 0.4,
                    },
                    {
                        "lineage": lineage,
                        "architecture": "tree",
                        "ee": 40,
                        "ie": 10,
                        "seed": seed,
                        "test_accuracy": 0.5,
                    },
                    {
                        "lineage": lineage,
                        "architecture": "tree",
                        "ee": 25,
                        "ie": 25,
                        "seed": seed,
                        "test_accuracy": 0.6,
                    },
                ]
            )
    source = tmp_path / "architectures.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    payload = {
        "schema_version": "checkpoint_metric_summary_v1",
        "input": {"path": str(source)},
        "any_of_filters": [
            {"architecture": "point"},
            {"architecture": "tree", "ee": 40, "ie": 10},
        ],
        "group_by": ["lineage", "architecture"],
        "unit_columns": ["seed"],
        "value_column": "test_accuracy",
        "expected_units_per_group": 3,
        "bootstrap": {"draws": 200, "seed": 17},
        "paired_comparisons": [
            {
                "name": "tree_minus_point",
                "arm_column": "architecture",
                "reference": "point",
                "comparison": "tree",
                "pair_columns": ["seed"],
                "stratum_columns": ["lineage"],
                "expected_pairs_per_stratum": 3,
            }
        ],
        "output_dir": str(tmp_path / "architecture_summary"),
    }
    config = tmp_path / "architecture.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    summarize_checkpoint_metrics(config_path=config)

    units = pd.read_csv(tmp_path / "architecture_summary/checkpoint_units.csv")
    assert len(units) == 12
    paired = pd.read_csv(tmp_path / "architecture_summary/paired_summary.csv")
    assert paired["mean"].tolist() == pytest.approx([0.1, 0.1])


def test_scientific_method_column_survives_bootstrap_metadata(tmp_path: Path) -> None:
    rows = []
    for method in ("fixed_train_mean", "zero"):
        for mechanism, offset in (("additive", 0.0), ("shunting", 0.1)):
            for seed in range(3):
                rows.append(
                    {
                        "method": method,
                        "mechanism": mechanism,
                        "seed": seed,
                        "accuracy_drop": 0.2 + offset + seed / 100,
                    }
                )
    source = tmp_path / "interventions.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    payload = {
        "schema_version": "checkpoint_metric_summary_v1",
        "input": {"path": str(source)},
        "group_by": ["method", "mechanism"],
        "unit_columns": ["seed"],
        "value_column": "accuracy_drop",
        "expected_units_per_group": 3,
        "bootstrap": {"draws": 200, "seed": 17},
        "paired_comparisons": [
            {
                "name": "shunting_minus_additive",
                "arm_column": "mechanism",
                "reference": "additive",
                "comparison": "shunting",
                "pair_columns": ["seed"],
                "stratum_columns": ["method"],
                "expected_pairs_per_stratum": 3,
            }
        ],
        "output_dir": str(tmp_path / "summary"),
    }
    config = tmp_path / "summary.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False))

    summarize_checkpoint_metrics(config_path=config)

    grouped = pd.read_csv(tmp_path / "summary/group_summary.csv")
    paired = pd.read_csv(tmp_path / "summary/paired_summary.csv")
    assert set(grouped["method"]) == {"fixed_train_mean", "zero"}
    assert set(paired["method"]) == {"fixed_train_mean", "zero"}
    assert set(grouped["interval_method"]) == {"experimental_unit_percentile"}
    assert set(paired["interval_method"]) == {"experimental_unit_percentile"}


def test_checkpoint_metric_summary_cli_returns_success(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _write_config(tmp_path, _rows())

    assert main(["--config", str(config)]) == 0
    assert "Summarized 12 rows" in capsys.readouterr().out


@pytest.mark.parametrize("version", ["legacy_difference", "legacy_grouped"])
def test_summary_config_records_explicit_paired_protocol(tmp_path, version):
    import json

    from dendritic_modeling.scripts.sweeps.paired_comparison import (
        paired_arm_comparison,
    )

    config = _write_config(tmp_path, _rows())
    payload = yaml.safe_load(config.read_text())
    payload["paired_comparisons"][0]["protocol_version"] = version
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    release = summarize_checkpoint_metrics(config_path=config)
    actual = pd.read_csv(tmp_path / "summary/paired_summary.csv")
    rows = pd.DataFrame(_rows()).query("analyzer == 'information'")
    _, expected = paired_arm_comparison(
        rows,
        arm_column="mechanism",
        reference="additive",
        comparison="shunting",
        metric="value",
        pair_columns=["seed"],
        stratum_columns=["policy", "metric_path"],
        require_pairs=3,
        draws=200,
        bootstrap_seed=17,
        protocol_version=version,
    )
    assert actual["seed"].tolist() == [row["seed"] for row in expected]
    assert ("contrast_operation" in actual) == (version == "legacy_grouped")
    assert release["config_sha256"] == _sha256(config)
    recorded = json.loads((tmp_path / "summary/summary_provenance.json").read_text())
    assert recorded["config_sha256"] == _sha256(config)


def test_summary_seeds_follow_python_values_of_numpy_group_keys(
    tmp_path: Path,
) -> None:
    from dendritic_modeling.analysis.statistics import stable_seed

    rows = [
        {"mechanism": mechanism, "depth": depth, "seed": seed, "value": seed + depth}
        for mechanism in ("additive", "shunting")
        for depth in (1, 2)
        for seed in range(3)
    ]
    source = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    config = tmp_path / "summary.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "checkpoint_metric_summary_v1",
                "input": {"path": str(source)},
                "group_by": ["mechanism", "depth"],
                "unit_columns": ["seed"],
                "value_column": "value",
                "expected_units_per_group": 3,
                "bootstrap": {"draws": 50, "seed": 17, "confidence": 0.95},
                "output_dir": str(tmp_path / "summary"),
            },
            sort_keys=False,
        )
    )
    summarize_checkpoint_metrics(config_path=config)
    summary = pd.read_csv(tmp_path / "summary/group_summary.csv")
    assert len(summary) == 4
    for row in summary.itertuples():
        identity = (row.mechanism, int(row.depth))
        assert int(row.seed) == stable_seed(17, "checkpoint-metric-summary", *identity)
