"""Tests for fail-closed paired sweep comparisons."""

from pathlib import Path

import pandas as pd
import pytest

import dendritic_modeling.scripts.sweeps.paired_comparison as paired_module
from dendritic_modeling.scripts.sweeps.paired_comparison import (
    collect_labeled_sweeps,
    paired_arm_comparison,
)


def _paired_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"seed": 1, "mechanism": "add", "arm": "none", "accuracy": 0.2},
            {"seed": 1, "mechanism": "add", "arm": "scaled", "accuracy": 0.3},
            {"seed": 2, "mechanism": "add", "arm": "none", "accuracy": 0.4},
            {"seed": 2, "mechanism": "add", "arm": "scaled", "accuracy": 0.6},
        ]
    )


def test_paired_arm_comparison_uses_seed_differences():
    paired, summaries = paired_arm_comparison(
        _paired_data(),
        arm_column="arm",
        reference="none",
        comparison="scaled",
        metric="accuracy",
        pair_columns=["seed"],
        stratum_columns=["mechanism"],
        require_pairs=2,
        draws=500,
        bootstrap_seed=7,
    )

    assert paired["delta__scaled_minus_none"].tolist() == pytest.approx([0.1, 0.2])
    assert summaries[0]["mean"] == pytest.approx(0.15)
    assert summaries[0]["n_units"] == 2
    assert summaries[0]["mechanism"] == "add"


def test_paired_arm_comparison_supports_symmetric_difference():
    paired, summaries = paired_arm_comparison(
        _paired_data(),
        arm_column="arm",
        reference="none",
        comparison="scaled",
        metric="accuracy",
        pair_columns=["seed"],
        stratum_columns=["mechanism"],
        require_pairs=2,
        draws=100,
        bootstrap_seed=7,
        contrast_operation="symmetric_difference",
    )

    assert paired["delta__scaled_minus_none"].tolist() == pytest.approx(
        [0.1 / 0.5, 0.2 / 1.0]
    )
    assert summaries[0]["mean"] == pytest.approx(0.2)
    assert summaries[0]["contrast_operation"] == "symmetric_difference"


def test_paired_arm_comparison_fails_on_unpaired_seed():
    data = _paired_data().iloc[:-1]
    with pytest.raises(ValueError, match="Unpaired experimental units"):
        paired_arm_comparison(
            data,
            arm_column="arm",
            reference="none",
            comparison="scaled",
            metric="accuracy",
            pair_columns=["seed"],
        )


def test_paired_arm_comparison_fails_on_duplicate_arm():
    data = pd.concat([_paired_data(), _paired_data().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate paired-arm observations"):
        paired_arm_comparison(
            data,
            arm_column="arm",
            reference="none",
            comparison="scaled",
            metric="accuracy",
            pair_columns=["seed"],
        )


def test_paired_arm_comparison_enforces_pair_count():
    with pytest.raises(ValueError, match="Expected 3 pairs"):
        paired_arm_comparison(
            _paired_data(),
            arm_column="arm",
            reference="none",
            comparison="scaled",
            metric="accuracy",
            pair_columns=["seed"],
            require_pairs=3,
        )


def test_collect_labeled_sweeps_attaches_declared_arm(monkeypatch, tmp_path):
    roots = {"direct": tmp_path / "direct", "population": tmp_path / "population"}

    class Collector:
        def __init__(self, root):
            self.root = root

        def collect_all(self):
            return pd.DataFrame([{"seed": 1, "accuracy_test": 0.5}])

    monkeypatch.setattr(paired_module, "DataCollector", Collector)
    data = collect_labeled_sweeps(roots, label_column="routing")

    assert data["routing"].tolist() == ["direct", "population"]
    assert set(data["sweep_root"]) == {str(path.resolve()) for path in roots.values()}


def test_collect_labeled_sweeps_rejects_column_collision(monkeypatch, tmp_path):
    class Collector:
        def __init__(self, root):
            pass

        def collect_all(self):
            return pd.DataFrame([{"routing": "from_config"}])

    monkeypatch.setattr(paired_module, "DataCollector", Collector)
    with pytest.raises(ValueError, match="collides"):
        collect_labeled_sweeps({"direct": tmp_path / "direct"}, label_column="routing")


@pytest.mark.parametrize(
    "version,reference", [("current", "current"), ("legacy_difference", "legacy")]
)
def test_paired_protocol_matches_executed_historical_reference(version, reference):
    import json
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/paired_protocol_reference.json").read_text()
    )[reference]
    units, summaries = paired_arm_comparison(
        pd.DataFrame(fixture["rows"]),
        arm_column="arm",
        reference="reference",
        comparison="comparison",
        metric="metric",
        pair_columns=["seed"],
        stratum_columns=["group"],
        draws=200,
        bootstrap_seed=17,
        protocol_version=version,
    )
    assert units.to_csv(index=False) == fixture["paired_csv"]
    assert summaries == fixture["summaries"]


def test_legacy_grouped_keeps_distinct_contrasts_and_default_unchanged():
    args = {
        "arm_column": "arm",
        "reference": "none",
        "comparison": "scaled",
        "metric": "accuracy",
        "pair_columns": ["seed"],
        "draws": 200,
        "bootstrap_seed": 17,
    }
    default_units, default = paired_arm_comparison(_paired_data(), **args)
    explicit_units, explicit = paired_arm_comparison(
        _paired_data(), **args, protocol_version="current"
    )
    pd.testing.assert_frame_equal(default_units, explicit_units)
    assert default == explicit
    _, old = paired_arm_comparison(
        _paired_data(), **args, protocol_version="legacy_difference"
    )
    _, grouped = paired_arm_comparison(
        _paired_data(), **args, protocol_version="legacy_grouped"
    )
    assert grouped[0].pop("contrast_operation") == "difference"
    assert grouped == old
    current_units, current = paired_arm_comparison(
        _paired_data(), **args, contrast_operation="symmetric_difference"
    )
    legacy_units, legacy = paired_arm_comparison(
        _paired_data(),
        **args,
        contrast_operation="symmetric_difference",
        protocol_version="legacy_grouped",
    )
    pd.testing.assert_frame_equal(
        legacy_units.rename(
            columns={"symmetric_delta__scaled_minus_none": "delta__scaled_minus_none"}
        ),
        current_units,
    )
    assert legacy == current


@pytest.mark.parametrize(
    "version,operation",
    [("typo", "difference"), ("legacy_difference", "symmetric_difference")],
)
def test_paired_protocol_rejects_unsupported_combinations(version, operation):
    with pytest.raises(ValueError, match=r"protocol version|supports only"):
        paired_arm_comparison(
            _paired_data(),
            arm_column="arm",
            reference="none",
            comparison="scaled",
            metric="accuracy",
            pair_columns=["seed"],
            protocol_version=version,
            contrast_operation=operation,
        )


def test_analysis_identity_hashes_the_imported_modules():
    records = paired_module._analysis_identity()["source_files"]
    assert [record["path"] for record in records] == [
        "src/dendritic_modeling/scripts/sweeps/paired_comparison.py",
        "src/dendritic_modeling/scripts/sweeps/collectors/data_collector.py",
        "src/dendritic_modeling/scripts/sweeps/collectors/config_extractor.py",
        "src/dendritic_modeling/analysis/statistics.py",
    ]
    assert records[0]["sha256"] == paired_module._sha256(Path(paired_module.__file__))


def test_the_command_line_selects_the_contrast_operation(monkeypatch, tmp_path):
    captured = {}

    def fake_analyze(sweep_root, output_dir, **kwargs):
        captured.update(kwargs)
        return tmp_path / "units.csv", tmp_path / "summary.json"

    monkeypatch.setattr(paired_module, "analyze_paired_sweep", fake_analyze)
    arguments = [str(tmp_path), "--output", str(tmp_path / "out")]
    arguments += ["--arm-column", "arm", "--reference", "a", "--comparison", "b"]
    arguments += ["--metric", "test_accuracy"]
    assert (
        paired_module.main([*arguments, "--contrast-operation", "symmetric_difference"])
        == 0
    )
    assert captured["contrast_operation"] == "symmetric_difference"
