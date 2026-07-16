"""Tests for fail-closed paired sweep comparisons."""

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
