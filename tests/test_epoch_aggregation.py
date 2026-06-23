"""Tests for shared per-epoch analysis aggregation helpers."""

import json

from dendritic_modeling.analysis.tools.compartment_snr import CompartmentSNRAnalyzer
from dendritic_modeling.analysis.tools.epoch_aggregation import (
    initialize_epoch_aggregation_structure,
    load_epoch_aggregation,
    populate_epoch_aggregation,
)
from dendritic_modeling.config.analysis import CompartmentSNRAnalysisParams


def test_epoch_aggregation_preserves_metadata_leaves_and_appends_metrics():
    seed_data = {
        "depth": 2,
        "gain_factors": [1.0, 2.0],
        "layer": {"metric": 0.5, "depth": 3},
    }

    agg = initialize_epoch_aggregation_structure(
        seed_data,
        preserved_leaf_keys={"depth", "gain_factors"},
    )
    populate_epoch_aggregation(
        agg,
        {
            "depth": 99,
            "gain_factors": [9.0],
            "layer": {"metric": 0.7, "depth": 8},
        },
        preserved_leaf_keys={"depth", "gain_factors"},
    )

    assert agg == {
        "depth": 2,
        "gain_factors": [1.0, 2.0],
        "layer": {"metric": [0.7], "depth": 3},
    }


def test_load_epoch_aggregation_sorts_epoch_json_and_skips_bad_names(tmp_path):
    epochs_dir = tmp_path / "epochs"
    epochs_dir.mkdir()
    (epochs_dir / "epoch10.json").write_text(json.dumps({"depth": 1, "metric": 10.0}))
    (epochs_dir / "epoch2.json").write_text(json.dumps({"depth": 1, "metric": 2.0}))
    (epochs_dir / "epoch_bad.json").write_text(json.dumps({"metric": 999.0}))
    (epochs_dir / "notes.json").write_text(json.dumps({"metric": 999.0}))

    loaded, epoch_numbers = load_epoch_aggregation(
        str(tmp_path),
        preserved_leaf_keys={"depth"},
    )

    assert epoch_numbers == [2, 10]
    assert loaded == {"depth": 1, "metric": [2.0, 10.0]}


def test_compartment_snr_epoch_wrappers_preserve_depth_metadata(tmp_path):
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())
    seed_data = {"branch": {"depth": 2, "vout_d'^2": 0.1}}

    agg = analyzer._initialize_agg_structure(seed_data)
    analyzer._populate_agg_dict(agg, {"branch": {"depth": 99, "vout_d'^2": 0.3}})

    assert agg == {"branch": {"depth": 2, "vout_d'^2": [0.3]}}

    epochs_dir = tmp_path / "epochs"
    epochs_dir.mkdir()
    (epochs_dir / "epoch3.json").write_text(
        json.dumps({"branch": {"depth": 2, "vout_d'^2": 0.4}})
    )
    (epochs_dir / "epoch1.json").write_text(
        json.dumps({"branch": {"depth": 2, "vout_d'^2": 0.2}})
    )

    loaded, epoch_numbers = analyzer.load_epochs_data(str(tmp_path))

    assert epoch_numbers == [1, 3]
    assert loaded == {"branch": {"depth": 2, "vout_d'^2": [0.2, 0.4]}}
