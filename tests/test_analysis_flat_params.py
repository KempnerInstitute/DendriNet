"""Deprecated flat analysis parameters reach the grouped fields the analyzers read."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest
import yaml

from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.analysis.tools.correlation_analysis import CorrelationAnalyzer
from dendritic_modeling.config import load_config
from dendritic_modeling.config.analysis import (
    AblationAnalysisParams,
    AblationSelectionParams,
    CorrelationAnalysisParams,
    InformationAnalysisParams,
    InformationEstimatorParams,
)
from dendritic_modeling.config.keys import field_default
from dendritic_modeling.config.legacy import (
    CORRELATION_FLAT_PARAMS,
    INFORMATION_FLAT_PARAMS,
)

BASE = {
    "data": {"dataset_name": "mnist"},
    "model": {"encoder": {"type": "identity", "params": {"input_dim": 4}}},
}


def _load(tmp_path, analysis):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**BASE, "analysis": analysis}))
    return load_config(str(path))


def _grouped(params, path):
    for key in path:
        params = getattr(params, key)
    return params


@pytest.mark.parametrize(
    "params_cls, table",
    [
        (InformationAnalysisParams, INFORMATION_FLAT_PARAMS),
        (CorrelationAnalysisParams, CORRELATION_FLAT_PARAMS),
    ],
)
def test_flat_and_grouped_defaults_agree(params_cls, table):
    params = params_cls()
    for name, path in table.items():
        assert field_default(params_cls, name) == _grouped(params, path), name


@pytest.mark.parametrize(
    "flat, value, path, attribute",
    [
        ("method", "decoder", ("estimator", "method"), "method"),
        ("n_neighbors", 3, ("estimator", "kraskov", "n_neighbors"), "n_neighbors"),
        ("n_bins", 7, ("estimator", "binned", "n_bins"), "n_bins"),
        ("computation_level", "layer_branch", None, "computation_level"),
        ("include_vinf", False, None, "include_vinf"),
        ("mi_null_shuffles", 3, None, "mi_null_shuffles"),
        ("compute_lda_weights", True, None, "compute_lda_weights"),
        ("per_layer_analysis", True, None, "per_layer_analysis"),
        ("pid_binarize_method", "mean", None, "pid_binarize_method"),
    ],
)
def test_flat_information_key_reaches_the_analyzer(
    tmp_path, caplog, flat, value, path, attribute
):
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, {"information": {"params": {flat: value}}})
    params = config.analysis.information.params
    grouped_path = path or INFORMATION_FLAT_PARAMS[flat]
    assert _grouped(params, grouped_path) == value
    assert getattr(InformationAnalyzer(params), attribute) == value
    assert f"{flat} -> {'.'.join(grouped_path)}" in caplog.text


def test_flat_selection_lists_reach_the_information_analyzer(tmp_path):
    config = _load(
        tmp_path,
        {"information": {"params": {"components": ["basic"], "scopes": ["layer"]}}},
    )
    analyzer = InformationAnalyzer(config.analysis.information.params)
    assert analyzer.compute_basic_mi and not analyzer.compute_pairwise_mi
    assert analyzer.per_layer_analysis and not analyzer.per_neuron_analysis


def test_flat_lda_shuffles_reach_random_weight_baseline(tmp_path):
    config = _load(
        tmp_path,
        {"information": {"params": {"lda_n_shuffles": 2, "compute_lda_weights": True}}},
    )
    params = config.analysis.information.params
    assert params.baselines.random_weights.n_shuffles == 2
    assert InformationAnalyzer(params).lda_n_shuffles == 2


def test_contradicting_flat_and_grouped_keys_raise(tmp_path):
    analysis = {
        "information": {
            "params": {"method": "decoder", "estimator": {"method": "binned"}}
        }
    }
    with pytest.raises(ValueError, match="conflicts with"):
        _load(tmp_path, analysis)


@pytest.mark.parametrize("flat_value", ["binned", "kraskov"])
def test_agreeing_or_default_flat_key_defers_to_grouped(tmp_path, flat_value):
    analysis = {
        "information": {
            "params": {"method": flat_value, "estimator": {"method": "binned"}}
        }
    }
    config = _load(tmp_path, analysis)
    assert config.analysis.information.params.estimator.method == "binned"


def test_flat_correlation_lists_reach_the_analyzer(tmp_path):
    config = _load(
        tmp_path,
        {"correlation": {"params": {"components": ["noise"], "pairs": ["ee"]}}},
    )
    params = config.analysis.correlation.params
    assert params.selection.components == ["noise"]
    analyzer = CorrelationAnalyzer(params)
    assert analyzer.compute_noise_correlations
    assert not analyzer.compute_total_correlations
    assert analyzer.analyze_EE and not analyzer.analyze_II


def test_programmatic_flat_fields_are_moved_and_conflicts_raise():
    params = InformationAnalysisParams(method="decoder")
    assert params.estimator.method == "decoder"
    with pytest.raises(ValueError, match="conflicts with"):
        InformationAnalysisParams(
            method="decoder", estimator=InformationEstimatorParams(method="binned")
        )


def test_programmatic_mapping_groups_are_typed_and_checked():
    params = InformationAnalysisParams(estimator={"kraskov": {"n_neighbors": 4}})
    assert params.estimator.kraskov.n_neighbors == 4
    ablation = AblationAnalysisParams(selection={"levels": ["layer"]})
    assert isinstance(ablation.selection, AblationSelectionParams)
    assert ablation.selection.levels == ["layer"]
    with pytest.raises(ValueError, match="did you mean 'levels'"):
        AblationAnalysisParams(selection={"level": ["layer"]})


def test_saved_config_replays_the_same_information_settings(tmp_path):
    config = _load(
        tmp_path, {"information": {"params": {"method": "decoder", "n_neighbors": 3}}}
    )
    saved = tmp_path / "saved.yaml"
    saved.write_text(json.dumps({"analysis": dataclasses.asdict(config.analysis)}))
    payload = {**BASE, **yaml.safe_load(saved.read_text())}
    saved.write_text(yaml.safe_dump(payload))
    replayed = load_config(str(saved)).analysis.information.params
    original = config.analysis.information.params
    assert dataclasses.asdict(replayed) == dataclasses.asdict(original)
