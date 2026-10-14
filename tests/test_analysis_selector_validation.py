"""Unknown analysis selector values raise instead of selecting nothing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from dendritic_modeling.analysis.core.ablation import AblationAnalyzer
from dendritic_modeling.analysis.tools.correlation_analysis import CorrelationAnalyzer
from dendritic_modeling.config import load_analysis_config
from dendritic_modeling.config.analysis import (
    AblationAnalysisParams,
    AblationMethodsParams,
    AblationSelectionParams,
    CorrelationAnalysisParams,
    CorrelationComputeParams,
    CorrelationSamplingParams,
    CorrelationSelectionParams,
)


def _load(tmp_path, analysis: dict):
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump({"analysis": analysis}))
    return load_analysis_config(str(path))


@pytest.mark.parametrize(
    "selection,match",
    [
        ({"levels": ["layers"]}, r"levels lists 'layers' \(did you mean 'layer'\?\)"),
        ({"targets": ["excitatory"]}, r"targets lists 'excitatory'"),
        ({"metrics": ["acuracy"]}, r"did you mean 'accuracy'"),
    ],
)
def test_misspelled_ablation_selectors_raise_at_load(tmp_path, selection, match):
    with pytest.raises(ValueError, match=match):
        _load(
            tmp_path,
            {"ablation": {"enabled": True, "params": {"selection": selection}}},
        )


def test_ablation_selector_error_lists_the_valid_values():
    with pytest.raises(ValueError) as error:
        AblationSelectionParams(targets=["dendrites"])
    assert "all_synapses, excitation, inhibition, upstream" in str(error.value)


def test_ablation_aliases_and_spellings_still_select(tmp_path):
    config = _load(
        tmp_path,
        {
            "ablation": {
                "enabled": True,
                "params": {
                    "selection": {
                        "levels": ["Layer", " compartment "],
                        "targets": ["exc", "inh", "all", "branch-input"],
                        "metrics": ["pred-mi", "cosine", "log likelihood", ""],
                    }
                },
            }
        },
    )
    analyzer = AblationAnalyzer(config.ablation.params)
    assert analyzer.layer_ablation and analyzer.compartment_ablation
    assert analyzer.ablate_excitation and analyzer.ablate_inhibition
    assert analyzer.ablate_all_synapses and analyzer.ablate_upstream
    assert analyzer.compute_pred_label_mi and analyzer.cosine_similarity
    assert analyzer.categorical_ll
    assert not (analyzer.accuracy or analyzer.auc or analyzer.mse)


def test_ablation_analyzer_checks_params_built_in_code():
    params = SimpleNamespace(
        selection=SimpleNamespace(levels=["layer"], targets=["E"], metrics=[]),
        methods=SimpleNamespace(ablation_methods=["lesion"], clamp_statistic="mean"),
    )
    with pytest.raises(ValueError, match="targets lists 'E'"):
        AblationAnalyzer(params)


def test_default_ablation_params_still_construct():
    analyzer = AblationAnalyzer(AblationAnalysisParams())
    assert not analyzer.layer_ablation
    assert analyzer.ablation_methods == ["lesion"]


@pytest.mark.parametrize(
    "methods,match",
    [
        ({"ablation_methods": ["lesion", "zero"]}, "lists 'zero'"),
        ({"clamp_statistic": "median"}, "clamp_statistic='median'"),
    ],
)
def test_unknown_ablation_methods_raise(methods, match):
    with pytest.raises(ValueError, match=match):
        AblationMethodsParams(**methods)


@pytest.mark.parametrize(
    "selection,match",
    [
        ({"components": ["totl"]}, r"components lists 'totl'"),
        ({"pairs": ["EE", "EO"]}, r"pairs lists 'EO'"),
        ({"scopes": ["layers"]}, r"did you mean 'layer'"),
    ],
)
def test_misspelled_correlation_selectors_raise_at_load(tmp_path, selection, match):
    with pytest.raises(ValueError, match=match):
        _load(
            tmp_path,
            {"correlation": {"enabled": True, "params": {"selection": selection}}},
        )


def test_flat_correlation_selectors_are_checked_too():
    with pytest.raises(ValueError, match="scopes lists 'neurons'"):
        CorrelationAnalysisParams(scopes=["neurons"])


def test_correlation_aliases_still_select():
    params = CorrelationAnalysisParams(
        selection=CorrelationSelectionParams(
            components=["all", "tuning_curves"],
            pairs=["ee", "exc-vout", "out_out"],
            scopes=["per_layer"],
        )
    )
    analyzer = CorrelationAnalyzer(params)
    assert analyzer.compute_total_correlations and analyzer.compute_tuning_curves
    assert not analyzer.compute_noise_correlations
    assert not analyzer.compute_signal_correlations
    assert analyzer.analyze_EE and analyzer.analyze_E_output
    assert analyzer.analyze_output_output
    assert not (analyzer.analyze_II or analyzer.analyze_EI or analyzer.analyze_I_output)
    assert analyzer.per_layer_analysis
    assert not (analyzer.per_einet_analysis or analyzer.per_neuron_analysis)


def test_correlation_analyzer_checks_params_built_in_code():
    params = SimpleNamespace(
        selection=SimpleNamespace(components=["noise", "signals"], pairs=[], scopes=[])
    )
    with pytest.raises(ValueError, match="components lists 'signals'"):
        CorrelationAnalyzer(params)


def test_correlation_computation_level_and_sampling_are_checked():
    with pytest.raises(ValueError, match="computation_level='layer'"):
        CorrelationComputeParams(computation_level="layer")
    with pytest.raises(ValueError, match="computation_level='branch'"):
        CorrelationAnalysisParams(computation_level="branch")
    with pytest.raises(ValueError, match="sampling_strategy='importance'"):
        CorrelationSamplingParams(sampling_strategy="importance")
    assert (
        CorrelationComputeParams(computation_level="Layer-Branch").computation_level
        == "layer_branch"
    )
    analyzer = CorrelationAnalyzer(
        SimpleNamespace(compute=SimpleNamespace(computation_level="All_Branch"))
    )
    assert analyzer.computation_level == "all_branch"


def test_default_correlation_params_keep_every_option():
    analyzer = CorrelationAnalyzer(CorrelationAnalysisParams())
    assert analyzer.computation_level == "layer_branch"
    assert analyzer.sampling_strategy == "stratified"
    assert analyzer.compute_total_correlations and analyzer.compute_noise_correlations
    assert analyzer.analyze_EE and analyzer.analyze_output_output
    assert analyzer.per_layer_analysis and analyzer.per_einet_analysis
