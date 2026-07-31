from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import TensorDataset

import dendritic_modeling.analysis as analysis_api
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.input_region_intervention import (
    ImageRegionSpec,
    InputRegionInterventionAnalyzer,
    InputRegionInterventionSettings,
)
from dendritic_modeling.analysis.tools.source_tuning import (
    SourceTuningOptions,
    SourceTuningSupportAnalyzer,
)
from dendritic_modeling.config import (
    AnalysisConfig,
    InputImageRegionConfig,
    InputRegionInterventionAnalysisParams,
    SourceTuningSupportAnalysisParams,
    load_analysis_config,
)


def _datasets():
    datasets = {}
    for index, split in enumerate(("train", "valid", "test")):
        inputs = torch.full((4, 4), float(index))
        labels = torch.tensor([0, 0, 1, 1])
        datasets[split] = TensorDataset(inputs, labels)
    return datasets


def _configured_analysis() -> AnalysisConfig:
    config = AnalysisConfig()
    config.performance_analysis.enabled = False
    config.source_tuning_support_analysis.enabled = True
    config.source_tuning_support_analysis.params = SourceTuningSupportAnalysisParams(
        image_shape=(2, 2),
        image_source_layers=[0],
        reference_split="train",
        evaluation_split="validation",
    )
    config.input_region_intervention_analysis.enabled = True
    config.input_region_intervention_analysis.params = (
        InputRegionInterventionAnalysisParams(
            image_shape=(1, 2, 2),
            regions=[
                InputImageRegionConfig(
                    name="foreground",
                    kind="foreground",
                    foreground_threshold=0.5,
                )
            ],
            reference_split="train",
            evaluation_split="test",
            target_population_polarities=["excitatory", "inhibitory"],
            identity_soma_class_mapping=[0, 1],
            identity_soma_population_name="e0",
            identity_soma_network_layer_index=0,
        )
    )
    return config


def test_registry_and_public_exports_use_canonical_analyzer_names():
    source_spec = get_analyzer_spec("source_tuning_support")
    input_spec = get_analyzer_spec("input_region_intervention")

    assert source_spec is not None
    assert source_spec.config_field == "source_tuning_support_analysis"
    assert source_spec.save_subdir == "source_tuning_support_analysis"
    assert source_spec.supports_training is False
    assert input_spec is not None
    assert input_spec.config_field == "input_region_intervention_analysis"
    assert input_spec.save_subdir == "input_region_intervention"
    assert input_spec.supports_training is False
    assert analysis_api.SourceTuningSupportAnalyzer is SourceTuningSupportAnalyzer
    assert analysis_api.SourceTuningOptions is SourceTuningOptions
    assert (
        analysis_api.InputRegionInterventionAnalyzer is InputRegionInterventionAnalyzer
    )
    assert (
        analysis_api.InputRegionInterventionSettings is InputRegionInterventionSettings
    )
    assert analysis_api.ImageRegionSpec is ImageRegionSpec


@pytest.mark.parametrize(
    ("filename", "field_name"),
    [
        ("source_tuning_support_analysis.yaml", "source_tuning_support_analysis"),
        (
            "input_region_intervention_analysis.yaml",
            "input_region_intervention_analysis",
        ),
    ],
)
def test_example_yaml_loads_as_typed_analysis_config(filename, field_name):
    path = Path(__file__).resolve().parents[1] / "configs" / "examples" / filename

    config = load_analysis_config(str(path))

    section = getattr(config, field_name)
    assert section.enabled is True
    assert section.training is False
    if field_name == "source_tuning_support_analysis":
        options = section.params.to_analyzer_params()
        assert isinstance(options, SourceTuningOptions)
        assert options.image_shape == (28, 28)
        assert section.params.reference_split == "train"
        assert section.params.evaluation_split == "validation"
    else:
        settings = section.params.to_analyzer_params()
        assert isinstance(settings, InputRegionInterventionSettings)
        assert all(isinstance(region, ImageRegionSpec) for region in settings.regions)
        assert settings.reference_split == "train"
        assert settings.evaluation_split == "test"
        assert settings.target_population_polarities == ("excitatory",)


def test_manager_builds_native_options_and_settings(tmp_path):
    manager = AnalysisManager(
        model=torch.nn.Linear(4, 2),
        data=_datasets(),
        analysis_config=_configured_analysis(),
        save_root=str(tmp_path),
    )

    assert isinstance(
        manager.source_tuning_support_analyzer, SourceTuningSupportAnalyzer
    )
    assert isinstance(
        manager.source_tuning_support_analyzer.options, SourceTuningOptions
    )
    assert isinstance(
        manager.input_region_intervention_analyzer,
        InputRegionInterventionAnalyzer,
    )
    assert isinstance(
        manager.input_region_intervention_analyzer.settings,
        InputRegionInterventionSettings,
    )
    assert isinstance(
        manager.input_region_intervention_analyzer.settings.regions[0],
        ImageRegionSpec,
    )
    settings = manager.input_region_intervention_analyzer.settings
    assert settings.target_population_polarities == ("excitatory", "inhibitory")
    assert settings.identity_soma_class_mapping == (0, 1)
    assert settings.identity_soma_population_name == "e0"
    assert settings.identity_soma_network_layer_index == 0


def test_source_tuning_config_accepts_channel_qualified_image_shape():
    options = SourceTuningSupportAnalysisParams(
        image_shape=(3, 32, 32),
        image_source_layers=[0],
    ).to_analyzer_params()

    assert isinstance(options, SourceTuningOptions)
    assert options.image_shape == (3, 32, 32)


def test_manager_routes_declared_reference_and_evaluation_splits(tmp_path):
    datasets = _datasets()
    manager = AnalysisManager(
        model=torch.nn.Linear(4, 2),
        data=datasets,
        analysis_config=_configured_analysis(),
        save_root=str(tmp_path),
    )
    runtime = manager.config.runtime.final

    with patch.object(manager, "source_tuning_support_analyzer") as source_analyzer:
        manager._run_source_tuning_support_analysis(
            filename="final",
            training=False,
            runtime=runtime,
        )
    source_kwargs = source_analyzer.analyze.call_args.kwargs
    assert source_kwargs["reference_data"] is datasets["train"]
    assert source_kwargs["data"] is datasets["valid"]
    assert source_kwargs["reference_split_name"] == "train"
    assert source_kwargs["evaluation_split_name"] == "validation"
    assert source_kwargs["runtime"] is runtime

    with patch.object(
        manager,
        "input_region_intervention_analyzer",
    ) as input_analyzer:
        manager._run_input_region_intervention_analysis(
            filename="final",
            training=False,
            runtime=runtime,
        )
    input_kwargs = input_analyzer.analyze.call_args.kwargs
    assert input_kwargs["reference_dataset"] is datasets["train"]
    assert input_kwargs["test_dataset"] is datasets["test"]
    assert input_kwargs["runtime"] is runtime
    assert input_kwargs["reference_runtime"] is runtime


@pytest.mark.parametrize(
    "config_field",
    ["source_tuning_support_analysis", "input_region_intervention_analysis"],
)
def test_new_analyzers_reject_training_phase_enablement(tmp_path, config_field):
    config = _configured_analysis()
    getattr(config, config_field).training = True

    with pytest.raises(ValueError, match="final-only"):
        AnalysisManager(
            model=Mock(),
            data=_datasets(),
            analysis_config=config,
            save_root=str(tmp_path),
        )
