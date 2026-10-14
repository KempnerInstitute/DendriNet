"""Analysis sections use their analyzer names; the older names still load."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.config.config import _map_analysis_config


def test_every_analysis_field_is_its_registry_name():
    fields = {item.name for item in dataclasses.fields(AnalysisConfig)} - {
        "runtime",
        "extensions",
        "require_completion",
    }
    registry = {
        name: spec.config_field for name, spec in get_registered_analyzers().items()
    }
    assert all(name == field for name, field in registry.items())
    assert set(registry) == fields


def test_old_section_names_load_with_one_warning(caplog):
    with caplog.at_level(logging.WARNING):
        mapped = _map_analysis_config(
            {
                "information_analysis": {"enabled": True},
                "weight_analysis": {"enabled": True},
                "local_rule_component_analysis": {"enabled": False},
            }
        )
    assert mapped["information"].enabled and mapped["weight"].enabled
    assert "local_rule_components" in mapped
    warnings = [r for r in caplog.records if "were renamed" in r.getMessage()]
    assert len(warnings) == 1
    assert "information_analysis -> information" in warnings[0].getMessage()


def test_old_and_new_names_of_one_section_raise():
    with pytest.raises(ValueError, match="configure the same analysis"):
        _map_analysis_config(
            {"information": {"enabled": True}, "information_analysis": {}}
        )


def test_performance_splits_use_the_validation_spelling():
    from dendritic_modeling.analysis.managers.analysis_manager import AnalysisManager
    from dendritic_modeling.config.analysis import PerformanceAnalysisConfig

    config = PerformanceAnalysisConfig(
        training_splits=["valid"], final_splits=["validation"]
    )
    assert config.training_splits == ["validation"] == config.final_splits
    # The metrics keep reporting the validation split under "valid".
    assert AnalysisManager._performance_splits(config, training=True) == ["valid"]
    with pytest.raises(ValueError, match="must list splits"):
        PerformanceAnalysisConfig(final_splits=["dev"])
