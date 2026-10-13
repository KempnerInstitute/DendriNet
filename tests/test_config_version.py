"""``config_version`` names the configuration format a file is written for."""

from __future__ import annotations

import dataclasses

import pytest

from dendritic_modeling.config import CONFIG_VERSION, config_from_mapping


def test_files_without_a_version_load_with_the_current_meaning():
    config = config_from_mapping({"experiment": {"seed": 1}})
    assert config.config_version == CONFIG_VERSION
    assert dataclasses.asdict(config)["config_version"] == CONFIG_VERSION


def test_the_current_version_loads():
    config = config_from_mapping({"config_version": CONFIG_VERSION})
    assert config.config_version == CONFIG_VERSION


def test_earlier_versions_must_be_upgraded():
    with pytest.raises(ValueError, match="dendritic-upgrade-config"):
        config_from_mapping({"config_version": CONFIG_VERSION - 1})


def test_later_versions_need_a_newer_library():
    with pytest.raises(ValueError, match="newer than this library"):
        config_from_mapping({"config_version": CONFIG_VERSION + 1})


@pytest.mark.parametrize("value", ["2", 2.0, True])
def test_the_version_is_an_integer(value):
    with pytest.raises(ValueError, match="config_version must be an integer"):
        config_from_mapping({"config_version": value})


@pytest.mark.parametrize("with_distributed", [False, True])
def test_a_top_level_fsdp_block_is_rejected(with_distributed):
    """Its settings were never read next to a ``distributed`` section."""
    data = {"fsdp": {"mixed_precision_dtype": "float32"}}
    if with_distributed:
        data["distributed"] = {"mode": "fsdp"}
    with pytest.raises(ValueError, match=r"distributed\.fsdp"):
        config_from_mapping(data)


def test_a_partly_upgraded_file_does_not_load():
    with pytest.raises(
        ValueError, match=r"only partly upgraded.*model\.core\.recurrent"
    ):
        config_from_mapping(
            {
                "config_version": CONFIG_VERSION,
                "unresolved_migration": ["model.core.recurrent: removed stack"],
            }
        )
