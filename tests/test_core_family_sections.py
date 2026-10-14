"""model.core sections that the selected core type never reads are rejected."""

from __future__ import annotations

import dataclasses
import re

import pytest
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.config.core_sections import unread_core_sections
from dendritic_modeling.config.model import CoreConfig

POPULATION = {
    "type": "population_network",
    "population_network": {
        "layers": [
            {
                "name": "layer0",
                "readout_population": "e",
                "populations": [
                    {
                        "name": "e",
                        "polarity": "excitatory",
                        "n_neurons": 3,
                        "branch_factors": [2],
                    }
                ],
                "connections": [{"source": "input", "target": "e"}],
            }
        ]
    },
}


def _load(tmp_path, core):
    payload = {
        "data": {"dataset_name": "mnist"},
        "model": {
            "encoder": {"type": "identity", "params": {"input_dim": 4}},
            "core": core,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return load_config(str(path))


@pytest.mark.parametrize(
    "section, value",
    [
        ("sparsity", {"type": "indexed_release"}),
        ("morphology", {"weight_transform": "exp"}),
        ("spatial", {"kernel_size": 5}),
        ("reactivation", {"init_policy": "occupancy_quantile"}),
        ("architecture", {"excitatory_layer_sizes": [8]}),
    ],
)
def test_population_core_rejects_sections_it_never_reads(tmp_path, section, value):
    with pytest.raises(
        ValueError, match=re.escape(f"does not read model.core.{section}")
    ):
        _load(tmp_path, {**POPULATION, section: value})


def test_population_core_accepts_seed_and_default_sections(tmp_path):
    config = _load(
        tmp_path,
        {
            **POPULATION,
            "initialization_seed": 3,
            "spatial": {"kernel_size": 3},
            "dynamics": {},
        },
    )
    assert config.model.core.initialization_seed == 3


def test_saved_population_config_with_every_section_reloads(tmp_path):
    original = _load(tmp_path, POPULATION).model.core
    reloaded = _load(tmp_path, dataclasses.asdict(original)).model.core
    assert dataclasses.asdict(reloaded) == dataclasses.asdict(original)


def test_einet_core_rejects_other_family_payloads(tmp_path):
    core = {"type": "einet", "population_network": POPULATION["population_network"]}
    with pytest.raises(
        ValueError, match=re.escape("does not read model.core.population_network")
    ):
        _load(tmp_path, core)
    with pytest.raises(
        ValueError, match=re.escape("does not read model.core.baseline_rnn")
    ):
        _load(tmp_path, {"type": "einet", "baseline_rnn": {"hidden_size": 4}})


def test_unified_core_reads_only_its_payload():
    core = CoreConfig(type="unified_ei", morphology=CoreConfig().morphology)
    assert unread_core_sections(core, ["morphology"]) == []
    core = CoreConfig(type="unified_ei", sparsity={"type": "indexed"})
    assert unread_core_sections(core, ["sparsity", "unified_ei"]) == ["sparsity"]


def test_saved_record_with_unread_sections_warns(tmp_path, caplog):
    import logging

    saved = dataclasses.asdict(_load(tmp_path, POPULATION).model.core)
    saved["implementation"]["adaptive_initialization"] = False
    with caplog.at_level(logging.WARNING):
        _load(tmp_path, saved)
    assert "Ignoring sections of a saved config" in caplog.text
    assert "model.core.implementation" in caplog.text


def test_alias_type_rejects_a_conflicting_written_morphology(tmp_path):
    core = {
        "type": "dendritic_additive",
        "architecture": {"excitatory_layer_sizes": [3], "inhibitory_layer_sizes": []},
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [0],
        },
    }
    with pytest.raises(ValueError, match=r"fixes morphology\.use_shunting=False"):
        _load(tmp_path, {**core, "morphology": {"use_shunting": True}})
    config = _load(tmp_path, {**core, "morphology": {"use_shunting": False}})
    assert config.model.core.morphology.use_shunting is False
    assert _load(tmp_path, core).model.core.morphology.use_shunting is False


_EINET_SECTIONS = {
    "architecture": {"excitatory_layer_sizes": [3], "inhibitory_layer_sizes": []},
    "connectivity": {
        "ee_synapses_per_branch_per_layer": [2],
        "ie_synapses_per_branch_per_layer": [0],
    },
}


@pytest.mark.parametrize(
    "core_type",
    [
        "einet",
        "dendritic_shunting",
        "flat_additive",
        "dendritic_signed",
        "ss_mlp",
        "tensor_map",
        "point_mlp",
        "dendritic_alexnet",
    ],
)
def test_cores_without_a_spatial_convolution_reject_the_spatial_section(
    tmp_path, core_type
):
    core = {"type": core_type}
    if core_type != "dendritic_alexnet":  # it sizes its blocks from vision
        core.update(_EINET_SECTIONS)
    with pytest.raises(ValueError, match=re.escape("does not read model.core.spatial")):
        _load(tmp_path, {**core, "spatial": {"kernel_size": 5}})
    assert _load(tmp_path, {**core, "spatial": {"kernel_size": 3}})


def test_dendritic_convolutions_read_every_spatial_option(tmp_path):
    spatial = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 1,
        "input_transform": "signed_split",
        "input_scale": 0.5,
        "output_adapter_mode": "learned_per_channel",
        "output_initial_scale": 2.0,
        "output_initial_threshold": 0.1,
    }
    for core_type in ("hierarchical_dendritic_conv", "spatial_dendritic_conv"):
        config = _load(
            tmp_path, {"type": core_type, **_EINET_SECTIONS, "spatial": spatial}
        )
        assert dataclasses.asdict(config.model.core.spatial) == spatial


@pytest.mark.parametrize(
    "core_type",
    ["sparse_active_matched_point_conv", "spatial_sparse_active_matched_point"],
)
def test_sparse_point_convolutions_reject_the_spatial_options_they_ignore(
    tmp_path, core_type
):
    core = {"type": core_type, **_EINET_SECTIONS}
    read = {"kernel_size": 5, "stride": 2, "padding": 1}
    assert _load(tmp_path, {**core, "spatial": read}).model.core.spatial.stride == 2
    with pytest.raises(
        ValueError,
        match=re.escape(
            "does not read model.core.spatial.input_transform, "
            "model.core.spatial.input_scale"
        ),
    ):
        _load(
            tmp_path,
            {
                **core,
                "spatial": {**read, "input_transform": "relu", "input_scale": 2.0},
            },
        )


def test_saved_records_and_plugin_core_types_keep_their_spatial_section(
    tmp_path, caplog
):
    import logging

    saved = dataclasses.asdict(
        _load(tmp_path, {"type": "einet", **_EINET_SECTIONS}).model.core
    )
    saved["spatial"]["kernel_size"] = 5
    with caplog.at_level(logging.WARNING):
        _load(tmp_path, saved)
    assert "does not read model.core.spatial" in caplog.text
    # Core types that plugins register read sections the loader cannot check.
    plugin = _load(tmp_path, {"type": "plugin_core", "spatial": {"kernel_size": 5}})
    assert plugin.model.core.spatial.kernel_size == 5


def test_the_factory_builds_exactly_the_builtin_core_types():
    from dendritic_modeling.config.core_sections import BUILTIN_CORE_TYPES
    from dendritic_modeling.networks.architectures import factory

    assert set(factory._BUILTIN_ARCHITECTURE_BUILDERS) == set(BUILTIN_CORE_TYPES)
