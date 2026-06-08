"""Tests for model architecture validation utilities."""

from types import SimpleNamespace

from dendritic_modeling.utils.model_architecture_validator import (
    ModelArchitectureValidator,
    _get_config_value,
    validate_model_architecture,
)


class ConfigNode(SimpleNamespace):
    """Simple attribute-based config object with dict-like get support."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_valid_config(
    *,
    dataset_name="mnist",
    task_params=None,
    transfer_overrides=None,
    architecture_overrides=None,
    connectivity_overrides=None,
):
    transfer_params = {
        "input_mode": 0,
        "independent_pathways": False,
        "excitatory_dim": 784,
        "inhibitory_dim": 784,
    }
    if transfer_overrides:
        transfer_params.update(transfer_overrides)

    architecture = {
        "transfer_params": transfer_params,
        "excitatory_layer_sizes": [8, 4],
        "inhibitory_layer_sizes": [4, 2],
    }
    if architecture_overrides:
        architecture.update(architecture_overrides)

    connectivity = {
        "excitatory_branch_factors": [2, 2],
        "inhibitory_branch_factors": [1, 1],
        "ee_synapses_per_branch_per_layer": [5, 5],
        "ei_synapses_per_branch_per_layer": [2, 2],
        "ie_synapses_per_branch_per_layer": [2, 2],
        "ii_synapses_per_branch_per_layer": [2, 2],
    }
    if connectivity_overrides:
        connectivity.update(connectivity_overrides)

    return ConfigNode(
        model=ConfigNode(
            core=ConfigNode(
                architecture=architecture,
                connectivity=connectivity,
                morphology={},
            )
        ),
        data=ConfigNode(dataset_name=dataset_name),
        task={"parameters": task_params or {}},
    )


def test_get_config_value_handles_dicts_and_objects():
    assert _get_config_value({"alpha": 1}, "alpha") == 1
    assert _get_config_value({"alpha": 1}, "missing", 99) == 99

    obj = ConfigNode(alpha=2)
    assert _get_config_value(obj, "alpha") == 2
    assert _get_config_value(obj, "missing", 77) == 77


def test_validate_model_architecture_accepts_valid_configuration():
    config = _make_valid_config()
    assert validate_model_architecture(config) is True


def test_validate_model_architecture_allows_input_mode0_independent_pathways():
    config = _make_valid_config(
        transfer_overrides={
            "input_mode": 0,
            "independent_pathways": True,
            "excitatory_dim": 400,
            "inhibitory_dim": 384,
        }
    )

    assert validate_model_architecture(config) is True


def test_validate_model_architecture_detects_invalid_input_mode():
    config = _make_valid_config(transfer_overrides={"input_mode": 2})
    validator = ModelArchitectureValidator(config)

    assert validator.validate() is False
    assert any("input_mode must be 0 or 1" in err for err in validator.errors)


def test_validate_model_architecture_detects_empty_required_synapse_lists():
    config = _make_valid_config(
        connectivity_overrides={
            "ee_synapses_per_branch_per_layer": [],
            "ei_synapses_per_branch_per_layer": [],
            "ie_synapses_per_branch_per_layer": [],
            "ii_synapses_per_branch_per_layer": [],
        }
    )
    validator = ModelArchitectureValidator(config)

    assert validator.validate() is False
    errors = "\n".join(validator.errors)
    assert "ee_synapses_per_branch_per_layer cannot be empty" in errors
    assert "ei_synapses_per_branch_per_layer cannot be empty" in errors
    assert "ie_synapses_per_branch_per_layer cannot be empty" in errors
    assert "ii_synapses_per_branch_per_layer cannot be empty" in errors


def test_validate_model_architecture_reports_high_synapse_warning():
    config = _make_valid_config(
        connectivity_overrides={"ee_synapses_per_branch_per_layer": [150, 5]}
    )
    validator = ModelArchitectureValidator(config)

    assert validator.validate() is True
    assert any("very high" in warning for warning in validator.warnings)


def test_extract_parameters_uses_multixor_task_bits_for_input_dim():
    config = _make_valid_config(
        dataset_name="multixor",
        task_params={"n_bits": 6},
        transfer_overrides={"excitatory_dim": None, "inhibitory_dim": None},
    )
    validator = ModelArchitectureValidator(config)
    validator.extract_parameters()

    assert validator.input_dim == 12


def test_validate_model_architecture_with_plain_dict_fails_cleanly():
    # Dict-only config exercises fallback paths where model/core sections are absent.
    assert validate_model_architecture({}) is False
