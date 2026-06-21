"""Tests for model architecture validation utilities."""

from types import SimpleNamespace

from dendritic_modeling.utils.model_architecture_validator import (
    ModelArchitectureValidator,
    _build_network_config,
    _get_config_value,
    _has_inhibitory_neurons,
    _ie_inhibitory_source_dim,
    _initial_layer_input_dims,
    _layer_input_dims,
    _layer_value,
    _previous_layer_inhibitory_size,
    _resolve_input_dim,
    _same_layer_inhibitory_size,
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


def test_build_network_config_preserves_section_merge_order():
    config = ConfigNode(
        model=ConfigNode(
            core=ConfigNode(
                architecture={"shared": "architecture", "hidden_dim": 8},
                connectivity=ConfigNode(shared="connectivity", num_synapses=9),
                transfer={"input_mode": 1},
                morphology=ConfigNode(shared="morphology", use_shunting=True),
            )
        )
    )

    network_config = _build_network_config(config)

    assert network_config == {
        "hidden_dim": 8,
        "num_synapses": 9,
        "shared": "morphology",
        "transfer_params": {"input_mode": 1},
        "use_shunting": True,
    }


def test_build_network_config_returns_empty_for_plain_dict_config():
    assert _build_network_config({}) == {}


def test_resolve_input_dim_preserves_dataset_and_fallback_priority():
    assert (
        _resolve_input_dim(
            dataset_name="mnist",
            task_params={},
            network_config={"input_dim": 12},
            transfer_params={"excitatory_dim": 13},
        )
        == 784
    )
    assert (
        _resolve_input_dim(
            dataset_name="multixor",
            task_params={"n_bits": 5},
            network_config={},
            transfer_params={},
        )
        == 10
    )
    assert (
        _resolve_input_dim(
            dataset_name="custom",
            task_params={"input_dim": 22},
            network_config={"input_dim": 21},
            transfer_params={"excitatory_dim": 23},
        )
        == 21
    )
    assert (
        _resolve_input_dim(
            dataset_name="custom",
            task_params={"input_dim": 22},
            network_config={},
            transfer_params={"excitatory_dim": 23},
        )
        == 22
    )
    assert (
        _resolve_input_dim(
            dataset_name="custom",
            task_params={},
            network_config={},
            transfer_params={"excitatory_dim": 23},
        )
        == 23
    )
    assert (
        _resolve_input_dim(
            dataset_name="custom",
            task_params={},
            network_config={},
            transfer_params={},
        )
        == 64
    )


def test_layer_value_reuses_last_value_or_default():
    assert _layer_value([3, 5], 0) == 3
    assert _layer_value([3, 5], 1) == 5
    assert _layer_value([3, 5], 4) == 5
    assert _layer_value([], 0) == 1
    assert _layer_value([], 0, default=7) == 7


def test_has_inhibitory_neurons_requires_positive_layer_size():
    assert not _has_inhibitory_neurons([])
    assert not _has_inhibitory_neurons([0, None, -1])
    assert _has_inhibitory_neurons([0, 2])


def test_same_and_previous_inhibitory_size_helpers_preserve_fallbacks():
    assert _same_layer_inhibitory_size([0, 3], 0) == 0
    assert _same_layer_inhibitory_size([0, 3], 1) == 3
    assert _same_layer_inhibitory_size([0, 3], 4) == 0

    assert _previous_layer_inhibitory_size([5, 7], 0) == 0
    assert _previous_layer_inhibitory_size([5, 7], 1) == 5
    assert _previous_layer_inhibitory_size([5, 7], 2) == 7
    assert _previous_layer_inhibitory_size([5, 7], 3) == 0


def test_initial_layer_input_dims_preserve_shared_and_independent_pathways():
    assert _initial_layer_input_dims(
        input_dim=12,
        independent_pathways=False,
        excitatory_dim=4,
        inhibitory_dim=6,
    ) == (12, 12)
    assert _initial_layer_input_dims(
        input_dim=12,
        independent_pathways=True,
        excitatory_dim=4,
        inhibitory_dim=6,
    ) == (4, 6)
    assert _initial_layer_input_dims(
        input_dim=12,
        independent_pathways=True,
        excitatory_dim=None,
        inhibitory_dim=0,
    ) == (6, 6)


def test_layer_input_dims_use_first_layer_and_previous_layer_sources():
    assert _layer_input_dims(
        layer_idx=0,
        input_dim=20,
        independent_pathways=True,
        excitatory_dim=8,
        inhibitory_dim=9,
        excitatory_layer_sizes=[4, 3],
        inhibitory_layer_sizes=[2],
    ) == (8, 9)
    assert _layer_input_dims(
        layer_idx=1,
        input_dim=20,
        independent_pathways=True,
        excitatory_dim=8,
        inhibitory_dim=9,
        excitatory_layer_sizes=[4, 3],
        inhibitory_layer_sizes=[2],
    ) == (4, 2)
    assert _layer_input_dims(
        layer_idx=2,
        input_dim=20,
        independent_pathways=True,
        excitatory_dim=8,
        inhibitory_dim=9,
        excitatory_layer_sizes=[4, 3],
        inhibitory_layer_sizes=[2],
    ) == (3, 0)


def test_ie_inhibitory_source_dim_preserves_input_mode_rules():
    assert (
        _ie_inhibitory_source_dim(
            layer_idx=1,
            input_mode=1,
            independent_pathways=True,
            input_dim=20,
            excitatory_dim=8,
            inhibitory_dim=9,
            inhibitory_layer_sizes=[2, 3],
        )
        == 9
    )
    assert (
        _ie_inhibitory_source_dim(
            layer_idx=1,
            input_mode=1,
            independent_pathways=True,
            input_dim=20,
            excitatory_dim=8,
            inhibitory_dim=None,
            inhibitory_layer_sizes=[2, 3],
        )
        == 10
    )
    assert (
        _ie_inhibitory_source_dim(
            layer_idx=1,
            input_mode=1,
            independent_pathways=False,
            input_dim=20,
            excitatory_dim=8,
            inhibitory_dim=9,
            inhibitory_layer_sizes=[2, 3],
        )
        == 8
    )
    assert (
        _ie_inhibitory_source_dim(
            layer_idx=0,
            input_mode=0,
            independent_pathways=False,
            input_dim=20,
            excitatory_dim=None,
            inhibitory_dim=None,
            inhibitory_layer_sizes=[2, 3],
        )
        == 2
    )
    assert (
        _ie_inhibitory_source_dim(
            layer_idx=2,
            input_mode=0,
            independent_pathways=False,
            input_dim=20,
            excitatory_dim=None,
            inhibitory_dim=None,
            inhibitory_layer_sizes=[2, 3],
        )
        == 3
    )


def test_validate_model_architecture_accepts_valid_configuration():
    config = _make_valid_config()
    assert validate_model_architecture(config) is True


def test_validation_checks_preserve_legacy_order():
    validator = ModelArchitectureValidator(_make_valid_config())

    assert [check.__name__ for check in validator._validation_checks()] == [
        "check_input_mode_consistency",
        "check_layer_dimensions",
        "check_synapses_per_branch",
        "check_topk_constraints",
        "check_branch_factors",
        "check_empty_lists",
        "check_transfer_dimensions",
        "check_biological_plausibility",
    ]


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
