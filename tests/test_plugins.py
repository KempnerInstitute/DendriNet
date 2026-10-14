"""Plugins register components through DendriNet's registries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import dendritic_modeling.plugins as plugins
from dendritic_modeling.analysis.managers.analysis_plan import (
    analyzer_initializers,
    extension_runners,
)
from dendritic_modeling.analysis.registry import register_analyzer, unregister_analyzer
from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.models.builders import (
    register_model_builder,
    resolve_model_builder,
    unregister_model_builder,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.factory import (
    TRAINER_STRATEGY_REGISTRY,
    get_available_trainer_strategies,
    get_trainer,
)
from dendritic_modeling.training.strategies import Trainer
from dendritic_modeling.utils.runtime_environment import runtime_environment


class _EntryPoint:
    def __init__(self, name, register, *, fail_import=False):
        self.name = name
        self.value = f"tests.fake_plugin_{name}:register"
        self.dist = SimpleNamespace(name=f"fake-{name}", version="1.2.3")
        self._register = register
        self._fail_import = fail_import

    def load(self):
        if self._fail_import:
            raise ImportError("broken plugin module")
        return self._register


@pytest.fixture
def fresh_plugins(monkeypatch):
    """Empty plugin state and a controllable set of installed entry points."""
    installed: list[_EntryPoint] = []
    monkeypatch.setattr(plugins, "_LOADED", {})
    monkeypatch.setattr(plugins, "_FAILED", {})
    monkeypatch.setitem(plugins._STATE, "discovered", False)
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda group: list(installed) if group == plugins.ENTRY_POINT_GROUP else [],
    )
    monkeypatch.delenv(plugins.PLUGINS_ENVIRONMENT_VARIABLE, raising=False)
    return installed


def test_installed_plugins_load_once_in_name_order(fresh_plugins):
    calls = []
    fresh_plugins.append(_EntryPoint("zeta", lambda: calls.append("zeta")))
    fresh_plugins.append(_EntryPoint("alpha", lambda: calls.append("alpha")))
    loaded = plugins.load_plugins()
    assert [record.name for record in loaded] == ["alpha", "zeta"]
    assert calls == ["alpha", "zeta"]
    assert loaded[0].distribution == "fake-alpha" and loaded[0].version == "1.2.3"
    plugins.load_plugins()
    assert calls == ["alpha", "zeta"]


def test_a_plugin_that_fails_raises_instead_of_being_skipped(fresh_plugins):
    def broken():
        raise RuntimeError("registration bug")

    fresh_plugins.append(_EntryPoint("broken", broken))
    with pytest.raises(plugins.PluginLoadError, match="registration bug"):
        plugins.load_plugins()


def test_a_plugin_that_cannot_be_imported_raises(fresh_plugins):
    fresh_plugins.append(_EntryPoint("missing", lambda: None, fail_import=True))
    with pytest.raises(plugins.PluginLoadError, match="could not be imported"):
        plugins.load_plugins()


@pytest.mark.parametrize("failure", ["register", "import"])
def test_a_failed_plugin_keeps_failing_later_loads(fresh_plugins, failure):
    """Its components may be partly registered: no later load may succeed."""
    calls = []

    def broken():
        calls.append("registered part")
        raise RuntimeError("registration bug")

    fresh_plugins.append(_EntryPoint("broken", broken, fail_import=failure == "import"))
    with pytest.raises(plugins.PluginLoadError):
        plugins.load_plugins()
    for _attempt in range(2):
        with pytest.raises(plugins.PluginLoadError, match="'broken'"):
            plugins.load_plugins()
        with pytest.raises(plugins.PluginLoadError, match="earlier in this process"):
            config_from_mapping({"experiment": {"seed": 1}})
    assert calls == ([] if failure == "import" else ["registered part"])
    assert plugins.loaded_plugins() == ()


def test_a_failed_activation_keeps_failing_later_loads(fresh_plugins):
    def broken():
        raise RuntimeError("registration bug")

    with pytest.raises(plugins.PluginLoadError, match="registration bug"):
        plugins.activate_plugin("notebook_plugin", broken)
    with pytest.raises(plugins.PluginLoadError, match="'notebook_plugin'"):
        plugins.load_plugins()
    with pytest.raises(plugins.PluginLoadError, match="'notebook_plugin'"):
        plugins.activate_plugin("other_plugin", lambda: None)


def test_automatic_loading_can_be_disabled(fresh_plugins, monkeypatch):
    calls = []
    fresh_plugins.append(_EntryPoint("alpha", lambda: calls.append("alpha")))
    monkeypatch.setenv(plugins.PLUGINS_ENVIRONMENT_VARIABLE, "0")
    assert plugins.load_plugins() == ()
    assert calls == []


def test_activated_plugins_are_recorded_in_the_run_environment(fresh_plugins):
    record = plugins.activate_plugin("notebook_plugin", lambda: None)
    assert record in plugins.loaded_plugins()
    with pytest.raises(ValueError, match="already loaded"):
        plugins.activate_plugin("notebook_plugin", lambda: None)
    recorded = runtime_environment()["plugins"]
    assert [item["name"] for item in recorded] == ["notebook_plugin"]
    assert recorded[0]["git"] == {"available": False}


class _Tiny(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def test_a_model_builder_builds_the_components_for_configs_it_applies_to(
    fresh_plugins,
):
    def build(model_config):
        encoder = nn.Identity()
        encoder.output_dim = 4
        return encoder, _Tiny(4, 3), _Tiny(3, 2), encoder

    register_model_builder(
        "tiny", applies=lambda config: config.task == "classification", build=build
    )
    try:
        model_config = config_from_mapping({}).model
        assert resolve_model_builder(model_config).name == "tiny"
        model, _ = initialize_model(model_config)
        assert model(torch.zeros(5, 4)).shape == (5, 2)
        register_model_builder("again", applies=lambda config: True, build=build)
        with pytest.raises(ValueError, match="Several model builders"):
            resolve_model_builder(model_config)
    finally:
        unregister_model_builder("tiny")
        unregister_model_builder("again")
    assert resolve_model_builder(model_config) is None


def _registries():
    """(register, lookup, unregister) for each registry a plugin can extend."""
    from dendritic_modeling.analysis.registry import get_analyzer_spec
    from dendritic_modeling.config.extensions import (
        apply_config_migrations,
        config_section_builder,
        register_config_migration,
        register_config_section,
        unregister_config_migration,
        unregister_config_section,
    )
    from dendritic_modeling.datasets.registry import (
        get_dataset_spec,
        register_dataset,
        unregister_dataset,
    )
    from dendritic_modeling.datasets.sequence.builders.registry import (
        has_sequence_dataset_builder,
        register_sequence_dataset_builder,
        unregister_sequence_dataset_builder,
    )
    from dendritic_modeling.networks.activations.factory import (
        get_activation_builder,
        register_activation_builder,
        unregister_activation_builder,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
        AdditiveRule,
        get_additive_rule,
        normalize_additive_mode,
        register_additive_rule,
        unregister_additive_rule,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
        get_sparse_layer,
        register_sparse_layer,
        unregister_sparse_layer,
    )
    from dendritic_modeling.networks.architectures.registry import (
        get_architecture_spec,
        register_architecture,
        unregister_architecture,
    )
    from dendritic_modeling.training.loss.functions import (
        MSELoss,
        get_loss_function,
        register_loss_function,
        unregister_loss_function,
    )
    from dendritic_modeling.training.optimizers.factory import (
        create_optimizer,
        register_optimizer,
        unregister_optimizer,
    )

    name = "from_plugin"
    section = f"analysis.{name}"
    return {
        "architecture": (
            lambda: register_architecture(name, lambda p, **_: nn.Identity()),
            lambda: get_architecture_spec(name) is not None,
            lambda: unregister_architecture(name),
        ),
        "dataset": (
            lambda: register_dataset(name, lambda task: (None, None, None)),
            lambda: get_dataset_spec(name) is not None,
            lambda: unregister_dataset(name),
        ),
        "activation": (
            lambda: register_activation_builder(name, lambda **_: nn.Identity()),
            lambda: get_activation_builder(name) is not None,
            lambda: unregister_activation_builder(name),
        ),
        "sparse_layer": (
            lambda: register_sparse_layer(name, _Sparse),
            lambda: isinstance(get_sparse_layer(name, 4, 2, 2), _Sparse),
            lambda: unregister_sparse_layer(name),
        ),
        "loss": (
            lambda: register_loss_function(name, MSELoss),
            lambda: isinstance(get_loss_function(name), MSELoss),
            lambda: unregister_loss_function(name),
        ),
        "optimizer": (
            lambda: register_optimizer(
                name, lambda params, config: torch.optim.SGD(params, lr=config.lr)
            ),
            lambda: isinstance(
                create_optimizer(
                    nn.Linear(2, 2).parameters(), SimpleNamespace(name=name, lr=0.1)
                ),
                torch.optim.SGD,
            ),
            lambda: unregister_optimizer(name),
        ),
        "analyzer": (
            lambda: register_analyzer(
                name,
                config_field=name,
                class_path="torch.nn.Identity",
                attribute_name=f"{name}_analyzer",
                save_subdir=name,
            ),
            lambda: get_analyzer_spec(name) is not None,
            lambda: unregister_analyzer(name),
        ),
        "additive_rule": (
            lambda: register_additive_rule(
                name, AdditiveRule(voltage=lambda layer, **_: None)
            ),
            lambda: get_additive_rule(name) is not None
            and normalize_additive_mode(name) == name,
            lambda: unregister_additive_rule(name),
        ),
        "model_builder": (
            lambda: register_model_builder(
                name,
                applies=lambda config: getattr(config, "plugin_probe", False),
                build=lambda config: None,
            ),
            lambda: resolve_model_builder(SimpleNamespace(plugin_probe=True))
            is not None,
            lambda: unregister_model_builder(name),
        ),
        "sequence_dataset": (
            lambda: register_sequence_dataset_builder(name, lambda *_, **__: None),
            lambda: has_sequence_dataset_builder(name),
            lambda: unregister_sequence_dataset_builder(name),
        ),
        "config_section": (
            lambda: register_config_section(section, lambda data: data),
            lambda: config_section_builder(section) is not None,
            lambda: unregister_config_section(section),
        ),
        "config_migration": (
            lambda: register_config_migration(name, lambda data: ["probe"]),
            lambda: apply_config_migrations({})[1] == [f"{name}: probe"],
            lambda: unregister_config_migration(name),
        ),
    }


def _listings():
    """The functions that list each registry's names, by registry.

    Each returns names that include the ``from_plugin`` component of
    :func:`_registries` once its plugin has loaded.
    """
    from dendritic_modeling.analysis.registry import (
        get_extension_analyzer_names,
        get_registered_analyzer_names,
        get_registered_analyzers,
    )
    from dendritic_modeling.config.extensions import (
        registered_config_migrations,
        registered_config_sections,
        registered_named_sections,
    )
    from dendritic_modeling.datasets.registry import get_registered_dataset_names
    from dendritic_modeling.datasets.sequence.builders.registry import (
        get_registered_sequence_dataset_names,
    )
    from dendritic_modeling.models.builders import registered_model_builders
    from dendritic_modeling.networks.activations.factory import (
        get_registered_activation_types,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
        available_additive_modes,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
        get_available_sparse_types,
    )
    from dendritic_modeling.networks.architectures.factory import (
        get_available_architectures,
    )
    from dendritic_modeling.networks.architectures.registry import (
        get_registered_architecture_names,
    )
    from dendritic_modeling.training.loss.functions import get_available_loss_functions
    from dendritic_modeling.training.optimizers.factory import get_available_optimizers

    return {
        "activation": {
            "get_registered_activation_types": get_registered_activation_types
        },
        "additive_rule": {"available_additive_modes": available_additive_modes},
        "analyzer": {
            "get_registered_analyzer_names": get_registered_analyzer_names,
            "get_registered_analyzers": get_registered_analyzers,
            "get_extension_analyzer_names": get_extension_analyzer_names,
        },
        "architecture": {
            "get_registered_architecture_names": get_registered_architecture_names,
            "get_available_architectures": get_available_architectures,
        },
        "config_migration": {
            "registered_config_migrations": registered_config_migrations
        },
        "config_section": {
            "registered_named_sections": lambda: registered_named_sections("analysis"),
            "registered_config_sections": lambda: [
                path.removeprefix("analysis.") for path in registered_config_sections()
            ],
        },
        "dataset": {"get_registered_dataset_names": get_registered_dataset_names},
        "loss": {"get_available_loss_functions": get_available_loss_functions},
        "model_builder": {"registered_model_builders": registered_model_builders},
        "optimizer": {"get_available_optimizers": get_available_optimizers},
        "sequence_dataset": {
            "get_registered_sequence_dataset_names": get_registered_sequence_dataset_names
        },
        "sparse_layer": {"get_available_sparse_types": get_available_sparse_types},
    }


class _Sparse(nn.Module):
    def __init__(self, in_features, out_features, K, **_):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)


@pytest.mark.parametrize("registry", sorted(_registries()))
def test_registries_load_plugins_for_names_they_do_not_know(fresh_plugins, registry):
    register, lookup, unregister = _registries()[registry]
    fresh_plugins.append(_EntryPoint("components", register))
    try:
        assert lookup()
        assert [record.name for record in plugins.loaded_plugins()] == ["components"]
    finally:
        unregister()


def test_every_registry_has_its_listings_tested():
    assert sorted(_listings()) == sorted(_registries())


@pytest.mark.parametrize(
    ("registry", "listing"),
    [
        (registry, listing)
        for registry in sorted(_listings())
        for listing in _listings()[registry]
    ],
)
def test_registry_listings_include_plugin_components(fresh_plugins, registry, listing):
    """Listing a registry loads the installed plugins first, as a lookup does."""
    register, _lookup, unregister = _registries()[registry]
    fresh_plugins.append(_EntryPoint("components", register))
    try:
        assert "from_plugin" in _listings()[registry][listing]()
        assert [record.name for record in plugins.loaded_plugins()] == ["components"]
    finally:
        unregister()


class _PluginTrainer(Trainer):
    pass


def test_strategies_may_be_registered_by_module_reference(fresh_plugins):
    from dendritic_modeling.training.factory import (
        _resolve_trainer_class,
        register_trainer_strategy,
    )

    register_trainer_strategy("lazy_strategy", f"{__name__}:_PluginTrainer")
    try:
        spec = TRAINER_STRATEGY_REGISTRY["lazy_strategy"]
        assert _resolve_trainer_class(spec["class"]) is _PluginTrainer
        with pytest.raises(ValueError, match="module:Class"):
            register_trainer_strategy("bad_reference", "no_module_given")
        register_trainer_strategy("not_a_trainer", f"{__name__}:_Tiny")
        with pytest.raises(TypeError, match="not a Trainer subclass"):
            _resolve_trainer_class(TRAINER_STRATEGY_REGISTRY["not_a_trainer"]["class"])
    finally:
        TRAINER_STRATEGY_REGISTRY.pop("lazy_strategy", None)
        TRAINER_STRATEGY_REGISTRY.pop("not_a_trainer", None)


def test_a_strategy_registered_by_a_plugin_is_found_when_first_requested(
    fresh_plugins,
):
    def register():
        TRAINER_STRATEGY_REGISTRY["plugin_strategy"] = {"class": _PluginTrainer}

    fresh_plugins.append(_EntryPoint("trainers", register))
    try:
        assert "plugin_strategy" in get_available_trainer_strategies()
        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_trainer("not_a_strategy", None, {})
    finally:
        TRAINER_STRATEGY_REGISTRY.pop("plugin_strategy", None)


def test_plugin_analyzers_run_in_their_registered_phase():
    register_analyzer(
        "plugin_probe",
        config_field="plugin_probe",
        class_path="torch.nn.Identity",
        attribute_name="plugin_probe_analyzer",
        save_subdir="plugin_probe",
        constructor_style="none",
        phase="model_only",
    )
    try:
        assert "plugin_probe" in [
            item.registry_name for item in analyzer_initializers()
        ]
        assert [runner.registry_name for runner in extension_runners("model_only")] == [
            "plugin_probe"
        ]
        assert extension_runners("pre_weight") == ()
        with pytest.raises(ValueError, match="Unsupported analyzer phase"):
            register_analyzer(
                "bad_phase",
                config_field="bad_phase",
                class_path="torch.nn.Identity",
                attribute_name="bad_phase_analyzer",
                save_subdir="bad_phase",
                phase="sometime",
            )
    finally:
        unregister_analyzer("plugin_probe")
