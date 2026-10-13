"""Configuration sections that plugins provide.

Without a registered builder a plugin section loads only when empty or
disabled; with one, the loader builds it and keeps it through a saved
``config.json``.
"""

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from dendritic_modeling.config import config_from_mapping, load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.extensions import (
    DisabledExtensionConfig,
    apply_config_migrations,
    canonical_section_path,
    register_config_migration,
    register_config_section,
    unregister_config_migration,
    unregister_config_section,
)


@pytest.mark.parametrize(
    "section",
    [
        "model.vision_replacement",
        "model.pretrained_replacement",
        "training.vision_replacement",
        "training.transformer_replacement",
    ],
)
def test_slots_without_a_plugin_fail_closed_when_enabled(tmp_path, section):
    top, key = section.split(".")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({top: {key: {"enabled": True}}}))
    with pytest.raises(ValueError, match="DendriNet plugin"):
        load_config(str(path))


def test_named_plugin_sections_without_a_plugin_are_unknown_keys():
    with pytest.raises(ValueError, match="vision_boundary_diagnostics"):
        config_from_mapping({"analysis": {"vision_boundary_diagnostics": {}}})
    with pytest.raises(ValueError, match=r"Unknown training\.main\.strategies block"):
        config_from_mapping(
            {"training": {"main": {"strategies": {"vision_distillation": {}}}}}
        )


def test_private_line_spelling_of_the_transformer_training_section_is_unknown():
    # The plugin that defines the section rewrites this spelling on load.
    with pytest.raises(ValueError, match="transformer_distillation"):
        config_from_mapping({"training": {"transformer_distillation": {}}})


def test_disabled_historical_metadata_does_not_change_the_configuration():
    historical = {
        "model": {
            "vision_replacement": {"enabled": False, "historical_metadata": {"x": 3}}
        },
        "training": {
            "transformer_replacement": {"enabled": False, "historical_metadata": [1]}
        },
    }
    assert asdict(config_from_mapping({})) == asdict(config_from_mapping(historical))
    with pytest.raises(ValueError, match="DendriNet plugin"):
        DisabledExtensionConfig(enabled=True)


def test_the_library_transformer_section_keeps_its_fields_and_rejects_others():
    config = config_from_mapping(
        {
            "model": {
                "transformer_replacement": {
                    "enabled": True,
                    "model_name": "native-example",
                    "layers": [0, 2],
                    "target_module": "mlp",
                }
            }
        }
    )
    assert config.model.transformer_replacement.layers == [0, 2]
    assert config.model.transformer_replacement.input_transform == "signed_split"
    with pytest.raises(ValueError, match="not library fields"):
        config_from_mapping(
            {
                "model": {
                    "transformer_replacement": {
                        "enabled": True,
                        "extension_setting": {"configured": True},
                    }
                }
            }
        )


def test_section_paths_are_validated():
    assert canonical_section_path("model.pretrained_replacement") == (
        "model.vision_replacement"
    )
    assert canonical_section_path("analysis.my_probe") == "analysis.my_probe"
    for bad in ("model.core", "training.main", "analysis.Bad-Name", "data.extra"):
        with pytest.raises(ValueError, match="not an extension section"):
            canonical_section_path(bad)


@dataclass
class _VisionSection:
    enabled: bool = False
    backbone: str = "alexnet"
    input_shape: list[int] = field(default_factory=lambda: [3, 224, 224])


@dataclass
class _StrategySection:
    boundary_weight: float = 1.0


@dataclass
class _ProbeSection:
    enabled: bool = False
    training: bool = False
    params: dict[str, Any] = field(default_factory=dict)


def _strict(cls):
    def build(data: Any):
        values = to_plain_dict(data)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)}")
        return cls(**values)

    return build


@pytest.fixture
def plugin_sections():
    paths = {
        "model.vision_replacement": _strict(_VisionSection),
        "training.vision_replacement": _strict(_StrategySection),
        "training.strategies.boundary_distillation": _strict(_StrategySection),
        "analysis.boundary_probe": _strict(_ProbeSection),
    }
    for path, builder in paths.items():
        register_config_section(path, builder)
    yield
    for path in paths:
        unregister_config_section(path)


@pytest.mark.usefixtures("plugin_sections")
def test_registered_sections_are_built_and_survive_a_saved_config():
    written = {
        "model": {
            "vision_replacement": {"enabled": True, "backbone": "vgg11"},
            "pretrained_replacement": {"backbone": "resnet18"},
        },
        "training": {
            "vision_replacement": {"boundary_weight": 0.5},
            "main": {
                "strategy": "boundary_distillation",
                "strategies": {"boundary_distillation": {"boundary_weight": 2.0}},
            },
        },
        "analysis": {"boundary_probe": {"enabled": True, "params": {"k": 3}}},
    }
    config = config_from_mapping(written)
    assert config.model.vision_replacement == _VisionSection(True, "vgg11")
    assert config.model.pretrained_replacement.backbone == "resnet18"
    assert config.training.vision_replacement.boundary_weight == 0.5
    strategies = config.training.main.strategies
    assert strategies.extensions == {"boundary_distillation": _StrategySection(2.0)}
    assert config.analysis.extensions["boundary_probe"].params == {"k": 3}
    assert config.analysis.section("boundary_probe").enabled is True
    assert config.analysis.any_enabled() is False  # no analyzer registered for it

    reloaded = config_from_mapping(asdict(config))
    assert asdict(reloaded) == asdict(config)


def test_configurations_built_in_code_hold_the_registered_slot_defaults(
    plugin_sections,
):
    from dendritic_modeling.config import Config
    from dendritic_modeling.config.model import TransformerReplacementConfig

    config = Config()
    assert config.model.vision_replacement == _VisionSection()
    assert config.model.pretrained_replacement == _VisionSection()
    assert config.training.vision_replacement == _StrategySection()
    assert config.training.transformer_replacement == DisabledExtensionConfig()
    assert isinstance(
        config.model.transformer_replacement, TransformerReplacementConfig
    )


@pytest.mark.usefixtures("plugin_sections")
def test_registered_sections_check_their_content_and_reject_duplicates():
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_mapping({"model": {"vision_replacement": {"typo": 1}}})
    with pytest.raises(ValueError, match="both directly and under"):
        config_from_mapping(
            {
                "analysis": {
                    "boundary_probe": {},
                    "extensions": {"boundary_probe": {}},
                }
            }
        )
    with pytest.raises(ValueError, match="already registered"):
        register_config_section("analysis.boundary_probe", _strict(_ProbeSection))


def _move_legacy_keys(data):
    """A plugin migration: legacy keys of its vision section move into it."""
    notes = []
    transfer = data.get("model", {}).get("core", {}).get("transfer", {})
    section = data.get("model", {}).get("vision_replacement")
    if isinstance(section, dict) and transfer.pop("zero_pad_output", False):
        section.setdefault("backbone", "zero_pad_alexnet")
        notes.append("model.core.transfer.zero_pad_output -> vision_replacement")
    analysis = data.get("analysis", {})
    if "boundary_probe_analysis" in analysis:
        analysis["boundary_probe"] = analysis.pop("boundary_probe_analysis")
        notes.append("analysis.boundary_probe_analysis -> analysis.boundary_probe")
    return notes


@pytest.fixture
def migration(plugin_sections):
    register_config_migration("vision", _move_legacy_keys)
    yield
    unregister_config_migration("vision")


@pytest.mark.usefixtures("migration")
def test_plugin_migrations_rewrite_historical_keys_before_checking(tmp_path, caplog):
    historical = {
        "model": {
            "core": {"transfer": {"zero_pad_output": True}},
            "vision_replacement": {"enabled": True},
        },
        "analysis": {"boundary_probe_analysis": {"enabled": True}},
    }
    original = yaml.safe_load(yaml.safe_dump(historical))
    config = config_from_mapping(historical)
    assert historical == original  # the caller's mapping is not changed
    assert config.model.vision_replacement.backbone == "zero_pad_alexnet"
    assert config.analysis.section("boundary_probe").enabled is True
    assert "vision: model.core.transfer.zero_pad_output" in caplog.text

    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump({"boundary_probe_analysis": {"enabled": True}}))
    from dendritic_modeling.config import load_analysis_config

    assert load_analysis_config(str(path)).section("boundary_probe").enabled

    from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping

    upgraded, notes, _ = upgrade_mapping(original)
    assert upgraded["model"]["vision_replacement"]["backbone"] == "zero_pad_alexnet"
    assert "boundary_probe" in upgraded["analysis"]
    assert any(note.startswith("vision: ") for note in notes)


def test_migrations_are_named_and_optional():
    data = {"model": {"core": {"type": "einet"}}}
    assert apply_config_migrations(data) == (data, [])
    with pytest.raises(TypeError, match="callable"):
        register_config_migration("broken", "not callable")
    register_config_migration("noop", lambda data: [])
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_config_migration("noop", lambda data: [])
        assert apply_config_migrations(data) == (data, [])
    finally:
        unregister_config_migration("noop")


@pytest.mark.requires_git_checkout
def test_tracked_recipes_exclude_extension_sections_and_resolve_parents():
    """Shipped configurations use only library sections."""
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(
        ["git", "ls-files", "--", "configs"], cwd=root, text=True
    ).splitlines()
    forbidden = {
        "vision_replacement",
        "pretrained_replacement",
        "vision_boundary_diagnostics",
    }

    def inspect(value, path, trail=()):
        if isinstance(value, dict):
            for key, item in value.items():
                parts = trail + tuple(str(key).split("."))
                assert not forbidden.intersection(parts), (path, parts)
                assert parts[-2:] != ("training", "transformer_replacement"), (
                    path,
                    parts,
                )
                assert not (key == "strategy" and item == "vision_distillation"), path
                inspect(item, path, parts)
        elif isinstance(value, list):
            for item in value:
                inspect(item, path, trail)

    for name in paths:
        path = root / name
        if path.suffix not in {".yaml", ".yml"}:
            continue
        value = yaml.safe_load(path.read_text())
        inspect(value, name)
        if isinstance(value, dict) and "extends" in value:
            parent = value["extends"]
            assert isinstance(parent, str) and parent, name
            assert (path.parent / parent).is_file(), (name, parent)
