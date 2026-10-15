"""Every YAML shipped under configs/ loads, and experiment configs build.

Each file is classified by its top-level keys and read by the loader of its
kind: experiment configs with ``load_config`` (strict), analysis profiles with
``load_analysis_config``, sweep files through the sweep manager (``extends``
composition, generation and strict loading of every generated config),
compression policies, validation-epoch selections and operator presets with
their own readers. Every experiment config then builds on CPU, runs a
forward pass and builds its optimizer and main trainer as ``train_experiments``
does, unless ``BUILT_ELSEWHERE`` names why not and which test builds it.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from dendritic_modeling.config import (
    ModelCompressionConfig,
    load_analysis_config,
    load_config,
)
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
from dendritic_modeling.scripts.analysis.select_validation_epochs import (
    _load_config as load_epoch_selection_config,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.sweeps.sweep_manager import SweepManager
from dendritic_modeling.scripts.training.train_experiments import _build_main_trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED = sorted((REPO_ROOT / "configs").rglob("*.yaml"))

# Experiment configs whose input size comes from their dataset, with the test
# that builds them on that input.
BUILT_ELSEWHERE = {
    "configs/examples/contextual_remapping_affine_mlp.yaml": (
        "input size set from the generated contextual task; built by "
        "tests/test_contextual_remapping_pipeline.py"
    ),
    "configs/examples/population_network_contextual_remapping.yaml": (
        "input size set from the generated contextual task; built by "
        "tests/test_contextual_remapping_pipeline.py"
    ),
    "configs/examples/population_network_target_levels.yaml": (
        "input size set from the generated contextual task; built by "
        "tests/test_population_target_levels.py"
    ),
}


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _kind(data: dict) -> str | None:
    keys = set(data)
    if "base_config" in keys:
        return "sweep"
    if "model" in keys:
        return "experiment"
    if keys == {"analysis"}:
        return "analysis"
    if {"dense_runtime", "sparse_topology"} & keys:
        return "compression"
    if data.get("schema_version") == "validation_epoch_selection_v1":
        return "epoch_selection"
    if {"architecture", "connectivity", "morphology"} <= keys:
        return "operator_preset"
    return None


def _load(path: Path):
    return yaml.safe_load(path.read_text())


EXPERIMENTS = [path for path in SHIPPED if _kind(_load(path)) == "experiment"]


@pytest.mark.parametrize("path", SHIPPED, ids=_relative)
def test_shipped_config_loads_with_its_loader(path):
    data = _load(path)
    kind = _kind(data)
    assert kind is not None, f"{_relative(path)} has no known config kind"
    if kind == "experiment":
        load_config(str(path))
    elif kind == "analysis":
        load_analysis_config(str(path))
    elif kind == "sweep":
        manager = SweepManager()
        config, _sources = manager._load_composed_config(str(path))
        generated = manager.generator.generate_configs(config)
        assert generated
        manager._validate_generated_configs(generated)
    elif kind == "compression":
        ModelCompressionConfig.load(str(path))
    elif kind == "epoch_selection":
        load_epoch_selection_config(path)
    else:
        ConfigurableEINetwork(config=data, input_dim=16)


def _infer_input_dim(model, cfg) -> int:
    for attr in ("output_dim", "input_dim"):
        value = getattr(getattr(model, "encoder", None), attr, None)
        if isinstance(value, int) and value > 0:
            return value
    configured = cfg.model.encoder.params.input_dim
    return configured if isinstance(configured, int) and configured > 0 else 784


@pytest.mark.parametrize("path", EXPERIMENTS, ids=_relative)
def test_experiment_config_builds_and_forwards_finite(path, tmp_path):
    name = _relative(path)
    for pattern, reason in BUILT_ELSEWHERE.items():
        if fnmatch.fnmatch(name, pattern):
            pytest.skip(reason)
    cfg = load_config(str(path))
    torch.manual_seed(0)
    model, _ = initialize_model(cfg.model)
    model.eval()
    assert sum(p.numel() for p in model.parameters()) > 0
    dim = _infer_input_dim(model, cfg)
    recurrent = bool(is_recurrent_core_config(cfg.model.core))
    x = torch.rand(2, 4, dim) if recurrent else torch.rand(2, dim)
    with torch.no_grad():
        out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(logits).all(), f"{name}: non-finite forward output"

    labels = torch.zeros(len(x), dtype=torch.long)
    setup = _build_main_trainer(
        unwrapped_model=model,
        training_config=cfg.training,
        model_config=cfg.model,
        experiment_config=cfg.experiment,
        run_save_path=str(tmp_path),
        encoded_train_ds=TensorDataset(x, labels),
        analysis_manager=None,
    )
    assert setup.strategy == str(cfg.training.main.strategy).lower()


@pytest.mark.parametrize(
    "core_type,expect_shunting",
    [
        ("rnn_dendritic_shunting", True),
        ("rnn_dendritic_additive", False),
        ("rnn_flat_shunting", True),
        ("rnn_flat_additive", False),
    ],
)
def test_recurrent_unified_ei_aliases_build(core_type, expect_shunting):
    """Covers the rnn_dendritic_*/rnn_flat_* core.type aliases. rnn_dendritic_shunting
    is the config_exp.yaml default and previously had zero test coverage."""
    import warnings

    from omegaconf import OmegaConf

    from dendritic_modeling.networks.architectures.factory import get_architecture

    branch_factors = [1] if "flat" in core_type else [2]
    params = OmegaConf.create(
        {
            "biological_neuron": False,
            "unified_ei": {
                "output_mode": "last",
                "layers": [
                    {
                        "excitatory": {
                            "n_neurons": 6,
                            "branch_factors": branch_factors,
                            "ff_excitatory_synapses": 3,
                            "rec_excitatory_synapses": 2,
                        },
                        "inhibitory": None,
                        "recurrent": True,
                        "dt": 1.0,
                    }
                ],
            },
        }
    )
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = get_architecture(core_type, params, input_dim=8)
    layer = net.layers[0].e_population.branch_layers[0]
    assert layer.use_shunting is expect_shunting
    out = net(torch.rand(2, 4, 8))
    logits = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(logits).all(), f"{core_type}: non-finite forward"
