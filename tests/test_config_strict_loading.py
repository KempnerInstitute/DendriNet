"""load_config rejects keys it does not read, at every level of the config."""

from __future__ import annotations

import copy
import logging

import pytest
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.config.analysis import (
    InformationEstimatorKraskovParams,
    InformationEstimatorParams,
)
from dendritic_modeling.config.keys import UnknownConfigKeyError
from dendritic_modeling.config.regularization import (
    PruningConfig,
    RegularizationConfig,
    SelectiveConfig,
)

BASE = yaml.safe_load("""
experiment: {seed: 3}
data: {dataset_name: mnist, processing: {flatten: true}}
model:
  task: classification
  encoder: {type: identity, params: {input_dim: 6}}
  core:
    type: einet
    architecture: {excitatory_layer_sizes: [3], inhibitory_layer_sizes: []}
    connectivity:
      ee_synapses_per_branch_per_layer: [2]
      ie_synapses_per_branch_per_layer: [0]
    sparsity: {type: standard}
  decoder: {type: identity, params: {hidden_dims: []}}
training:
  main:
    strategy: standard
    common: {epochs: 1, synapse_decay_rate: 0.0}
    optimizer: {name: adam, lr: 0.001}
analysis:
  information_analysis:
    enabled: false
    params: {estimator: {method: kraskov}}
outputs: {results_dir: results}
""")


def _load(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return load_config(str(path))


def _with(path, value, base=BASE):
    payload = copy.deepcopy(base)
    node = payload
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    return payload


def test_base_config_loads(tmp_path):
    config = _load(tmp_path, BASE)
    assert config.data.dataset_name == "mnist"
    assert config.metadata == {}


@pytest.mark.parametrize(
    "path, section, hint",
    [
        (("trainig",), "top-level", "training"),
        (("data", "datset_name"), "data", "dataset_name"),
        (("data", "processing", "flaten"), "data.processing", "flatten"),
        (("model", "decodr"), "model", "decoder"),
        (("model", "encoder", "parms"), "model.encoder", "params"),
        (("model", "encoder", "params", "input_dimm"), "model.encoder.params", None),
        (("model", "decoder", "params", "hiden_dims"), "model.decoder.params", None),
        (("model", "core", "sparsty"), "model.core", "sparsity"),
        (("model", "core", "sparsity", "temprature"), "model.core.sparsity", None),
        (("model", "core", "morphology", "use_shunt"), "model.core.morphology", None),
        (
            ("model", "core", "connectivity", "structured", "enabeld"),
            "model.core.connectivity.structured",
            "enabled",
        ),
        (("model", "core", "reactivaton"), "model.core", "reactivation"),
        (("training", "mian"), "training", "main"),
        (("training", "main", "optimizr"), "training.main", "optimizer"),
        (("training", "main", "optimizer", "lrr"), "training.main.optimizer", "lr"),
        (
            ("training", "main", "regularization", "l1_weigth"),
            "training.main.regularization",
            "l1_weight",
        ),
        (
            ("training", "main", "pruning", "selective", "layer"),
            "training.main.pruning.selective",
            "layers",
        ),
        (
            ("training", "main", "learning_strategy_config", "rule_varient"),
            "training.main.learning_strategy_config",
            "rule_variant",
        ),
        (
            (
                "training",
                "main",
                "learning_strategy_config",
                "three_factor",
                "dynamics_mod",
            ),
            "training.main.learning_strategy_config.three_factor",
            "dynamics_mode",
        ),
        (("training", "main", "comon"), "training.main", "common"),
        (("analysis", "informaton"), "analysis", "information"),
        (
            ("analysis", "information_analysis", "enabeld"),
            "analysis.information",
            "enabled",
        ),
        (
            ("analysis", "information_analysis", "params", "estimator", "methd"),
            "analysis.information.params.estimator",
            "method",
        ),
        (
            ("analysis", "runtime", "training", "batchsize"),
            "analysis.runtime.training",
            "batch_size",
        ),
        (("outputs", "run_nam"), "outputs", "run_name"),
    ],
)
def test_misspelled_key_raises_with_its_section(tmp_path, path, section, hint):
    with pytest.raises(UnknownConfigKeyError) as caught:
        _load(tmp_path, _with(path, 1))
    message = str(caught.value)
    if section == "top-level":
        assert message.startswith("Unknown top-level section(s): trainig")
    else:
        assert f"Unknown {section} option(s): {path[-1]}" in message
    if hint is not None:
        assert f"did you mean {hint!r}" in message


def test_non_mapping_section_raises(tmp_path):
    with pytest.raises(ValueError, match=r"model\.core\.dynamics must be a mapping"):
        _load(tmp_path, _with(("model", "core", "dynamics"), 5))


def test_metadata_and_underscore_keys_are_recorded(tmp_path):
    payload = copy.deepcopy(BASE)
    payload["metadata"] = {"paper_contract": {"study_id": "s1"}}
    payload["_sweep_config_id"] = 7
    config = _load(tmp_path, payload)
    assert config.metadata == {
        "paper_contract": {"study_id": "s1"},
        "_sweep_config_id": 7,
    }


def test_conflicting_underscore_key_and_metadata_raise(tmp_path):
    payload = copy.deepcopy(BASE)
    payload["metadata"] = {"_sweep_variant": "a"}
    payload["_sweep_variant"] = "b"
    with pytest.raises(ValueError, match=r"conflicts with metadata\._sweep_variant"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("wandb",), {"use_wandb": False, "project": "p"}, "Ignoring wandb"),
        (
            ("training", "main", "common", "checkpointing"),
            {"enabled": True},
            "Ignoring training.main.common.checkpointing: it never configured",
        ),
        (
            ("model", "core", "reactivation", "match_additive_init_to_shunting"),
            False,
            "Ignoring model.core.reactivation.match_additive_init_to_shunting",
        ),
        (
            ("analysis", "ablation_analysis", "params", "accuracy"),
            True,
            "was replaced by params.selection and params.methods",
        ),
        (("experiment", "profiling_output_dir"), "/tmp/p", "it was never read"),
        (
            ("model", "encoder", "load_save_root"),
            "./trained",
            "removed encoder pretraining script",
        ),
        (("experiment", "enable_hooks"), False, "never disabled anything"),
        (
            ("model", "encoder", "params", "encoder_kernel_sizes"),
            [5, 5],
            "derive kernel sizes and strides",
        ),
        (
            ("training", "main", "pruning", "pruning_frequency"),
            5,
            "pruning runs once after training",
        ),
        (
            ("training", "main", "learning_strategy_config", "rls_forgetting"),
            0.9,
            "no RLS estimator",
        ),
        (
            (
                "training",
                "main",
                "learning_strategy_config",
                "five_factor",
                "rls_forgetting",
            ),
            0.9,
            "no RLS estimator",
        ),
        (
            (
                "training",
                "main",
                "learning_strategy_config",
                "morphology_aware",
                "morphology_centrality_metric",
            ),
            "closeness",
            "inverse branch depth",
        ),
    ],
)
def test_retired_keys_warn_and_are_dropped(tmp_path, caplog, path, value, message):
    with caplog.at_level(logging.WARNING):
        _load(tmp_path, _with(path, value))
    assert message in caplog.text


@pytest.mark.parametrize(
    "path, value, message",
    [
        (
            ("model", "core", "reactivation", "match_additive_init_to_shunting"),
            True,
            "replaced by model.core.reactivation.init_policy",
        ),
        (("model", "core_network"), {"type": "einet"}, "before September 2025"),
        (
            ("analysis", "compositional_ood_analysis"),
            {"enabled": True},
            "was removed on 2026-08-29",
        ),
        (("experiment", "enable_profiling"), True, "never implemented"),
        (("wandb",), {"use_wandb": True}, "Weights & Biases"),
        (("training", "main", "pruning", "dynamic_pruning"), True, "never implemented"),
    ],
)
def test_retired_keys_that_cannot_be_reproduced_raise(tmp_path, path, value, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, _with(path, value))


@pytest.mark.parametrize(
    "first, second",
    [
        (("model", "core", "ei_unified"), ("model", "core", "unified_ei")),
        (("model", "core", "spiking"), ("model", "core", "dynamics")),
        (
            ("model", "core", "connectivity", "structured_connectivity"),
            ("model", "core", "connectivity", "structured"),
        ),
        (
            ("model", "core", "sparsity", "annealed_topk"),
            ("model", "core", "sparsity", "dense_to_sparse"),
        ),
    ],
)
def test_alias_and_canonical_spelling_together_raise(tmp_path, first, second):
    payload = _with(second, {}, _with(first, {}))
    with pytest.raises(ValueError, match="deprecated spelling"):
        _load(tmp_path, payload)


def test_analysis_registry_and_field_names_together_raise(tmp_path):
    payload = copy.deepcopy(BASE)
    payload["analysis"]["information"] = {"enabled": False}
    with pytest.raises(ValueError, match="configure the same analysis"):
        _load(tmp_path, payload)


def test_learning_rate_key_under_common_moves_to_param_groups(tmp_path, caplog):
    payload = _with(("training", "main", "common", "lr"), 0.05)
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, payload)
    assert config.training.main.common.param_groups.lr == 0.05
    assert "training.main.common.lr is deprecated" in caplog.text

    conflicting = _with(
        ("training", "main", "common", "param_groups"), {"lr": 0.01}, payload
    )
    with pytest.raises(ValueError, match="conflicts with"):
        _load(tmp_path, conflicting)


def test_nested_sections_are_typed(tmp_path):
    payload = _with(
        ("analysis", "information_analysis", "params", "estimator"),
        {"method": "kraskov", "kraskov": {"n_neighbors": 5}},
    )
    payload = _with(
        ("training", "main", "regularization"),
        {"l1_weight": 0.1, "selective": {"layers": [0]}},
        payload,
    )
    payload = _with(("training", "main", "pruning"), {"enabled": False}, payload)
    config = _load(tmp_path, payload)

    estimator = config.analysis.information.params.estimator
    assert isinstance(estimator, InformationEstimatorParams)
    assert isinstance(estimator.kraskov, InformationEstimatorKraskovParams)
    assert estimator.kraskov.n_neighbors == 5
    regularization = config.training.main.regularization
    assert isinstance(regularization, RegularizationConfig)
    assert isinstance(regularization.selective, SelectiveConfig)
    assert regularization.selective.layers == [0]
    assert isinstance(config.training.main.pruning, PruningConfig)


def test_extends_chain_is_composed(tmp_path):
    parent = tmp_path / "parent.yaml"
    parent.write_text(yaml.safe_dump(BASE, sort_keys=False))
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = child_dir / "child.yaml"
    child.write_text(
        yaml.safe_dump(
            {"extends": "../parent.yaml", "experiment": {"seed": 11}},
            sort_keys=False,
        )
    )
    config = load_config(str(child))
    assert config.experiment.seed == 11
    assert config.data.dataset_name == "mnist"


def test_extends_cycle_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="extends cycle"):
        load_config(str(tmp_path / "a.yaml"))


def test_sweep_file_base_config_is_checked(tmp_path):
    payload = {
        "slurm_config": {"time": "01:00:00"},
        "base_config": _with(("data", "datset_name"), "mnist"),
    }
    with pytest.raises(UnknownConfigKeyError, match="did you mean 'dataset_name'"):
        _load(tmp_path, payload)


def test_cache_clear_is_a_deprecated_no_op(tmp_path):
    with pytest.deprecated_call():
        load_config.cache_clear()


@pytest.mark.parametrize(
    "path, value, message",
    [
        (
            ("model", "core", "sparsity", "deepst"),
            {"sigmma": 0.1},
            "did you mean 'sigma'",
        ),
        (
            ("model", "core", "sparsity", "dense_to_sparse"),
            {"end_stp": 10},
            "did you mean 'end_step'",
        ),
        (
            ("training", "main", "common", "impact_rewiring"),
            {"fraction": 0.1},
            "did you mean 'segment_fraction'",
        ),
        (
            ("training", "main", "common", "impact_rewiring"),
            {"seed": 3},
            "is derived",
        ),
    ],
)
def test_synapse_and_rewiring_payload_typos_raise(tmp_path, path, value, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, _with(path, value))


@pytest.mark.parametrize(
    "strategy, block, warns",
    [
        ("standard", {"rule_variant": "5f"}, True),
        ("soma_dfa", {"rule_variant": "5f"}, True),
        ("local_ca", {"rule_variant": "5f"}, False),
        ("standard", {}, False),
    ],
)
def test_local_rule_block_ignored_by_strategy_warns(
    tmp_path, caplog, strategy, block, warns
):
    payload = _with(("training", "main", "strategy"), strategy)
    payload["training"]["main"]["learning_strategy_config"] = block
    with caplog.at_level(logging.WARNING):
        _load(tmp_path, payload)
    assert ("has no effect with strategy" in caplog.text) is warns


def test_overrides_are_applied_before_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(BASE, sort_keys=False))
    config = load_config(
        str(path),
        overrides={
            "training.main.common.epochs": 7,
            "outputs.run_name": "cli",
            "training.main.learning_strategy_config.error_broadcast_mode": (
                "per_soma_shared"
            ),
        },
    )
    assert config.training.main.common.epochs == 7
    assert config.outputs.run_name == "cli"
    assert config.training.main.learning_strategy_config.error_broadcast_mode == (
        "per_soma_shared"
    )
    with pytest.raises(UnknownConfigKeyError, match="epochz"):
        load_config(str(path), overrides={"training.main.common.epochz": 7})
    with pytest.raises(ValueError, match="is not a mapping"):
        load_config(str(path), overrides={"training.main.strategy.name": "x"})


def test_overrides_without_a_file_start_from_defaults():
    config = load_config(overrides={"experiment.seed": 11})
    assert config.experiment.seed == 11


def test_analysis_options_the_analyzers_ignore_are_rejected(tmp_path, caplog):
    snr = ("analysis", "compartment_snr_analysis", "params")
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, _with((*snr, "global_analysis"), False))
    assert "global_aggregation controls the global summary" in caplog.text
    assert config.analysis.compartment_snr.params.input_capture_mode == ("branch")
    alias = _load(tmp_path, _with((*snr, "input_capture_mode"), "network_inputs"))
    assert alias.analysis.compartment_snr.params.input_capture_mode == ("network")
    with pytest.raises(ValueError, match="input_capture_mode"):
        _load(tmp_path, _with((*snr, "input_capture_mode"), "branches"))

    representation = ("analysis", "representation_accessibility_analysis", "params")
    with pytest.raises(ValueError, match="draws must be 1"):
        _load(
            tmp_path,
            _with((*representation, "probes"), [{"type": "ridge", "draws": 3}]),
        )
    with pytest.raises(ValueError, match="name no declared projection"):
        _load(tmp_path, _with((*representation, "information"), {"enabled": True}))


def test_loading_a_config_does_not_import_torch():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = (
        "import logging, sys\n"
        "logging.disable(logging.WARNING)\n"
        "from dendritic_modeling.config import load_config\n"
        "from dendritic_modeling.config.training import CommonTrainerConfig\n"
        f"load_config({str(root / 'configs' / 'config_exp.yaml')!r})\n"
        "CommonTrainerConfig(impact_rewiring={'frequency': 10})\n"
        "heavy = [m for m in ('torch', 'triton', 'scipy') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr[-2000:]
