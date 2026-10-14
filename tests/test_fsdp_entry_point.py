"""The FSDP entry point sets runs up exactly as train_experiments does."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.models import RecurrentClassifier
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.training.train_experiments import _build_cli_parser
from dendritic_modeling.scripts.training.train_experiments_fsdp import (
    create_large_model,
)

SRC = Path(__file__).resolve().parents[1] / "src"
RUN_RECORDS = {
    "config.json",
    "final_model.pt",
    "init_gate_stats.json",
    "model_resources.json",
    "nparams",
    "operator_card.json",
    "resolved_seeds.json",
    "runtime_environment.json",
    "training_summary.json",
}


def _population_core():
    return {
        "type": "population_network",
        "population_network": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 1,
                        "weight_transform": "softplus",
                        "use_shunting": True,
                        "reactivate": True,
                        "reactivation_type": "param_tanh",
                        "topk_type": "standard",
                    },
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                            "population": {"ff_inhibitory_synapses": 0},
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 6,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ]
        },
    }


def _config(tmp_path, *, mode):
    payload = {
        "experiment": {"seed": 3, "train_valid_split": 0.75},
        "data": {
            "dataset_name": "linear_separable",
            "base_dir": str(tmp_path / "data"),
            "dataset_params": {
                "linear_separable": {
                    "n_samples": 96,
                    "input_dim": 8,
                    "ne": 4,
                    "ni": 4,
                }
            },
        },
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": {}},
            "core": _population_core(),
            "decoder": {"type": "MLP", "params": {"hidden_dims": [], "output_dim": 2}},
        },
        "training": {
            "main": {
                "strategy": "standard",
                "common": {
                    "epochs": 2,
                    "batch_size": 16,
                    "num_workers": 0,
                    "synapse_decay_rate": 0.0,
                    "early_stopping": False,
                    "load_best_state_dict": False,
                },
            }
        },
        "distributed": {
            "mode": mode,
            "fsdp": {"mixed_precision": False, "sharding_strategy": "FULL_SHARD"},
        },
        "analysis": {},
        "outputs": {"run_name": "run", "results_dir": str(tmp_path / "unused")},
    }
    path = tmp_path / f"{mode}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run(module, config_path, output_dir, *, distributed):
    env = {**os.environ, "PYTHONPATH": str(SRC), "OMP_NUM_THREADS": "1"}
    command = [sys.executable]
    if distributed:
        command += [
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            f"--master_port={_free_port()}",
        ]
    command += [
        "-m",
        f"dendritic_modeling.scripts.training.{module}",
        str(config_path),
        "--output_dir",
        str(output_dir),
        "--seed",
        "5",
    ]
    completed = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    (run_dir,) = [path for path in output_dir.iterdir() if path.is_dir()]
    return run_dir


@pytest.mark.slow
@pytest.mark.multiprocess
@pytest.mark.requires_gpu
def test_both_entry_points_build_the_same_run_from_one_yaml(tmp_path):
    standard = _run(
        "train_experiments",
        _config(tmp_path, mode="none"),
        tmp_path / "standard",
        distributed=False,
    )
    fsdp = _run(
        "train_experiments_fsdp",
        _config(tmp_path, mode="fsdp"),
        tmp_path / "fsdp",
        distributed=True,
    )

    assert RUN_RECORDS <= {path.name for path in fsdp.iterdir()}
    configs = [
        json.loads((run / "config.json").read_text()) for run in (standard, fsdp)
    ]
    for config in configs:
        # Command-line overrides and the sample-derived input width are recorded.
        assert config["experiment"]["seed"] == 5
        assert config["model"]["encoder"]["params"]["input_dim"] == 8
        assert config["model"]["encoder"]["params"]["input_shape"] == [8]
    assert configs[0]["model"] == configs[1]["model"]

    states = [
        torch.load(run / "final_model.pt", map_location="cpu")
        for run in (standard, fsdp)
    ]
    assert list(states[0]) == list(states[1])
    model, _encoder = initialize_model(load_config(str(fsdp / "config.json")).model)
    model.load_state_dict(states[1])
    summary = json.loads((fsdp / "training_summary.json").read_text())
    assert summary["world_size"] == 1
    assert len(summary["train_losses"]) == 2


def test_recurrent_fsdp_models_use_the_recurrent_wrapper(tmp_path):
    payload = {
        "experiment": {"seed": 3},
        "data": {"dataset_name": "variable_delay_dms"},
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": 4}},
            "core": {
                "type": "unified_ei",
                "unified_ei": {
                    "output_mode": "last",
                    "layers": [
                        {
                            "recurrent": True,
                            "dt": 1.0,
                            "excitatory": {
                                "n_neurons": 4,
                                "branch_factors": [2],
                                "ff_excitatory_synapses": 2,
                                "rec_excitatory_synapses": 2,
                                "topk_type": "standard",
                            },
                            "inhibitory": {
                                "n_neurons": 2,
                                "branch_factors": [],
                                "ff_excitatory_synapses": 2,
                                "rec_excitatory_synapses": 2,
                                "topk_type": "standard",
                            },
                        }
                    ],
                },
            },
            "decoder": {"type": "identity", "params": {}},
        },
    }
    path = tmp_path / "recurrent.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    model = create_large_model(load_config(str(path)))
    # Classifier would add a learned per-class output scale here.
    assert type(model) is RecurrentClassifier
    assert not hasattr(model, "log_output_scale")


def test_the_fsdp_parser_takes_the_training_overrides():
    args = _build_cli_parser("FSDP").parse_args(
        ["config.yaml", "--output_dir", "out", "--run_name", "r", "--seed", "4"]
    )
    assert (args.output_dir, args.run_name, args.seed) == ("out", "r", 4)


@pytest.mark.parametrize(
    ("launcher", "mode", "world_size", "rank_env", "outcome"),
    [
        ("standard", "none", 1, False, "none"),
        ("standard", "none", 2, True, "ddp"),
        ("standard", "ddp", 2, True, "ddp"),
        ("standard", "fsdp", 1, False, "train_experiments_fsdp"),
        ("standard", "ddp", 1, False, "needs torchrun"),
        ("fsdp", "fsdp", 2, True, "fsdp"),
        ("fsdp", "none", 1, True, "fsdp"),
        ("fsdp", "ddp", 2, True, "trains with train_experiments"),
    ],
)
def test_distributed_mode_matches_the_launch(
    monkeypatch, launcher, mode, world_size, rank_env, outcome
):
    from types import SimpleNamespace

    from dendritic_modeling.scripts.training.train_experiments import (
        _resolve_distributed_mode,
    )

    if rank_env:
        monkeypatch.setenv("RANK", "0")
    else:
        monkeypatch.delenv("RANK", raising=False)
    config = SimpleNamespace(distributed=SimpleNamespace(mode=mode))
    if outcome in {"none", "ddp", "fsdp"}:
        _resolve_distributed_mode(config, launcher=launcher, world_size=world_size)
        assert config.distributed.mode == outcome
    else:
        with pytest.raises(ValueError, match=outcome):
            _resolve_distributed_mode(config, launcher=launcher, world_size=world_size)
