#!/usr/bin/env python
"""Numerical regression checks for multi-population routing changes.

This is a review artifact, not package code. It exercises production builders,
parameter grouping, optimizers, standard/local/recurrent trainers, real
MNIST/CIFAR datasets, and the new qualified multi-population routing path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from dendritic_modeling.config import ParamGroupsConfig, load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.training import OptimizerConfig
from dendritic_modeling.models import RecurrentClassifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils.config_utils import prepare_trainer_config
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.scripts.training.train_experiments import (
    normalize_optimizer_config,
    normalize_param_groups_config,
)
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.training.optimizers import create_optimizer

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "review_round2" / "numerical_regression_20260605"
RUNS_DIR = ARTIFACT_DIR / "runs"
DATA_ROOT = REPO_ROOT / "outputs"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_number(value: float) -> bool:
    return math.isfinite(float(value))


def safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def nested_set(obj: Any, path: list[str], value: Any) -> None:
    current = obj
    for key in path[:-1]:
        if isinstance(current, dict):
            current = current[key]
        else:
            current = getattr(current, key)
    if isinstance(current, dict):
        current[path[-1]] = value
    else:
        setattr(current, path[-1], value)


def dataset_param_section(cfg: Any, dataset_name: str) -> dict[str, Any]:
    params = cfg.data.dataset_params
    if isinstance(params, dict):
        params.setdefault(dataset_name, {})
        return params[dataset_name]
    section = getattr(params, dataset_name)
    if isinstance(section, dict):
        return section
    payload = to_plain_dict(section)
    setattr(params, dataset_name, payload)
    return payload


def load_review_config(config_path: str, *, run_name: str, seed: int) -> Any:
    load_config.cache_clear()
    cfg = copy.deepcopy(load_config(str(REPO_ROOT / config_path)))
    cfg.experiment.seed = seed
    cfg.experiment.enable_hooks = True
    cfg.experiment.checkpointing = {"enabled": False}
    cfg.data.base_dir = str(DATA_ROOT)
    cfg.outputs.results_dir = str(RUNS_DIR)
    cfg.outputs.run_name = run_name
    cfg.wandb.use_wandb = False
    return cfg


def subset_dataset(dataset, n: int, *, seed: int):
    n = min(int(n), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)


def batch_to_device(batch, device: torch.device):
    x = batch[0].to(device)
    y = batch[1].to(device)
    seq_lengths = batch[2].to(device) if len(batch) > 2 else None
    return x, y, seq_lengths


def forward_model(model, x, seq_lengths=None):
    if seq_lengths is not None:
        return model(x, seq_lengths=seq_lengths)
    return model(x)


def evaluate_model(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | bool]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    finite_outputs = True
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y, seq_lengths = batch_to_device(batch, device)
            logits = forward_model(model, x, seq_lengths)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            finite_outputs = finite_outputs and bool(torch.isfinite(logits).all())
            loss = loss_fn(logits, y.long())
            total_loss += safe_float(loss) * x.shape[0]
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_seen += int(x.shape[0])
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "finite_outputs": finite_outputs,
        "n": total_seen,
    }


def make_optimizer(model, cfg):
    common = cfg.training.main.common
    param_groups_cfg = normalize_param_groups_config(common.param_groups)
    if param_groups_cfg is None:
        param_groups_cfg = ParamGroupsConfig()
    model_param_groups = model.get_param_groups(param_groups_cfg)
    optimizer_cfg = normalize_optimizer_config(cfg.training.main.optimizer)
    base_optimizer = create_optimizer(model_param_groups, optimizer_cfg)
    return CustomWeightDecayOptimizer(
        model=model,
        optimizer=base_optimizer,
        weight_decay=getattr(common, "weight_decay_rate", 0.0),
        weight_boosting=getattr(common, "weight_boosting", False),
    )


def gradient_probe(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    name_filters: list[str] | None = None,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    x, y, seq_lengths = batch_to_device(batch, device)
    model.train()
    model.zero_grad(set_to_none=True)
    logits = forward_model(model, x, seq_lengths)
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, y.long())
    loss.backward()

    total_sq = 0.0
    nonzero = 0
    finite_grads = True
    filtered: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite_grads = finite_grads and bool(torch.isfinite(grad).all())
        norm = float(grad.norm().cpu())
        total_sq += norm * norm
        if norm > 0:
            nonzero += 1
        for filt in name_filters or []:
            if filt in name:
                filtered[filt] = filtered.get(filt, 0.0) + norm
    model.zero_grad(set_to_none=True)
    return {
        "loss": safe_float(loss),
        "total_grad_norm": math.sqrt(total_sq),
        "nonzero_grad_param_tensors": nonzero,
        "finite_grads": finite_grads,
        "filtered_grad_norms": filtered,
    }


def build_model_and_data(cfg):
    run_save_path, train_ds, valid_ds, _test_ds, model, _encoder = setup_environment(
        model_config=cfg.model,
        training_config=cfg.training,
        data_config=cfg.data,
        wandb_config=cfg.wandb,
        outputs_config=cfg.outputs,
        experiment_config=cfg.experiment,
        is_main=True,
    )
    return Path(run_save_path), train_ds, valid_ds, model


def configure_short_training(
    cfg,
    *,
    epochs: int,
    batch_size: int,
    lr: float | None = None,
    early_stopping: bool = False,
) -> None:
    common = cfg.training.main.common
    common.epochs = int(epochs)
    common.batch_size = int(batch_size)
    common.shuffle = True
    common.early_stopping = bool(early_stopping)
    common.load_best_state_dict = False
    common.grad_clip_value = min(float(getattr(common, "grad_clip_value", 5.0)), 5.0)
    if lr is not None:
        common.param_groups.lr = lr
        common.param_groups.topk_lr = lr
        common.param_groups.decoder_lr = lr


def run_config_training_case(
    *,
    case_name: str,
    config_path: str,
    seed: int,
    epochs: int,
    batch_size: int,
    train_subset: int,
    valid_subset: int,
    strategy_override: str | None = None,
    local_rule_source: str | None = None,
    dataset_param_updates: dict[str, dict[str, Any]] | None = None,
    lr: float | None = None,
    require_train_improvement: bool = True,
    gradient_filters: list[str] | None = None,
) -> dict[str, Any]:
    set_all_seeds(seed)
    cfg = load_review_config(config_path, run_name=case_name, seed=seed)
    configure_short_training(cfg, epochs=epochs, batch_size=batch_size, lr=lr)

    if strategy_override is not None:
        cfg.training.main.strategy = strategy_override
    if local_rule_source is not None:
        src_cfg = load_review_config(
            local_rule_source, run_name="local_source", seed=seed
        )
        cfg.training.main.learning_strategy_config = copy.deepcopy(
            src_cfg.training.main.learning_strategy_config
        )
        cfg.training.main.strategy = "local_ca"
    for dataset_name, updates in (dataset_param_updates or {}).items():
        dataset_param_section(cfg, dataset_name).update(updates)

    run_save_path, train_ds, valid_ds, model = build_model_and_data(cfg)
    train_small = subset_dataset(train_ds, train_subset, seed=seed + 11)
    valid_small = subset_dataset(valid_ds, valid_subset, seed=seed + 17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    initial_train = evaluate_model(
        model, train_small, batch_size=batch_size, device=device, max_batches=6
    )
    initial_valid = evaluate_model(
        model, valid_small, batch_size=batch_size, device=device, max_batches=6
    )
    grad_before = gradient_probe(
        model,
        train_small,
        batch_size=batch_size,
        device=device,
        name_filters=gradient_filters,
    )

    optimizer = make_optimizer(model, cfg)
    trainer_config = prepare_trainer_config(
        cfg.training.main,
        cfg.experiment,
        save_path=str(run_save_path / "main_network"),
    )
    trainer_config["plot_losses"] = False
    trainer_config["suppress_prints"] = True
    trainer_config["print_every"] = 1
    trainer_config["task"] = cfg.model.task
    trainer = get_trainer(
        cfg.training.main.strategy,
        optimizer=optimizer,
        trainer_config_dict=trainer_config,
        analysis_manager=None,
    )
    t0 = time.time()
    results = trainer.train(model, train_small, valid_small)
    elapsed = time.time() - t0

    final_train = evaluate_model(
        model, train_small, batch_size=batch_size, device=device, max_batches=6
    )
    final_valid = evaluate_model(
        model, valid_small, batch_size=batch_size, device=device, max_batches=6
    )
    grad_after = gradient_probe(
        model,
        train_small,
        batch_size=batch_size,
        device=device,
        name_filters=gradient_filters,
    )

    train_improved = final_train["loss"] <= initial_train["loss"] * 1.02
    no_explosion = final_train["loss"] <= initial_train["loss"] * 1.35
    pass_checks = [
        initial_train["finite_outputs"],
        final_train["finite_outputs"],
        grad_before["finite_grads"],
        grad_after["finite_grads"],
        finite_number(grad_before["total_grad_norm"]),
        finite_number(grad_after["total_grad_norm"]),
        grad_before["total_grad_norm"] > 0.0,
        grad_after["total_grad_norm"] > 0.0,
        train_improved if require_train_improvement else no_explosion,
    ]

    return {
        "case": case_name,
        "config": config_path,
        "strategy": cfg.training.main.strategy,
        "device": str(device),
        "run_save_path": str(run_save_path),
        "nparams": sum(p.numel() for p in model.parameters()),
        "train_subset": len(train_small),
        "valid_subset": len(valid_small),
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_train": initial_train,
        "initial_valid": initial_valid,
        "final_train": final_train,
        "final_valid": final_valid,
        "train_losses": [safe_float(v) for v in results.get("train_losses", [])],
        "valid_losses": [safe_float(v) for v in results.get("valid_losses", [])],
        "gradient_before": grad_before,
        "gradient_after": grad_after,
        "train_improved_within_2pct": train_improved,
        "no_loss_explosion": no_explosion,
        "elapsed_sec": elapsed,
        "passed": bool(all(pass_checks)),
    }


def make_synthetic_sequence_dataset(
    *,
    n_samples: int,
    seq_len: int,
    input_dim: int,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, seq_len, input_dim, generator=generator)
    # Easy temporal classification target: sign of the late-window mean with a
    # small early-context interaction, so recurrent state and delayed routing
    # have a learnable signal.
    score = x[:, seq_len // 2 :, 0].mean(dim=1) + 0.25 * x[:, : seq_len // 2, 1].mean(
        dim=1
    )
    y = (score > 0).long()
    seq_lengths = torch.full((n_samples,), seq_len, dtype=torch.long)
    return TensorDataset(x, y, seq_lengths)


def build_multi_population_recurrent_model(input_dim: int, seed: int):
    set_all_seeds(seed)
    core_params = {
        "multi_population": {
            "output_mode": "last",
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e0",
                    "recurrent": True,
                    "population_defaults": {
                        "ff_excitatory_synapses": 3,
                        "ff_inhibitory_synapses": 2,
                        "rec_excitatory_synapses": 2,
                        "rec_inhibitory_synapses": 1,
                        "somatic_synapses": True,
                        "reactivate": True,
                        "reactivation_type": "param_tanh",
                        "dbl_init_method": "analytical_expectation",
                        "weight_transform": "softplus",
                        "use_shunting": True,
                        "tau_base": 5.0,
                        "tau_ratio": 1.0,
                    },
                    "populations": [
                        {
                            "name": "i0",
                            "polarity": "inhibitory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                            "population": {
                                "ff_inhibitory_synapses": 0,
                                "rec_inhibitory_synapses": 0,
                            },
                        },
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 8,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i0"},
                        {"source": "input", "target": "e0"},
                        {"source": "i0", "target": "e0"},
                        {
                            "source": "layer1.e1",
                            "target": "e0",
                            "timing": "delayed",
                        },
                    ],
                },
                {
                    "name": "layer1",
                    "readout_population": "e1",
                    "recurrent": True,
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 2,
                        "rec_excitatory_synapses": 2,
                        "rec_inhibitory_synapses": 1,
                        "somatic_synapses": True,
                        "reactivate": True,
                        "reactivation_type": "param_tanh",
                        "dbl_init_method": "analytical_expectation",
                        "weight_transform": "softplus",
                        "use_shunting": True,
                        "tau_base": 8.0,
                        "tau_ratio": 1.0,
                    },
                    "populations": [
                        {
                            "name": "i1",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                            "population": {
                                "ff_inhibitory_synapses": 0,
                                "rec_inhibitory_synapses": 0,
                            },
                        },
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 6,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "layer0.e0", "target": "i1"},
                        {"source": "layer0.e0", "target": "e1"},
                        {"source": "layer0.i0", "target": "e1"},
                        {"source": "i1", "target": "e1"},
                        {"source": "e1", "target": "e1", "timing": "delayed"},
                    ],
                },
            ],
        }
    }
    core = get_architecture("multi_population_ei", core_params, input_dim=input_dim)
    decoder = MLP(
        input_dim=core.output_dim, hidden_dims=[12], activation="relu", output_dim=2
    )
    return RecurrentClassifier(Identity(input_dim), core, decoder)


def run_direct_multi_population_recurrent_case(seed: int) -> dict[str, Any]:
    case_name = "direct_multi_population_recurrent_qualified_feedback"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_multi_population_recurrent_model(input_dim=6, seed=seed).to(device)
    train_ds = make_synthetic_sequence_dataset(
        n_samples=384, seq_len=8, input_dim=6, seed=seed + 101
    )
    valid_ds = make_synthetic_sequence_dataset(
        n_samples=128, seq_len=8, input_dim=6, seed=seed + 202
    )
    initial_train = evaluate_model(model, train_ds, batch_size=64, device=device)
    initial_valid = evaluate_model(model, valid_ds, batch_size=64, device=device)
    grad_before = gradient_probe(
        model,
        train_ds,
        batch_size=64,
        device=device,
        name_filters=["layer0", "layer1", "populations.i0", "populations.e1"],
    )
    optimizer_cfg = OptimizerConfig(name="adam", lr=0.01)
    param_cfg = ParamGroupsConfig(
        lr=0.01,
        split_params=True,
        topk_lr=0.01,
        blocklinear_lr=0.002,
        reactivation_lr=0.002,
        decoder_lr=0.01,
    )
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=create_optimizer(model.get_param_groups(param_cfg), optimizer_cfg),
        weight_decay=0.0,
    )
    trainer_config = {
        "epochs": 6,
        "batch_size": 64,
        "shuffle": True,
        "loss_function": "ce",
        "early_stopping": False,
        "load_best_state_dict": False,
        "plot_losses": False,
        "suppress_prints": True,
        "print_every": 1,
        "task": "classification",
        "grad_clip": 1.0,
        "save_path": str(RUNS_DIR / case_name),
    }
    trainer = get_trainer(
        "recurrent",
        optimizer=optimizer,
        trainer_config_dict=trainer_config,
        analysis_manager=None,
    )
    t0 = time.time()
    results = trainer.train(model, train_ds, valid_ds)
    elapsed = time.time() - t0
    final_train = evaluate_model(model, train_ds, batch_size=64, device=device)
    final_valid = evaluate_model(model, valid_ds, batch_size=64, device=device)
    grad_after = gradient_probe(
        model,
        train_ds,
        batch_size=64,
        device=device,
        name_filters=["layer0", "layer1", "populations.i0", "populations.e1"],
    )
    with torch.no_grad():
        x, _y, seq_lengths = next(iter(DataLoader(valid_ds, batch_size=16)))
        out, hidden = model.core_network(
            x.to(device),
            seq_lengths=seq_lengths.to(device),
            return_hidden=True,
        )
    route_ok = (
        out.shape == (16, model.core_network.output_dim)
        and len(hidden) == 2
        and "i0" in hidden[0].outputs
        and "e1" in hidden[1].outputs
        and torch.isfinite(out).all().item()
    )
    train_improved = final_train["loss"] < initial_train["loss"] * 0.98
    pass_checks = [
        route_ok,
        initial_train["finite_outputs"],
        final_train["finite_outputs"],
        grad_before["finite_grads"],
        grad_after["finite_grads"],
        grad_before["total_grad_norm"] > 0.0,
        grad_after["total_grad_norm"] > 0.0,
        train_improved,
    ]
    return {
        "case": case_name,
        "strategy": "recurrent",
        "device": str(device),
        "nparams": sum(p.numel() for p in model.parameters()),
        "epochs": 6,
        "batch_size": 64,
        "initial_train": initial_train,
        "initial_valid": initial_valid,
        "final_train": final_train,
        "final_valid": final_valid,
        "train_losses": [safe_float(v) for v in results.get("train_losses", [])],
        "valid_losses": [safe_float(v) for v in results.get("valid_losses", [])],
        "gradient_before": grad_before,
        "gradient_after": grad_after,
        "hidden_route_ok": bool(route_ok),
        "train_improved_by_2pct": bool(train_improved),
        "elapsed_sec": elapsed,
        "passed": bool(all(pass_checks)),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "artifact_dir": str(ARTIFACT_DIR),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_cases": len(results),
        "num_passed": sum(1 for item in results if item.get("passed")),
        "all_passed": all(item.get("passed") for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use smaller subsets")
    parser.add_argument("--json-out", default=str(ARTIFACT_DIR / "metrics.json"))
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    if args.quick:
        mnist_train, mnist_valid, cifar_train, cifar_valid = 768, 256, 768, 256
        mnist_epochs, cifar_epochs, local_epochs = 2, 2, 1
    else:
        mnist_train, mnist_valid, cifar_train, cifar_valid = 2048, 512, 2048, 512
        mnist_epochs, cifar_epochs, local_epochs = 4, 3, 2

    cases = [
        {
            "case_name": "old_standard_mnist_bp",
            "config_path": "configs/examples/standard_mnist.yaml",
            "seed": 4201,
            "epochs": mnist_epochs,
            "batch_size": 256,
            "train_subset": mnist_train,
            "valid_subset": mnist_valid,
            "require_train_improvement": True,
        },
        {
            "case_name": "multi_population_mnist_bp",
            "config_path": "configs/examples/multi_population_ei_feedforward.yaml",
            "seed": 4201,
            "epochs": mnist_epochs,
            "batch_size": 256,
            "train_subset": mnist_train,
            "valid_subset": mnist_valid,
            "require_train_improvement": True,
            "gradient_filters": ["core_network.layers.0.populations.i"],
        },
        {
            "case_name": "multi_population_independent_transfer_mnist_bp",
            "config_path": "configs/examples/multi_population_ei_independent_transfer.yaml",
            "seed": 4202,
            "epochs": mnist_epochs,
            "batch_size": 256,
            "train_subset": mnist_train,
            "valid_subset": mnist_valid,
            "require_train_improvement": True,
            "gradient_filters": ["core_network.layers.0.populations.i"],
        },
        {
            "case_name": "old_standard_cifar10_bp",
            "config_path": "configs/examples/standard_cifar10.yaml",
            "seed": 4301,
            "epochs": cifar_epochs,
            "batch_size": 256,
            "train_subset": cifar_train,
            "valid_subset": cifar_valid,
            "require_train_improvement": True,
        },
        {
            "case_name": "multi_population_two_layer_cifar10_bp",
            "config_path": "configs/examples/multi_population_ei_two_layer.yaml",
            "seed": 4301,
            "epochs": cifar_epochs,
            "batch_size": 256,
            "train_subset": cifar_train,
            "valid_subset": cifar_valid,
            "require_train_improvement": True,
            "gradient_filters": [
                "core_network.layers.0.populations.i0",
                "core_network.layers.1.populations.e1",
            ],
        },
        {
            "case_name": "old_local_learning_mnist",
            "config_path": "configs/examples/local_learning_mnist.yaml",
            "seed": 4401,
            "epochs": local_epochs,
            "batch_size": 256,
            "train_subset": min(mnist_train, 1024),
            "valid_subset": min(mnist_valid, 256),
            "require_train_improvement": False,
        },
        {
            "case_name": "multi_population_local_learning_mnist",
            "config_path": "configs/examples/multi_population_ei_feedforward.yaml",
            "local_rule_source": "configs/examples/local_learning_mnist.yaml",
            "seed": 4401,
            "epochs": local_epochs,
            "batch_size": 256,
            "train_subset": min(mnist_train, 1024),
            "valid_subset": min(mnist_valid, 256),
            "require_train_improvement": False,
            "gradient_filters": ["core_network.layers.0.populations.i"],
        },
        {
            "case_name": "old_recurrent_dms_unified_ei",
            "config_path": "configs/examples/recurrent_dms.yaml",
            "seed": 4501,
            "epochs": 2 if not args.quick else 1,
            "batch_size": 64,
            "train_subset": 512 if not args.quick else 256,
            "valid_subset": 128,
            "require_train_improvement": False,
            "dataset_param_updates": {
                "variable_delay_dms": {
                    "n_samples": 640 if not args.quick else 320,
                    "min_delay": 3,
                    "max_delay": 8,
                    "stimulus_duration": 4,
                    "probe_duration": 4,
                }
            },
        },
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"[numerical] running {case['case_name']}", flush=True)
        try:
            result = run_config_training_case(**case)
        except Exception as exc:
            result = {
                "case": case["case_name"],
                "config": case.get("config_path"),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(
            f"[numerical] {result['case']} passed={result.get('passed')} "
            f"initial={result.get('initial_train', {}).get('loss')} "
            f"final={result.get('final_train', {}).get('loss')}",
            flush=True,
        )

    print(
        "[numerical] running direct_multi_population_recurrent_qualified_feedback",
        flush=True,
    )
    try:
        direct_result = run_direct_multi_population_recurrent_case(seed=4601)
    except Exception as exc:
        direct_result = {
            "case": "direct_multi_population_recurrent_qualified_feedback",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    results.append(direct_result)
    print(
        f"[numerical] {direct_result['case']} passed={direct_result.get('passed')} "
        f"initial={direct_result.get('initial_train', {}).get('loss')} "
        f"final={direct_result.get('final_train', {}).get('loss')}",
        flush=True,
    )

    summary = summarize(results)
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
