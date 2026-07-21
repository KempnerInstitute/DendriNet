"""Generate OLMo transformer-replacement benchmark configs.

The Slurm jobs call these package-owned helpers instead of ignored root
scripts, keeping benchmark config generation reproducible from tracked code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

OLMO_VARIANTS = {
    "1b": {
        "model_name": "allenai/OLMo-2-0425-1B",
        "num_layers": 16,
        "width": 1024,
        "output_chunk_size": 2048,
        "layerwise_batch_size": 4,
        "layerwise_lr": 0.0007,
        "seed_base": 610,
    },
    "7b": {
        "model_name": "allenai/OLMo-2-1124-7B",
        "num_layers": 32,
        "width": 2048,
        "output_chunk_size": 4096,
        "layerwise_batch_size": 2,
        "layerwise_lr": 0.0005,
        "seed_base": 611,
    },
    "13b": {
        "model_name": "allenai/OLMo-2-1124-13B",
        "num_layers": 40,
        "width": 2560,
        "output_chunk_size": 4096,
        "layerwise_batch_size": 1,
        "layerwise_lr": 0.0003,
        "seed_base": 612,
    },
}


def _variant_info(variant: str) -> dict[str, Any]:
    key = str(variant).lower()
    if key not in OLMO_VARIANTS:
        valid = ", ".join(sorted(OLMO_VARIANTS))
        raise ValueError(f"Unsupported OLMo variant {variant!r}; valid: {valid}")
    return dict(OLMO_VARIANTS[key])


def _layers(num_layers: int) -> list[int]:
    return list(range(int(num_layers)))


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), path)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _base_dendritic_ffn_core(
    *,
    branch_factors: list[int],
    seed: int,
    width: int = 1024,
    output_chunk_size: int = 2048,
) -> dict[str, Any]:
    return {
        "type": "dendritic_ffn",
        "population_network": {
            "layers": [
                {
                    "name": "transformer_mlp",
                    "readout_population": "ffn",
                    "population_defaults": {
                        "ff_excitatory_synapses": 64,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 0,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "use_shunting": False,
                        "reactivate": True,
                        "reactivation_type": "param_tanh",
                        "reactivation_init_policy": "occupancy_quantile",
                        "weight_transform": "softplus",
                        "topk_type": "indexed_rewire",
                        "topk_init_method": "xavier_normal",
                        "indexed_seed": int(seed),
                        "indexed_rewire_frequency": 100,
                        "indexed_rewire_quantile": 0.05,
                        "indexed_rewire_until_step": None,
                        "indexed_output_chunk_size": int(output_chunk_size),
                        "efficient_blocklinear": True,
                    },
                    "populations": [
                        {
                            "name": "ffn",
                            "polarity": "excitatory",
                            "n_neurons": int(width),
                            "branch_factors": list(branch_factors),
                        }
                    ],
                    "connections": [
                        {
                            "source": "input",
                            "target": "ffn",
                        }
                    ],
                }
            ],
        },
    }


def _transformer_model_block(
    *,
    model_name: str,
    layers: list[int],
    replacement_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "model_name": model_name,
        "layers": list(layers),
        "target_module": "mlp",
        "layers_attr": "model.layers",
        "input_transform": "signed_split",
        "preserve_device_dtype": True,
        "model_kwargs": {
            "trust_remote_code": True,
            "torch_dtype": "auto",
        },
        "replacement_kwargs": {
            "output_init_std": 0.02,
            **(replacement_kwargs or {}),
        },
    }


def _dolmino_text_block(
    *, model_name: str, max_documents: int, max_tokens: int
) -> dict:
    return {
        "dataset_name": "allenai/dolmino-mix-1124",
        "dataset_config": "default",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "max_documents": int(max_documents),
        "tokenizer_name": model_name,
        "max_tokens": int(max_tokens),
    }


def _layerwise_config(
    *,
    model_name: str,
    layers: list[int],
    seed: int,
    save_dir: str,
    max_steps: int,
    width: int,
    output_chunk_size: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    return {
        "experiment": {"seed": int(seed)},
        "model": {
            "task": "language_modeling",
            "transformer_replacement": _transformer_model_block(
                model_name=model_name,
                layers=layers,
            ),
            "core": _base_dendritic_ffn_core(
                branch_factors=[2, 2],
                seed=seed,
                width=width,
                output_chunk_size=output_chunk_size,
            ),
        },
        "training": {
            "main": {
                "optimizer": {
                    "name": "adamw",
                    "lr": float(lr),
                    "weight_decay": 0.0001,
                }
            },
            "transformer_replacement": {
                "enabled": True,
                "mode": "layerwise_distillation",
                "teacher_source": "hf_text",
                "train_target": "replacement_only",
                "layers": list(layers),
                "device": "auto",
                "dtype": "float32",
                "use_amp": True,
                "sequence_length": 128,
                "train_samples": 8192,
                "valid_samples": 64,
                "batch_size": int(batch_size),
                "max_steps": int(max_steps),
                "eval_every": 100,
                "log_every": 25,
                "loss": "mse",
                "save_dir": save_dir,
                "save_replacements": True,
                "text": _dolmino_text_block(
                    model_name=model_name,
                    max_documents=1200,
                    max_tokens=200000,
                ),
            },
        },
    }


def generate_all_layer_configs(
    *,
    variant: str,
    config_root: str | Path,
    results_root: str | Path,
    max_steps: int = 800,
    num_layers: int | None = None,
) -> dict[str, Any]:
    info = _variant_info(variant)
    variant = str(variant).lower()
    model_name = str(info["model_name"])
    num_layers = int(num_layers or info["num_layers"])
    all_layers = _layers(num_layers)

    config_dir = Path(config_root) / f"olmo2_{variant}_dolmino_all_layers"
    result_dir = Path(results_root) / f"olmo2_{variant}_dolmino_all_layers"
    config_dir.mkdir(parents=True, exist_ok=True)

    layer_configs = []
    layer_runs = []
    for layer in all_layers:
        seed = int(info["seed_base"]) + layer
        path = config_dir / f"olmo2_{variant}_dolmino_layer{layer:02d}.yaml"
        _save_yaml(
            path,
            _layerwise_config(
                model_name=model_name,
                layers=[layer],
                seed=seed,
                save_dir=str(result_dir / f"layer_{layer:02d}"),
                max_steps=max_steps,
                width=int(info["width"]),
                output_chunk_size=int(info["output_chunk_size"]),
                batch_size=int(info["layerwise_batch_size"]),
                lr=float(info["layerwise_lr"]),
            ),
        )
        layer_configs.append(str(path))
        layer_runs.append(
            {
                "layer": layer,
                "config": str(path),
                "save_dir": str(result_dir / f"layer_{layer:02d}"),
                "checkpoint": str(
                    result_dir / f"layer_{layer:02d}" / f"layer_{layer}_replacement.pt"
                ),
            }
        )

    all_layers_config = config_dir / f"olmo2_{variant}_dolmino_all_layers.yaml"
    _save_yaml(
        all_layers_config,
        _layerwise_config(
            model_name=model_name,
            layers=all_layers,
            seed=int(info["seed_base"]) + 1000,
            save_dir=str(result_dir / "all_layers_patched"),
            max_steps=0,
            width=int(info["width"]),
            output_chunk_size=int(info["output_chunk_size"]),
            batch_size=int(info["layerwise_batch_size"]),
            lr=float(info["layerwise_lr"]),
        ),
    )

    manifest = {
        "variant": variant,
        "model_name": model_name,
        "num_layers": num_layers,
        "config_dir": str(config_dir),
        "results_root": str(result_dir),
        "layer_configs": layer_configs,
        "layer_runs": layer_runs,
        "assembled_checkpoint_dir": str(result_dir / "all_layer_checkpoints"),
        "all_layers_config": str(all_layers_config),
    }
    _save_json(config_dir / "manifest.json", manifest)
    return manifest


def _joint_config(
    *,
    model_name: str,
    layers: list[int],
    seed: int,
    branch_factors: list[int],
    lr: float,
    save_dir: str,
    warm_start_dir: str,
    width: int,
    output_chunk_size: int,
    max_steps: int = 800,
) -> dict[str, Any]:
    return {
        "experiment": {"seed": int(seed)},
        "model": {
            "task": "language_modeling",
            "transformer_replacement": _transformer_model_block(
                model_name=model_name,
                layers=layers,
            ),
            "core": _base_dendritic_ffn_core(
                branch_factors=branch_factors,
                seed=seed,
                width=width,
                output_chunk_size=output_chunk_size,
            ),
        },
        "training": {
            "main": {
                "optimizer": {
                    "name": "adamw",
                    "lr": float(lr),
                    "weight_decay": 0.0001,
                }
            },
            "transformer_replacement": {
                "enabled": True,
                "mode": "joint_lm_distillation",
                "teacher_source": "hf_text",
                "train_target": "replacement_only",
                "layers": list(layers),
                "device": "auto",
                "dtype": "float32",
                "use_amp": True,
                "gradient_checkpointing": False,
                "sequence_length": 128,
                "train_samples": 8192,
                "valid_samples": 16,
                "batch_size": 1,
                "max_steps": int(max_steps),
                "eval_every": 50,
                "log_every": 10,
                "lm_loss_weight": 1.0,
                "kl_loss_weight": 0.2,
                "kl_temperature": 2.0,
                "hidden_loss_weight": 0.05,
                "hidden_loss_layers": _hidden_supervision_layers(layers),
                "max_grad_norm": 1.0,
                "warm_start_dir": warm_start_dir,
                "loss": "mse",
                "save_dir": save_dir,
                "save_replacements": True,
                "text": _dolmino_text_block(
                    model_name=model_name,
                    max_documents=1600,
                    max_tokens=300000,
                ),
            },
        },
    }


def _hidden_supervision_layers(layers: list[int]) -> list[int]:
    """Choose roughly quarter-, half-, and full-depth supervision sites."""
    if not layers:
        return []
    ordered = sorted({int(layer) for layer in layers})
    indices = {
        max(0, len(ordered) // 4 - 1),
        max(0, len(ordered) // 2 - 1),
        len(ordered) - 1,
    }
    return [ordered[index] for index in sorted(indices)]


def generate_joint_configs(
    *,
    variant: str,
    config_root: str | Path,
    results_root: str | Path,
    max_steps: int = 800,
) -> dict[str, Any]:
    info = _variant_info(variant)
    variant = str(variant).lower()
    model_name = str(info["model_name"])
    layers = _layers(int(info["num_layers"]))

    config_dir = Path(config_root) / f"olmo2_{variant}_joint_all_layers"
    result_dir = Path(results_root) / f"olmo2_{variant}_joint_all_layers"
    config_dir.mkdir(parents=True, exist_ok=True)
    warm_start_root = (
        Path(results_root)
        / f"olmo2_{variant}_dolmino_all_layers"
        / "all_layer_checkpoints"
    )

    specs = [
        ("joint_b22_warm", [2, 2], 2610, 0.0001, str(warm_start_root)),
        ("joint_b22_scratch", [2, 2], 2611, 0.0002, ""),
        ("joint_b222_scratch", [2, 2, 2], 2612, 0.00015, ""),
    ]

    configs = []
    for name, branch_factors, seed, lr, warm_start_dir in specs:
        path = config_dir / f"olmo2_{variant}_{name}.yaml"
        save_dir = result_dir / name
        _save_yaml(
            path,
            _joint_config(
                model_name=model_name,
                layers=layers,
                seed=seed,
                branch_factors=branch_factors,
                lr=lr,
                save_dir=str(save_dir),
                warm_start_dir=warm_start_dir,
                width=int(info["width"]),
                output_chunk_size=int(info["output_chunk_size"]),
                max_steps=max_steps,
            ),
        )
        configs.append(
            {
                "name": name,
                "config": str(path),
                "save_dir": str(save_dir),
                "branch_factors": branch_factors,
                "warm_start_dir": warm_start_dir,
            }
        )

    manifest = {
        "variant": variant,
        "model_name": model_name,
        "num_layers": len(layers),
        "config_dir": str(config_dir),
        "results_root": str(result_dir),
        "configs": configs,
    }
    _save_json(config_dir / "manifest.json", manifest)
    return manifest


def _ei_stack_config(
    *,
    model_name: str,
    layers: list[int],
    seed: int,
    save_dir: str,
    max_steps: int = 800,
) -> dict[str, Any]:
    return {
        "experiment": {"seed": int(seed)},
        "model": {
            "task": "language_modeling",
            "transformer_replacement": _transformer_model_block(
                model_name=model_name,
                layers=layers,
                replacement_kwargs={
                    "kind": "ei_stack",
                    "output_scale": 1.0,
                },
            ),
            "core": {
                "type": "population_network",
                "population_network": {
                    "layers": [
                        {
                            "name": "transformer_mlp",
                            "readout_population": "e",
                            "population_defaults": {
                                "ff_excitatory_synapses": 32,
                                "ff_inhibitory_synapses": 16,
                                "rec_excitatory_synapses": 0,
                                "rec_inhibitory_synapses": 0,
                                "somatic_synapses": True,
                                "use_shunting": True,
                                "reactivate": True,
                                "reactivation_type": "param_tanh",
                                "reactivation_init_policy": "occupancy_quantile",
                                "weight_transform": "softplus",
                                "topk_type": "indexed_rewire",
                                "topk_init_method": "xavier_normal",
                                "indexed_seed": int(seed),
                                "indexed_rewire_frequency": 100,
                                "indexed_rewire_quantile": 0.05,
                                "indexed_rewire_until_step": None,
                                "indexed_output_chunk_size": 2048,
                                "efficient_blocklinear": True,
                            },
                            "populations": [
                                {
                                    "name": "i",
                                    "polarity": "inhibitory",
                                    "n_neurons": 50,
                                    "branch_factors": [2],
                                    "population": {
                                        "ff_excitatory_synapses": 16,
                                        "ff_inhibitory_synapses": 8,
                                    },
                                },
                                {
                                    "name": "e",
                                    "polarity": "excitatory",
                                    "n_neurons": 200,
                                    "branch_factors": [2, 2],
                                },
                            ],
                            "connections": [
                                {
                                    "source": "input",
                                    "target": "i",
                                },
                                {
                                    "source": "input",
                                    "target": "e",
                                },
                                {
                                    "source": "i",
                                    "target": "e",
                                },
                            ],
                        }
                    ],
                },
            },
        },
        "training": {
            "main": {
                "optimizer": {
                    "name": "adamw",
                    "lr": 0.00015,
                    "weight_decay": 0.0001,
                }
            },
            "transformer_replacement": {
                "enabled": True,
                "mode": "joint_lm_distillation",
                "teacher_source": "hf_text",
                "train_target": "replacement_only",
                "layers": list(layers),
                "device": "auto",
                "dtype": "float32",
                "use_amp": True,
                "gradient_checkpointing": False,
                "sequence_length": 128,
                "train_samples": 8192,
                "valid_samples": 16,
                "batch_size": 1,
                "max_steps": int(max_steps),
                "eval_every": 50,
                "log_every": 10,
                "lm_loss_weight": 1.0,
                "kl_loss_weight": 0.2,
                "kl_temperature": 2.0,
                "hidden_loss_weight": 0.05,
                "hidden_loss_layers": _hidden_supervision_layers(layers),
                "max_grad_norm": 1.0,
                "warm_start_dir": "",
                "loss": "mse",
                "save_dir": save_dir,
                "save_replacements": True,
                "text": _dolmino_text_block(
                    model_name=model_name,
                    max_documents=1600,
                    max_tokens=300000,
                ),
            },
        },
    }


def generate_ei_stack_configs(
    *,
    config_root: str | Path,
    results_root: str | Path,
    variant: str = "1b",
    max_steps: int = 800,
) -> dict[str, Any]:
    info = _variant_info(variant)
    variant = str(variant).lower()
    model_name = str(info["model_name"])
    layers = _layers(int(info["num_layers"]))

    config_dir = Path(config_root) / f"olmo2_{variant}_ei_stack_all_mlp"
    result_dir = Path(results_root) / f"olmo2_{variant}_ei_stack_all_mlp"
    config_dir.mkdir(parents=True, exist_ok=True)

    name = "e200_i50_b22"
    path = config_dir / f"olmo2_{variant}_ei_stack_e200_i50.yaml"
    save_dir = result_dir / name
    _save_yaml(
        path,
        _ei_stack_config(
            model_name=model_name,
            layers=layers,
            seed=3610,
            save_dir=str(save_dir),
            max_steps=max_steps,
        ),
    )

    manifest = {
        "model_name": model_name,
        "num_layers": len(layers),
        "config_dir": str(config_dir),
        "results_root": str(result_dir),
        "configs": [
            {
                "name": name,
                "config": str(path),
                "save_dir": str(save_dir),
                "excitatory_cells": 200,
                "inhibitory_cells": 50,
            }
        ],
    }
    _save_json(config_dir / "manifest.json", manifest)
    return manifest
