"""
Setup utilities for dendritic modeling training experiments.

This module contains helper functions for setting up the training environment,
including model initialization, dataset loading, and logging configuration.
"""

import logging
import os
from dataclasses import asdict
from datetime import datetime

import torch
import wandb
from omegaconf import OmegaConf

from dendritic_modeling.config import DataConfig, ModelConfig, TrainingConfig
from dendritic_modeling.config.model import DecoderConfig, EncoderConfig
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, RecurrentClassifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.utils import set_seed
from dendritic_modeling.utils.logging_config import LoggerManager

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()


def _config_to_plain_dict(cfg) -> dict:
    """Convert config sections to plain dicts without leaking OmegaConf internals."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if OmegaConf.is_config(cfg):
        resolved = OmegaConf.to_container(cfg, resolve=True)
        return resolved if isinstance(resolved, dict) else {}
    try:
        return asdict(cfg)
    except TypeError:
        if hasattr(cfg, "__dict__"):
            return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}


def _get_dataset_specific_params(data_config: DataConfig) -> dict:
    """Extract dataset-specific parameters for the selected dataset."""
    dataset_name = getattr(data_config, "dataset_name", "")
    dataset_params = getattr(data_config, "dataset_params", None)
    if dataset_params is None:
        return {}
    if isinstance(dataset_params, dict):
        return _config_to_plain_dict(dataset_params.get(dataset_name, {}))
    if OmegaConf.is_config(dataset_params):
        resolved = OmegaConf.to_container(dataset_params, resolve=True)
        if isinstance(resolved, dict):
            return _config_to_plain_dict(resolved.get(dataset_name, {}))
        return {}
    if hasattr(dataset_params, dataset_name):
        return _config_to_plain_dict(getattr(dataset_params, dataset_name))
    return {}


def _get_experiment_value(experiment_config, key: str, default):
    """Read experiment config values from either dicts or dataclass-like configs."""
    if isinstance(experiment_config, dict):
        return experiment_config.get(key, default)
    return getattr(experiment_config, key, default)


def _warn_ignored_legacy_model_sections(model_config: ModelConfig) -> None:
    """Warn when legacy encoder/decoder config is set but ignored."""
    ignored_sections = []

    if _config_to_plain_dict(getattr(model_config, "encoder", None)) != asdict(
        EncoderConfig()
    ):
        ignored_sections.append("model.encoder")

    if _config_to_plain_dict(getattr(model_config, "decoder", None)) != asdict(
        DecoderConfig()
    ):
        ignored_sections.append("model.decoder")

    if ignored_sections:
        logger.warning(
            "%s ignored because model.pretrained_replacement.enabled=true",
            ", ".join(ignored_sections),
        )


def _is_recurrent_core_config(core_cfg) -> bool:
    """Detect whether a core config describes a recurrent model."""

    def _get_type(cfg) -> str:
        if isinstance(cfg, dict):
            return str(cfg.get("type", "")).lower()
        return str(getattr(cfg, "type", "")).lower()

    def _get_recurrent_enabled(cfg) -> bool:
        rec = (
            cfg.get("recurrent", {})
            if isinstance(cfg, dict)
            else getattr(cfg, "recurrent", None)
        )
        if isinstance(rec, dict):
            return bool(rec.get("enabled", False))
        return bool(getattr(rec, "enabled", False))

    def _has_unified_recurrent_layers(cfg) -> bool:
        unified = (
            cfg.get("unified_ei", cfg.get("ei_unified", {}))
            if isinstance(cfg, dict)
            else getattr(cfg, "unified_ei", {})
        )
        if not unified:
            return False
        layers = (
            unified.get("layers", [])
            if isinstance(unified, dict)
            else getattr(unified, "layers", [])
        )
        for layer in layers:
            if isinstance(layer, dict) and layer.get("recurrent", False):
                return True
            if getattr(layer, "recurrent", False):
                return True
        return False

    def _has_recurrent_einet_layers(cfg) -> bool:
        def _any_positive_in(mapping, keys):
            for key in keys:
                val = (
                    mapping.get(key, 0)
                    if isinstance(mapping, dict)
                    else getattr(mapping, key, 0)
                )
                if isinstance(val, list):
                    if any(v > 0 for v in val):
                        return True
                elif val is not None and val > 0:
                    return True
            return False

        # New path: check connectivity.rec_* fields
        conn = (
            cfg.get("connectivity", {})
            if isinstance(cfg, dict)
            else getattr(cfg, "connectivity", {})
        )
        if conn and _any_positive_in(
            conn,
            (
                "rec_ee_synapses_per_branch",
                "rec_ie_synapses_per_branch",
                "rec_ei_synapses_per_branch",
                "rec_ii_synapses_per_branch",
            ),
        ):
            return True

        # Old path (backward compat): check recurrent_ei block
        recurrent_ei = (
            cfg.get("recurrent_ei", {})
            if isinstance(cfg, dict)
            else getattr(cfg, "recurrent_ei", {})
        )
        if not recurrent_ei:
            return False

        enabled = (
            recurrent_ei.get("enabled", False)
            if isinstance(recurrent_ei, dict)
            else getattr(recurrent_ei, "enabled", False)
        )
        if enabled:
            per_layer = (
                recurrent_ei.get("recurrent_layers", None)
                if isinstance(recurrent_ei, dict)
                else getattr(recurrent_ei, "recurrent_layers", None)
            )
            if per_layer is None:
                return True
            if isinstance(per_layer, list):
                return any(bool(v) for v in per_layer)
            return bool(per_layer)

        # Also check old-style synapse fields in recurrent_ei
        return _any_positive_in(
            recurrent_ei,
            (
                "rec_excitatory_synapses",
                "rec_inhibitory_synapses",
                "rec_excitatory_synapses_per_layer",
                "rec_inhibitory_synapses_per_layer",
            ),
        )

    core_type = _get_type(core_cfg)
    if core_type in {"gru", "lstm", "vanilla_rnn", "rnn"}:
        return True
    if core_type in {"ei_unified", "unified_ei", "ei_net", "unified_einet"}:
        return _has_unified_recurrent_layers(core_cfg)
    if core_type in {
        "einet",
        "dendritic_shunting",
        "dendritic_additive",
        "flat_shunting",
        "flat_additive",
        "dendritic_mlp",
    }:
        return _has_recurrent_einet_layers(core_cfg) or _get_recurrent_enabled(core_cfg)
    return _get_recurrent_enabled(core_cfg)


def _build_from_pretrained_replacement(model_config: ModelConfig):
    """Build encoder/core/decoder from a ``pretrained_replacement`` config.

    Loads the backbone once, splits it into prefix and suffix via
    :func:`build_split_plan`, then creates the core network in between.
    Returns ``(encoder, core_network, decoder_network, encoder_network)``.
    """
    from dendritic_modeling.networks.architectures.classical.pretrained import (
        build_encoder_decoder_from_plan,
        build_split_plan,
    )
    from dendritic_modeling.networks.architectures.factory import (
        _SPATIAL_DENDRITIC_CONV_TYPES,
        _SPATIAL_DENDRITIC_TYPES,
    )

    pr = model_config.pretrained_replacement
    prefix_freeze = pr.trainability.encoder.mode == "frozen"
    suffix_freeze = pr.trainability.decoder.mode == "frozen"

    # Enable spatial mode when the core is a spatial dendritic type.
    # This preserves TensorSpec metadata and skips auto-flatten on the encoder.
    core_type = (getattr(model_config.core, "type", "") or "").lower()
    is_spatial_conv = core_type in _SPATIAL_DENDRITIC_CONV_TYPES
    spatial = core_type in _SPATIAL_DENDRITIC_TYPES or is_spatial_conv

    plan = build_split_plan(
        backbone=pr.backbone,
        weights=pr.weights,
        replace_start=pr.replace.start,
        replace_end=pr.replace.end,
        omit_layers=list(pr.omit_layers) if pr.omit_layers else None,
        input_shape=list(pr.input_shape) if pr.input_shape else None,
        spatial=spatial,
    )

    encoder, decoder_network = build_encoder_decoder_from_plan(
        plan,
        prefix_freeze=prefix_freeze,
        suffix_freeze=suffix_freeze,
    )

    # Build the core network.
    core_params = _config_to_plain_dict(model_config.core)

    if is_spatial_conv:
        # ----- Validate spatial contract -----
        if plan.prefix_output_spec is None:
            raise ValueError(
                f"Spatial conv core (type={core_type!r}) requires a spatial "
                f"encoder output, but the prefix does not produce spatial "
                f"features (prefix_output_spec is None). Either split the "
                f"backbone earlier so the prefix outputs (B, C, H, W), or "
                f"use a flat core type (e.g., tensor_map, einet)."
            )
        if plan.suffix_input_spec is None:
            raise ValueError(
                f"Spatial conv core (type={core_type!r}) produces spatial "
                f"output (B, C_out, H_out, W_out), but the suffix expects "
                f"flat input (suffix_input_spec is None). Either split the "
                f"backbone so the suffix starts with a spatial layer, or "
                f"use a flat core type (e.g., tensor_map, einet)."
            )

        # Build the core.
        core_input_dim = plan.prefix_output_spec.channels
        core_network = get_architecture(
            model_config.core.type,
            core_params,
            input_dim=core_input_dim,
        )

        # Validate output channels match suffix expectation.
        if hasattr(core_network, "out_channels"):
            if core_network.out_channels != plan.suffix_input_spec.channels:
                raise ValueError(
                    f"Spatial conv core out_channels={core_network.out_channels} "
                    f"does not match suffix expected channels="
                    f"{plan.suffix_input_spec.channels}. Set "
                    f"architecture.excitatory_layer_sizes[-1] to "
                    f"{plan.suffix_input_spec.channels}."
                )

        # Validate spatial dimensions match suffix expectation.
        if hasattr(core_network, "compute_output_shape"):
            H_out, W_out = core_network.compute_output_shape(
                plan.prefix_output_spec.height,
                plan.prefix_output_spec.width,
            )
            if (
                H_out != plan.suffix_input_spec.height
                or W_out != plan.suffix_input_spec.width
            ):
                raise ValueError(
                    f"Spatial conv core output shape ({H_out}, {W_out}) does "
                    f"not match suffix expected spatial dims "
                    f"({plan.suffix_input_spec.height}, "
                    f"{plan.suffix_input_spec.width}). Adjust kernel_size, "
                    f"stride, and padding in core.spatial."
                )
    else:
        # Tensor map / flat: input_dim = flattened dim, pass suffix_input_dim
        # for the output projection.
        core_network = get_architecture(
            model_config.core.type,
            core_params,
            input_dim=encoder.output_dim,
            suffix_input_dim=plan.suffix_input_dim,
        )

        if (
            hasattr(core_network, "output_dim")
            and core_network.output_dim != plan.suffix_input_dim
        ):
            raise ValueError(
                f"Core output_dim={core_network.output_dim} does not match "
                f"suffix expected input_dim={plan.suffix_input_dim}. "
                f"The suffix decoder's first Linear layer expects "
                f"{plan.suffix_input_dim} features."
            )

    return encoder, core_network, decoder_network, encoder


def initialize_model(model_config: ModelConfig):
    """Initialize the model based on configuration."""

    # New API: pretrained_replacement takes precedence when enabled
    pr = getattr(model_config, "pretrained_replacement", None)
    if pr is not None and getattr(pr, "enabled", False):
        _warn_ignored_legacy_model_sections(model_config)
        encoder, core_network, decoder_network, encoder_network = (
            _build_from_pretrained_replacement(model_config)
        )
    else:
        # Legacy path
        encoder_config = model_config.encoder
        core_config = model_config.core
        decoder_config = model_config.decoder

        # Build encoder parameters dict
        encoder_params = {}
        if encoder_config.type.lower() == "identity":
            encoder_params["input_dim"] = encoder_config.params.input_dim or 784
        else:
            encoder_params = _config_to_plain_dict(encoder_config.params)

        # Create encoder
        encoder_network = get_architecture(encoder_config.type, encoder_params)
        if isinstance(encoder_network, Identity):
            encoder = encoder_network
        elif hasattr(encoder_network, "encoder"):
            encoder = encoder_network.encoder
        else:
            encoder = encoder_network

        # Create core network
        core_params = _config_to_plain_dict(core_config)
        core_network = get_architecture(
            core_config.type, core_params, input_dim=encoder.output_dim
        )

        # Build decoder parameters
        decoder_params_dict = _config_to_plain_dict(decoder_config.params)
        # Filter out None values to allow auto-setting
        decoder_params_dict = {
            k: v for k, v in decoder_params_dict.items() if v is not None
        }
        decoder_params = {
            **decoder_params_dict,
            "input_dim": core_network.output_dim,
        }

        # Create decoder
        decoder_network = get_architecture(decoder_config.type, decoder_params)

    # Detect recurrent mode (supports dendritic_rnn and ei_unified paths).
    is_recurrent = _is_recurrent_core_config(model_config.core)

    # Create model based on task
    if is_recurrent:
        if model_config.task == "classification":
            model = RecurrentClassifier(encoder, core_network, decoder_network)
        elif model_config.task == "regression":
            model = Regressor(encoder, core_network, decoder_network)
        else:
            raise ValueError(f"Invalid task: {model_config.task}")
    elif model_config.task == "classification":
        model = Classifier(encoder, core_network, decoder_network)
    elif model_config.task == "regression":
        model = Regressor(encoder, core_network, decoder_network)
    else:
        raise ValueError(f"Invalid task: {model_config.task}")

    logger.info(f"Initialized model: {type(model).__name__}")

    return model, encoder_network


def setup_environment(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    wandb_config,
    outputs_config,
    experiment_config,
    is_main: bool = True,
):
    """Setup training environment.

    Args:
        is_main: When False (non-rank-0 DDP processes), skip output
            directory creation, file-handler setup, and wandb init.
            The caller is responsible for broadcasting run_save_path
            from rank 0 to other ranks.
    """

    # Set seed
    seed = experiment_config.seed
    fast_mode = bool(_get_experiment_value(experiment_config, "fast_mode", False))
    deterministic = bool(
        _get_experiment_value(experiment_config, "deterministic", not fast_mode)
    )
    cudnn_benchmark = bool(
        _get_experiment_value(
            experiment_config, "cudnn_benchmark", fast_mode and not deterministic
        )
    )
    allow_tf32 = bool(_get_experiment_value(experiment_config, "allow_tf32", fast_mode))
    float32_matmul_precision = str(
        _get_experiment_value(
            experiment_config,
            "float32_matmul_precision",
            "high" if fast_mode else "highest",
        )
    )

    set_seed(
        seed,
        deterministic=deterministic,
        cudnn_benchmark=cudnn_benchmark,
        allow_tf32=allow_tf32,
        float32_matmul_precision=float32_matmul_precision,
    )
    logger.info(f"Set random seed to {seed}")
    logger.info(
        "Runtime performance: fast_mode=%s deterministic=%s cudnn_benchmark=%s "
        "allow_tf32=%s matmul_precision=%s",
        fast_mode,
        deterministic,
        cudnn_benchmark and not deterministic,
        allow_tf32,
        float32_matmul_precision,
    )

    # Setup output directory — only rank 0 creates dirs and log files.
    # Non-rank-0 processes get a placeholder run_save_path that will be
    # overwritten by the broadcast from rank 0 in the caller.
    output_dir = outputs_config.results_dir
    run_name = getattr(outputs_config, "run_name", "run")

    if is_main:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        # Check if this is part of a sweep (has _sweep_config_id)
        if hasattr(outputs_config, "_sweep_config_id") or (
            hasattr(outputs_config, "run_name")
            and "/config_" in str(outputs_config.results_dir)
        ):
            run_save_path = output_dir
            logger.info(f"Sweep mode detected - using directory: {run_save_path}")
        else:
            run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_save_path = os.path.join(output_dir, run_id)
            os.makedirs(run_save_path, exist_ok=True)
            logger.info(f"Created run directory: {run_save_path}")

        logger_manager.set_log_directory(run_save_path)

        log_file = os.path.join(run_save_path, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    else:
        # Placeholder — caller will broadcast the real path from rank 0.
        run_save_path = output_dir

    # Setup wandb if enabled (rank 0 only; caller disables on other ranks)
    if wandb_config.use_wandb and is_main:
        try:
            wandb.init(
                entity=wandb_config.entity,
                project=wandb_config.project,
                group=wandb_config.group,
                tags=wandb_config.tags,
                name=run_name,
                config={
                    "model": asdict(model_config),
                    "training": asdict(training_config),
                    "data": asdict(data_config),
                    "experiment": asdict(experiment_config),
                },
            )
            logger.info("Successfully initialized W&B")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B logging")
            wandb_config.use_wandb = False

    base_dir = data_config.base_dir or output_dir
    dataset_specific_params = _get_dataset_specific_params(data_config)
    processing_params = _config_to_plain_dict(getattr(data_config, "processing", None))

    task_cfg = type(
        "TaskConfig",
        (),
        {
            "dataset": data_config.dataset_name,
            # Pass the configured base_dir through directly. Dataset loaders decide
            # whether to treat it as a dataset root (for example ImageNet train/val
            # trees) or as a parent directory that still needs dataset-specific
            # subdirectories.
            "data_path": (base_dir if base_dir else None),
            "train_valid_split": experiment_config.train_valid_split,
            "parameters": {
                **processing_params,
                **dataset_specific_params,
                "label_noise_seed": experiment_config.seed,
            },
        },
    )()

    unified_datasets = get_unified_datasets(
        task_cfg=task_cfg,
    )
    train_ds, valid_ds, test_ds = unified_datasets

    logger.info(
        f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples"
    )

    # Update encoder input dimension (only if params exists)
    input_sample: torch.Tensor = train_ds[0][0]

    # Detect if this is sequence data (recurrent mode)
    is_recurrent = _is_recurrent_core_config(model_config.core)

    if is_recurrent:
        # Sequence data: [seq_len, input_dim] -- use last dim as feature dim
        input_dim = input_sample.shape[-1]
    else:
        # Image/vector data -- use total elements
        input_dim = input_sample.numel()
    input_shape = list(input_sample.shape)

    if (
        hasattr(model_config.encoder, "params")
        and model_config.encoder.params is not None
    ):
        model_config.encoder.params.input_dim = input_dim
        model_config.encoder.params.input_shape = input_shape
    else:
        # Create params if it doesn't exist (e.g., for identity encoder)
        model_config.encoder.params = {
            "input_dim": input_dim,
            "input_shape": input_shape,
        }

    # Initialize model
    model, encoder_network = initialize_model(model_config)

    return run_save_path, train_ds, valid_ds, test_ds, model, encoder_network
