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
from dendritic_modeling.config.conversion import to_plain_dict as _config_to_plain_dict
from dendritic_modeling.config.model import DecoderConfig, EncoderConfig
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, RecurrentClassifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.utils import resolve_experiment_seeds, save_dict, set_seed
from dendritic_modeling.utils.dataset_fingerprint import fingerprint_dataset_splits
from dendritic_modeling.utils.logging_config import LoggerManager

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()


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


def _config_child(config, key: str, default=None):
    """Read a child from dict, OmegaConf, or attribute-based configs."""
    if config is None:
        return default
    if isinstance(config, dict) or OmegaConf.is_config(config):
        return config.get(key, default)
    return getattr(config, key, default)


def _set_config_default(config, key: str, value) -> None:
    """Set a config value only when it is absent or explicitly ``None``."""
    if config is None or _config_child(config, key, None) is not None:
        return
    if isinstance(config, dict) or OmegaConf.is_config(config):
        config[key] = value
    else:
        setattr(config, key, value)


def _set_transfer_topology_seed(transfer_config, topology_seed: int) -> None:
    """Seed random input partitioning while preserving explicit transfer seeds."""
    if transfer_config is None:
        return
    if _config_child(transfer_config, "split_seed", None) is not None:
        return
    if _config_child(transfer_config, "seed", None) is not None:
        return
    _set_config_default(transfer_config, "split_seed", topology_seed)


def _seed_population_topology(population_config, topology_seed: int) -> None:
    if population_config is None:
        return
    _set_config_default(population_config, "indexed_seed", topology_seed)
    _set_config_default(
        _config_child(population_config, "population", None),
        "indexed_seed",
        topology_seed,
    )


def _seed_named_population_topology(population_definition, topology_seed: int) -> None:
    """Seed a named population's nested ``PopulationConfig`` overrides.

    ``PopulationDefinitionConfig`` deliberately keeps model options inside its
    ``population`` mapping. Adding ``indexed_seed`` beside ``name`` or
    ``polarity`` makes dataclass construction fail, so population-network
    definitions need a narrower path than unified-E/I population configs.
    """
    if population_definition is None:
        return
    population_overrides = _config_child(population_definition, "population", None)
    if population_overrides is None:
        population_overrides = {}
        if isinstance(population_definition, dict) or OmegaConf.is_config(
            population_definition
        ):
            population_definition["population"] = population_overrides
        else:
            population_definition.population = population_overrides
    _set_config_default(population_overrides, "indexed_seed", topology_seed)


def _set_initialization_namespace(config, namespace: str) -> None:
    """Fill an absent/empty component namespace without replacing user intent."""
    if config is None:
        return
    configured = _config_child(config, "initialization_namespace", None)
    if configured not in (None, ""):
        return
    if isinstance(config, dict) or OmegaConf.is_config(config):
        config["initialization_namespace"] = namespace
    else:
        config.initialization_namespace = namespace


def _seed_population_initialization(
    population_config,
    model_seed: int,
    *,
    namespace: str,
) -> None:
    """Configure stable parameter-initialization streams for one population."""
    if population_config is None:
        return
    _set_config_default(population_config, "initialization_seed", model_seed)
    _set_initialization_namespace(population_config, namespace)


def _seed_named_population_initialization(
    population_definition,
    model_seed: int,
    *,
    namespace: str,
) -> None:
    """Seed the nested overrides of a named population definition."""
    if population_definition is None:
        return
    population_overrides = _config_child(population_definition, "population", None)
    if population_overrides is None:
        population_overrides = {}
        if isinstance(population_definition, dict) or OmegaConf.is_config(
            population_definition
        ):
            population_definition["population"] = population_overrides
        else:
            population_definition.population = population_overrides
    _seed_population_initialization(
        population_overrides,
        model_seed,
        namespace=namespace,
    )


def _apply_initialization_seed_defaults(
    model_config: ModelConfig,
    model_seed: int,
) -> None:
    """Propagate the model stream into stable population/component substreams."""
    core = _config_child(model_config, "core", None)
    if core is None:
        return
    _set_config_default(core, "initialization_seed", model_seed)
    core_initialization_seed = int(
        _config_child(core, "initialization_seed", model_seed)
    )

    core_type = str(_config_child(core, "type", "")).strip().lower()
    for parameter_name, active_types in (
        (
            "heterogeneous_leak_ctrnn",
            {"heterogeneous_leak_ctrnn", "heterogeneous_ctrnn"},
        ),
        ("legendre_memory", {"legendre_memory"}),
    ):
        parameters = _config_child(core, parameter_name, None)
        if core_type in active_types or parameters:
            _set_config_default(
                parameters,
                "initialization_seed",
                core_initialization_seed,
            )

    unified_ei = _config_child(core, "unified_ei", None)
    for layer_idx, layer in enumerate(_config_child(unified_ei, "layers", []) or []):
        _seed_population_initialization(
            _config_child(layer, "excitatory", None),
            core_initialization_seed,
            namespace=f"unified.layer.{layer_idx}.excitatory",
        )
        _seed_population_initialization(
            _config_child(layer, "inhibitory", None),
            core_initialization_seed,
            namespace=f"unified.layer.{layer_idx}.inhibitory",
        )

    population_network = _config_child(core, "population_network", None)
    for layer_idx, layer in enumerate(
        _config_child(population_network, "layers", []) or []
    ):
        layer_name = str(_config_child(layer, "name", f"layer{layer_idx}"))
        _set_config_default(
            _config_child(layer, "population_defaults", None),
            "initialization_seed",
            core_initialization_seed,
        )
        default_initialization_seed = _config_child(
            _config_child(layer, "population_defaults", None),
            "initialization_seed",
            core_initialization_seed,
        )
        for population_idx, population in enumerate(
            _config_child(layer, "populations", []) or []
        ):
            population_name = str(
                _config_child(population, "name", f"population{population_idx}")
            )
            _seed_named_population_initialization(
                population,
                int(default_initialization_seed),
                namespace=(f"population_network.{layer_name}.{population_name}"),
            )


def _validate_sealed_test_dataset(experiment_config, test_dataset) -> None:
    """Enforce the declared held-out-test audit boundary after construction."""
    if not bool(_get_experiment_value(experiment_config, "sealed_test", False)):
        return
    if bool(_get_experiment_value(experiment_config, "allow_test_data", True)):
        raise ValueError("experiment.sealed_test=true requires allow_test_data=false")
    max_test_samples = _get_experiment_value(
        experiment_config, "max_test_samples", None
    )
    if max_test_samples is None:
        raise ValueError(
            "experiment.sealed_test=true requires max_test_samples to be set"
        )
    max_test_samples = int(max_test_samples)
    if max_test_samples < 0:
        raise ValueError("experiment.max_test_samples must be nonnegative")
    actual_test_samples = len(test_dataset)
    if actual_test_samples > max_test_samples:
        raise ValueError(
            "sealed test dataset exceeds experiment.max_test_samples: "
            f"{actual_test_samples} > {max_test_samples}"
        )


def _apply_topology_seed_defaults(
    model_config: ModelConfig, topology_seed: int
) -> None:
    """Apply one topology stream wherever the model lacks an explicit seed."""
    core = _config_child(model_config, "core", None)
    if core is None:
        return

    core_type = str(_config_child(core, "type", "")).strip().lower()
    for parameter_name, active_types in (
        (
            "heterogeneous_leak_ctrnn",
            {"heterogeneous_leak_ctrnn", "heterogeneous_ctrnn"},
        ),
        ("legendre_memory", {"legendre_memory"}),
    ):
        parameters = _config_child(core, parameter_name, None)
        if core_type in active_types or parameters:
            _set_config_default(parameters, "topology_seed", topology_seed)

    sparsity = _config_child(core, "sparsity", None)
    _set_config_default(_config_child(sparsity, "indexed", None), "seed", topology_seed)
    connectivity = _config_child(core, "connectivity", None)
    _set_config_default(
        _config_child(connectivity, "structured", None), "seed", topology_seed
    )
    _set_transfer_topology_seed(_config_child(core, "transfer", None), topology_seed)

    recurrent_ei = _config_child(core, "recurrent_ei", None)
    _set_transfer_topology_seed(
        _config_child(recurrent_ei, "transfer", None), topology_seed
    )
    _set_transfer_topology_seed(
        _config_child(recurrent_ei, "transfer_params", None), topology_seed
    )

    unified_ei = _config_child(core, "unified_ei", None)
    _set_transfer_topology_seed(
        _config_child(unified_ei, "transfer_params", None), topology_seed
    )
    for layer in _config_child(unified_ei, "layers", []) or []:
        _seed_population_topology(
            _config_child(layer, "excitatory", None), topology_seed
        )
        _seed_population_topology(
            _config_child(layer, "inhibitory", None), topology_seed
        )

    population_network = _config_child(core, "population_network", None)
    _set_transfer_topology_seed(
        _config_child(population_network, "transfer_params", None), topology_seed
    )
    for layer in _config_child(population_network, "layers", []) or []:
        _set_config_default(layer, "connection_seed", topology_seed)
        _seed_population_topology(
            _config_child(layer, "population_defaults", None), topology_seed
        )
        for population in _config_child(layer, "populations", []) or []:
            _seed_named_population_topology(population, topology_seed)


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
            "%s ignored because model.pretrained_replacement.enabled=true or "
            "model.vision_replacement.enabled=true",
            ", ".join(ignored_sections),
        )


def _is_recurrent_core_config(core_cfg) -> bool:
    """Detect whether a core config describes a recurrent model."""
    return is_recurrent_core_config(core_cfg)


def _active_vision_replacement_config(model_config: ModelConfig):
    """Return the active vision replacement config, preferring the new name."""
    vr = getattr(model_config, "vision_replacement", None)
    pr = getattr(model_config, "pretrained_replacement", None)
    if vr is not None and getattr(vr, "enabled", False):
        if pr is not None and getattr(pr, "enabled", False):
            logger.warning(
                "Both model.vision_replacement and model.pretrained_replacement "
                "are enabled; using model.vision_replacement."
            )
        return vr
    if pr is not None and getattr(pr, "enabled", False):
        return pr
    return None


def _build_from_pretrained_replacement(model_config: ModelConfig):
    """Build encoder/core/decoder from a vision replacement config.

    Loads the backbone once, splits it into prefix and suffix via
    :func:`build_split_plan`, then creates the core network in between.
    Returns ``(encoder, core_network, decoder_network, encoder_network)``.
    """
    from dendritic_modeling.networks.architectures.classical.pretrained import (
        build_encoder_decoder_from_plan,
        build_pretrained_core_from_plan,
        build_split_plan,
    )
    from dendritic_modeling.networks.architectures.factory import (
        _SPATIAL_DENDRITIC_CONV_TYPES,
        _SPATIAL_DENDRITIC_TYPES,
        _SPATIAL_PATCH_POINT_TYPES,
    )

    pr = _active_vision_replacement_config(model_config)
    if pr is None:
        raise ValueError("No enabled vision/pretrained replacement config found")
    prefix_freeze = pr.trainability.encoder.mode == "frozen"
    suffix_freeze = pr.trainability.decoder.mode == "frozen"

    # Enable spatial mode when the core is a spatial dendritic type.
    # This preserves TensorSpec metadata and skips auto-flatten on the encoder.
    core_type = (getattr(model_config.core, "type", "") or "").lower()
    core_source = str(getattr(pr, "core_source", "configured")).strip().lower()
    if core_source not in {"configured", "pretrained_span"}:
        raise ValueError(
            "vision_replacement.core_source must be 'configured' or "
            f"'pretrained_span', got {core_source!r}"
        )
    use_pretrained_span = core_source == "pretrained_span"
    is_spatial_conv = core_type in (
        _SPATIAL_DENDRITIC_CONV_TYPES | _SPATIAL_PATCH_POINT_TYPES
    )
    # The configured core type declares the replacement interface even when
    # ``pretrained_span`` supplies the implementation.  This lets an exact
    # retained convolution span use the identical spatial boundary as a
    # dendritic-convolution replacement.
    spatial = core_type in _SPATIAL_DENDRITIC_TYPES or is_spatial_conv

    plan = build_split_plan(
        backbone=pr.backbone,
        weights=pr.weights,
        replace_start=pr.replace.start,
        replace_end=pr.replace.end,
        target_modules=list(getattr(pr, "target_modules", []) or []),
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

    if use_pretrained_span:
        core_network = build_pretrained_core_from_plan(plan, freeze=False)
    elif is_spatial_conv:
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

    # Vision replacement takes precedence when enabled.
    pr = _active_vision_replacement_config(model_config)
    if pr is not None:
        _warn_ignored_legacy_model_sections(model_config)
        encoder, core_network, decoder_network, encoder_network = (
            _build_from_pretrained_replacement(model_config)
        )
    else:
        tr = getattr(model_config, "transformer_replacement", None)
        if tr is not None and getattr(tr, "enabled", False):
            raise ValueError(
                "model.transformer_replacement.enabled=true uses the dedicated "
                "transformer replacement training path. Run "
                "`python -m dendritic_modeling.scripts.training.train_transformer_replacement "
                "<config.yaml>` instead of initialize_model()."
            )

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

        # Create core network. Pass the typed CoreConfig through the factory so
        # dataclass defaults and validation remain the canonical construction
        # surface; the factory still accepts plain dicts for programmatic callers.
        core_network = get_architecture(
            core_config.type, core_config, input_dim=encoder.output_dim
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

    # Detect recurrent mode across structured E/I and population-network cores.
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
        model = Classifier(
            encoder,
            core_network,
            decoder_network,
            learned_output_scale=_get_experiment_value(
                model_config, "learned_output_scale", True
            ),
            fixed_output_scale=_get_experiment_value(
                model_config, "fixed_output_scale", 1.0
            ),
            output_scale_mode=_get_experiment_value(
                model_config, "output_scale_mode", None
            ),
        )
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

    resolved_seeds = resolve_experiment_seeds(experiment_config, write_back=True)
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
    strict_deterministic = resolved_seeds.strict_deterministic
    if strict_deterministic:
        deterministic = True
        cudnn_benchmark = False
        allow_tf32 = False
        float32_matmul_precision = "highest"

    set_seed(
        resolved_seeds.dataset_seed,
        deterministic=deterministic,
        cudnn_benchmark=cudnn_benchmark,
        allow_tf32=allow_tf32,
        float32_matmul_precision=float32_matmul_precision,
        strict_deterministic=strict_deterministic,
    )
    logger.info(
        "Resolved RNG seeds: seed=%s dataset=%s split=%s model=%s topology=%s "
        "loader=%s evaluation=%s probe=%s strict_deterministic=%s",
        resolved_seeds.seed,
        resolved_seeds.dataset_seed,
        resolved_seeds.split_seed,
        resolved_seeds.model_seed,
        resolved_seeds.topology_seed,
        resolved_seeds.loader_seed,
        resolved_seeds.evaluation_seed,
        resolved_seeds.probe_seed,
        strict_deterministic,
    )
    logger.info(
        "Runtime performance: fast_mode=%s deterministic=%s cudnn_benchmark=%s "
        "allow_tf32=%s matmul_precision=%s strict_deterministic=%s",
        fast_mode,
        deterministic,
        cudnn_benchmark and not deterministic,
        allow_tf32,
        float32_matmul_precision,
        strict_deterministic,
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

        if bool(getattr(outputs_config, "exact_run_dir", False)):
            run_save_path = output_dir
            logger.info(
                "Exact run-directory mode - using manifest path: %s",
                run_save_path,
            )
        # Check if this is part of a sweep (has _sweep_config_id)
        elif hasattr(outputs_config, "_sweep_config_id") or (
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

    if is_main:
        save_dict(
            resolved_seeds.asdict(),
            run_save_path,
            "resolved_seeds.json",
        )

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
                "label_noise_seed": resolved_seeds.dataset_seed,
                "split_seed": resolved_seeds.split_seed,
            },
        },
    )()
    task_cfg.parameters.setdefault("seed", resolved_seeds.dataset_seed)

    unified_datasets = get_unified_datasets(
        task_cfg=task_cfg,
    )
    train_ds, valid_ds, test_ds = unified_datasets
    _validate_sealed_test_dataset(experiment_config, test_ds)

    if is_main and bool(
        _get_experiment_value(experiment_config, "record_dataset_fingerprints", False)
    ):
        save_dict(
            fingerprint_dataset_splits(
                train=train_ds,
                valid=valid_ds,
                test=test_ds,
            ),
            run_save_path,
            "dataset_fingerprints.json",
        )

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

    _apply_topology_seed_defaults(model_config, resolved_seeds.topology_seed)
    _apply_initialization_seed_defaults(model_config, resolved_seeds.model_seed)
    set_seed(
        resolved_seeds.model_seed,
        deterministic=deterministic,
        cudnn_benchmark=cudnn_benchmark,
        allow_tf32=allow_tf32,
        float32_matmul_precision=float32_matmul_precision,
        strict_deterministic=strict_deterministic,
    )

    # Initialize model from its independent weight stream. Topology samplers
    # receive their own explicit seeds above and therefore do not consume it.
    model, encoder_network = initialize_model(model_config)

    return run_save_path, train_ds, valid_ds, test_ds, model, encoder_network
