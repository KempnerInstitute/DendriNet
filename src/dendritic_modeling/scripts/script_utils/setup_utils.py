"""
Setup utilities for dendritic modeling training experiments.

This module contains helper functions for setting up the training environment,
including model initialization, dataset loading, and logging configuration.
"""

import logging
import math
import os
import re
from datetime import datetime

import torch
from omegaconf import OmegaConf

from dendritic_modeling.config import DataConfig, ModelConfig, TrainingConfig
from dendritic_modeling.config.conversion import (
    dataset_specific_params as _get_dataset_specific_params,
    to_plain_dict as _config_to_plain_dict,
)
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.datasets.parameters import set_processing_parameters
from dendritic_modeling.datasets.paths import resolve_data_root
from dendritic_modeling.datasets.registry import reject_unsupported_training_strategy
from dendritic_modeling.models import Classifier, RecurrentClassifier, Regressor
from dendritic_modeling.models.builders import resolve_model_builder
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import (
    _EINET_TYPES,
    get_architecture,
)
from dendritic_modeling.networks.architectures.factory_spatial import (
    FEATURE_MAP_CORE_TYPES,
    IMAGE_INPUT_CORE_TYPES,
    SPATIAL_DENDRITIC_CONV_TYPES,
    SPATIAL_DENDRITIC_TYPES,
)
from dendritic_modeling.networks.architectures.vision import FlattenedSpatialCore
from dendritic_modeling.networks.checkpoints import load_initial_model_checkpoint
from dendritic_modeling.utils import resolve_experiment_seeds, save_dict, set_seed
from dendritic_modeling.utils.dataset_fingerprint import fingerprint_dataset_splits
from dendritic_modeling.utils.logging_config import LoggerManager
from dendritic_modeling.utils.reproducibility import (
    resolve_runtime_performance_settings,
)
from dendritic_modeling.utils.runtime_environment import (
    RUNTIME_ENVIRONMENT_FILE,
    runtime_environment,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()


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


def _has_key(config, key: str) -> bool:
    if isinstance(config, dict) or OmegaConf.is_config(config):
        return key in config
    return hasattr(config, key)


def _unified_structured_seed(topology_seed: int, layer_index: int) -> int:
    """The structured-connectivity seed training gives a unified E/I layer.

    Unified layers draw their masks from the same stream components (only the
    pathway, level and shape separate them), so one seed would give layers of
    equal shape equal masks; each later layer gets its own seed.
    """
    if not layer_index:
        return int(topology_seed)
    return (
        int(topology_seed) + stable_seed_offset("unified_ei.structured", layer_index)
    ) % ((1 << 63) - 1)


def _seed_population_structured_rules(layer, topology_seed: int) -> None:
    """Seed the unseeded structured rules of a population-network layer.

    A population's rule is ``population_defaults.structured_connectivity``
    merged with its own ``population.structured_connectivity``; the masks of
    every population, layer and pathway still draw from separate streams of
    the seed. A written seed (including ``global_rng``) is kept.
    """
    defaults = _config_child(
        _config_child(layer, "population_defaults", None),
        "structured_connectivity",
        None,
    )
    _set_config_default(defaults, "seed", topology_seed)
    for population in _config_child(layer, "populations", []) or []:
        own = _config_child(
            _config_child(population, "population", None),
            "structured_connectivity",
            None,
        )
        # Without its own seed key a population inherits the layer default.
        if own is not None and (defaults is None or _has_key(own, "seed")):
            _set_config_default(own, "seed", topology_seed)


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


# Feedforward core types built through ``prepare_ei_network_params``. Their
# builders ignored ``core.initialization_seed`` before 2026-10-04, so training
# leaves it unset and keeps their historical global-stream initialization; an
# explicitly configured seed selects keyed initialization. The comparison
# control ``routed_ei`` uses the global stream too, so training initializes it
# as direct construction after ``torch.manual_seed(model_seed)`` and any
# encoder draws would.
_GLOBAL_STREAM_FEEDFORWARD_CORE_TYPES = frozenset(
    set(_EINET_TYPES)
    | set(SPATIAL_DENDRITIC_TYPES)
    | set(SPATIAL_DENDRITIC_CONV_TYPES)
    | {"ss_mlp", "ss_mlp_flat", "routed_ei"}
)


def _keeps_global_initialization_stream(core) -> bool:
    """Whether training must not fill ``core.initialization_seed``."""
    core_type = str(_config_child(core, "type", "") or "").strip().lower()
    if core_type not in _GLOBAL_STREAM_FEEDFORWARD_CORE_TYPES:
        return False
    # Recurrent EINet configurations already consume the core seed.
    return core_type not in _EINET_TYPES or not _is_recurrent_core_config(core)


def _apply_initialization_seed_defaults(
    model_config: ModelConfig,
    model_seed: int,
) -> None:
    """Propagate the model stream into stable population/component substreams."""
    core = _config_child(model_config, "core", None)
    if core is None:
        return
    if not _keeps_global_initialization_stream(core):
        _set_config_default(core, "initialization_seed", model_seed)
    configured_core_seed = _config_child(core, "initialization_seed", None)
    core_initialization_seed = int(
        model_seed if configured_core_seed is None else configured_core_seed
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
    """Enforce the declared held-out-test boundary after construction.

    ``max_test_samples`` bounds the test split whenever it is set; a sealed
    test additionally requires ``allow_test_data: false`` and the bound.
    """
    sealed = bool(_get_experiment_value(experiment_config, "sealed_test", False))
    max_test_samples = _get_experiment_value(
        experiment_config, "max_test_samples", None
    )
    if sealed:
        if bool(_get_experiment_value(experiment_config, "allow_test_data", True)):
            raise ValueError(
                "experiment.sealed_test=true requires allow_test_data=false"
            )
        if max_test_samples is None:
            raise ValueError(
                "experiment.sealed_test=true requires max_test_samples to be set"
            )
    if max_test_samples is None:
        return
    max_test_samples = int(max_test_samples)
    if max_test_samples < 0:
        raise ValueError("experiment.max_test_samples must be nonnegative")
    actual_test_samples = len(test_dataset)
    if actual_test_samples > max_test_samples:
        raise ValueError(
            "test dataset exceeds experiment.max_test_samples: "
            f"{actual_test_samples} > {max_test_samples}"
        )


def _apply_topology_seed_defaults(
    model_config: ModelConfig, topology_seed: int
) -> None:
    """Fill unset topology seeds from the experiment topology stream.

    A seed written in the config is kept; this only fills seeds that are unset,
    including the structured-connectivity seeds of structured E/I cores,
    population-network populations and unified E/I populations.
    """
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

    unified_ei = _config_child(core, "unified_ei", None)
    _set_transfer_topology_seed(
        _config_child(unified_ei, "transfer_params", None), topology_seed
    )
    for layer_index, layer in enumerate(_config_child(unified_ei, "layers", []) or []):
        for role in ("excitatory", "inhibitory"):
            population = _config_child(layer, role, None)
            _seed_population_topology(population, topology_seed)
            # config_version 3: an unseeded structured rule draws its masks
            # from the topology seed instead of the global random stream.
            _set_config_default(
                _config_child(population, "structured_connectivity", None),
                "seed",
                _unified_structured_seed(topology_seed, layer_index),
            )

    population_network = _config_child(core, "population_network", None)
    _set_transfer_topology_seed(
        _config_child(population_network, "transfer_params", None), topology_seed
    )
    for layer in _config_child(population_network, "layers", []) or []:
        _set_config_default(layer, "connection_seed", topology_seed)
        # A layer's explicit population_defaults.indexed_seed is the base seed
        # of its populations, exactly as for initialization seeds; the builder
        # namespaces it by layer and population.
        population_defaults = _config_child(layer, "population_defaults", None)
        _seed_population_topology(population_defaults, topology_seed)
        layer_indexed_seed = _config_child(
            population_defaults, "indexed_seed", topology_seed
        )
        for population in _config_child(layer, "populations", []) or []:
            _seed_named_population_topology(population, int(layer_indexed_seed))
        # config_version 3: an unseeded structured rule draws its masks from
        # the topology seed instead of the global random stream.
        _seed_population_structured_rules(layer, topology_seed)


def _is_recurrent_core_config(core_cfg) -> bool:
    """Detect whether a core config describes a recurrent model."""
    return is_recurrent_core_config(core_cfg)


def _reject_unapplied_output_scale(model_config, *, wrapper: str) -> None:
    """Fail when output-scale settings differ from the unscaled wrapper output.

    Recurrent and regression wrappers return the decoder output unchanged, which
    equals ``output_scale_mode='fixed'`` with ``fixed_output_scale=1``.
    """
    unapplied = []
    fixed_output_scale = float(
        _get_experiment_value(model_config, "fixed_output_scale", 1.0)
    )
    if fixed_output_scale != 1.0:
        unapplied.append(f"model.fixed_output_scale={fixed_output_scale!r}")
    output_scale_mode = _get_experiment_value(model_config, "output_scale_mode", None)
    if (
        output_scale_mode is not None
        and str(output_scale_mode).strip().lower() != "fixed"
    ):
        unapplied.append(f"model.output_scale_mode={output_scale_mode!r}")
    if unapplied:
        raise ValueError(
            f"The {wrapper} model wrapper applies no output scale, so "
            + ", ".join(unapplied)
            + " would be ignored. Use output_scale_mode: fixed with "
            "fixed_output_scale: 1.0, or remove these fields."
        )


def _reject_empty_validation_selection(valid_ds, training_config, experiment_config):
    """Fail when validation drives selection but the validation set is empty.

    With no validation samples the validation loss is 0.0 every epoch, so
    load_best_state_dict restores the epoch-1 weights and early stopping counts
    every later epoch as no improvement.
    """
    try:
        empty = len(valid_ds) == 0
    except TypeError:
        return
    common = getattr(getattr(training_config, "main", None), "common", None)
    selection = [
        name
        for name in ("load_best_state_dict", "early_stopping")
        if bool(getattr(common, name, False))
    ]
    if empty and selection:
        split = _get_experiment_value(experiment_config, "train_valid_split", None)
        raise ValueError(
            f"experiment.train_valid_split={split} leaves no validation samples, "
            f"but {' and '.join(selection)} select on validation loss, which is "
            "undefined without validation samples. Lower train_valid_split, "
            "or set training.main.common.load_best_state_dict and early_stopping "
            "to false."
        )


def _training_class_count(dataset) -> int | None:
    """Return the class count of stored 1-D integer training labels, else None.

    A dataset that lists its ``classes`` reports their number; otherwise the
    count is the largest stored label plus one.
    """
    indices = None
    while isinstance(dataset, torch.utils.data.Subset):
        subset_indices = torch.as_tensor(dataset.indices)
        indices = subset_indices if indices is None else subset_indices[indices]
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, (list, tuple)) and classes:
        return len(classes)
    if isinstance(dataset, torch.utils.data.TensorDataset):
        labels = dataset.tensors[1] if len(dataset.tensors) > 1 else None
    else:
        labels = getattr(dataset, "targets", None)
    if labels is None:
        return None
    labels = torch.as_tensor(labels)
    if labels.dim() != 1 or labels.is_floating_point() or labels.dtype == torch.bool:
        return None
    if indices is not None:
        labels = labels[indices]
    if labels.numel() == 0 or int(labels.min().item()) < 0:
        return None
    return int(labels.max().item()) + 1


def _check_classification_output_width(model_config, model, train_ds) -> None:
    """Compare the classifier's output width with the training labels.

    Labels beyond the outputs raise; outputs beyond the labels only warn,
    because a subset need not contain every class.
    """
    if str(getattr(model_config, "task", "")) != "classification":
        return
    width = getattr(getattr(model, "decoder_network", None), "output_dim", None)
    n_classes = _training_class_count(train_ds)
    if width is None or n_classes is None:
        return
    if n_classes > int(width):
        raise ValueError(
            f"The training labels span {n_classes} classes but the model has "
            f"{width} outputs. Set model.decoder.params.output_dim (or, with an "
            f"identity decoder, the core output size) to {n_classes}."
        )
    if n_classes < int(width):
        logger.warning(
            "The model has %d outputs but the training labels span %d classes; "
            "the extra outputs are trained toward zero probability.",
            width,
            n_classes,
        )


def _image_input_shape(core_config, encoder_config, encoder) -> tuple[int, int, int]:
    """Return the ``(C, H, W)`` images an image-input core reads, or raise.

    Image cores read the model input directly, so they need the identity
    encoder and ``model.encoder.params.input_shape`` ``[C, H, W]``. Training
    sets ``input_shape`` and ``input_dim`` (``C * H * W``) from a data sample,
    which therefore must be an image; a configuration built without data may
    give ``input_dim`` as ``C``.
    """
    core_type = str(core_config.type).strip().lower()
    if not isinstance(encoder, Identity):
        raise ValueError(
            f"model.core.type={core_type!r} reads images directly, so it needs "
            f"the identity encoder, not model.encoder.type={encoder_config.type!r}"
        )
    configured_shape = _config_child(encoder_config.params, "input_shape", None)
    shape = [int(value) for value in configured_shape or []]
    if len(shape) != 3 or int(encoder.output_dim) not in {math.prod(shape), shape[0]}:
        raise ValueError(
            f"model.core.type={core_type!r} reads images (C, H, W), but the "
            f"model input has model.encoder.params.input_shape={shape} and "
            f"input_dim={encoder.output_dim}. Training sets both from the data, "
            "so load the dataset as images (data.processing.flatten: false); "
            "otherwise set input_shape to [C, H, W] and input_dim to C * H * W."
        )
    vision = _config_child(core_config, "vision", None) or {}
    configured_channels = _config_child(vision, "input_channels", None)
    if configured_channels is not None and int(configured_channels) != shape[0]:
        raise ValueError(
            f"model.core.vision.input_channels={configured_channels} differs from "
            f"the {shape[0]} channels of the input images {shape}"
        )
    return shape[0], shape[1], shape[2]


def _check_identity_decoder_output(decoder_config, core_output_dim: int) -> None:
    """An identity decoder's written output_dim must equal the core output."""
    if str(decoder_config.type).strip().lower() not in {"identity", "none"}:
        return
    expected = getattr(decoder_config.params, "output_dim", None)
    if expected is not None and int(expected) != int(core_output_dim):
        raise ValueError(
            f"model.decoder.type={decoder_config.type!r} passes the core output "
            f"through, but model.decoder.params.output_dim={expected} differs "
            f"from the core output dimension {core_output_dim}"
        )


def run_directory(outputs_config, *, now: datetime | None = None) -> str:
    """The directory a run writes into.

    ``outputs.exact_run_dir: true`` (set by sweeps and analysis tools) uses
    ``results_dir`` itself, as do configs generated by earlier sweep versions,
    whose ``results_dir`` ends in ``config_<i>``. Otherwise each run gets a new
    ``<run_name>_<timestamp>`` directory under ``results_dir``.
    """
    output_dir = outputs_config.results_dir
    if bool(getattr(outputs_config, "exact_run_dir", False)):
        return output_dir
    if re.fullmatch(r"config_\d+", os.path.basename(os.path.normpath(output_dir))):
        return output_dir
    run_name = getattr(outputs_config, "run_name", "run")
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{run_name}_{stamp}")


def claim_run_directory(outputs_config, *, now: datetime | None = None) -> str:
    """Create the run directory and return its path.

    A timestamped directory that already exists, because another run with the
    same ``results_dir`` and ``run_name`` started in the same second, is not
    shared: the run takes the first free ``<name>_<n>`` instead. Creation is
    atomic, so concurrent runs cannot claim the same directory.
    """
    path = run_directory(outputs_config, now=now)
    if path == outputs_config.results_dir:
        os.makedirs(path, exist_ok=True)
        return path
    candidate, suffix = path, 1
    while True:
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            candidate = f"{path}_{suffix}"
            suffix += 1


def _reject_unbuilt_extension_sections(model_config: ModelConfig) -> None:
    """Fail for an enabled plugin section that no registered model builder builds."""
    for name in ("vision_replacement", "pretrained_replacement"):
        if getattr(getattr(model_config, name, None), "enabled", False):
            raise ValueError(
                f"model.{name} is enabled, but no installed DendriNet plugin "
                "builds it (see dendritic_modeling.models.builders)"
            )
    tr = getattr(model_config, "transformer_replacement", None)
    if tr is not None and getattr(tr, "enabled", False):
        raise ValueError(
            "model.transformer_replacement.enabled=true replaces layers of a "
            "pretrained transformer, which initialize_model() does not build. "
            "Build it with dendritic_modeling.networks.architectures.transformer."
            "load_transformer_with_dendritic_replacements, or train it with the "
            "program of the plugin that provides the section."
        )


def initialize_model(model_config: ModelConfig):
    """Initialize the model based on configuration.

    A registered model builder that applies to ``model_config`` (see
    :mod:`dendritic_modeling.models.builders`) builds the encoder, core and
    decoder; otherwise they are built from ``model.encoder``, ``model.core``
    and ``model.decoder``. Both are then wrapped for the task.
    """

    builder = resolve_model_builder(model_config)
    if builder is not None:
        encoder, core_network, decoder_network, encoder_network = builder.build(
            model_config
        )
    else:
        _reject_unbuilt_extension_sections(model_config)
        encoder, core_network, decoder_network, encoder_network = (
            _build_model_components(model_config)
        )

    # Detect recurrent mode across structured E/I and population-network cores.
    is_recurrent = _is_recurrent_core_config(model_config.core)
    if is_recurrent or model_config.task == "regression":
        _reject_unapplied_output_scale(
            model_config,
            wrapper="recurrent" if is_recurrent else "regression",
        )

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


def _build_model_components(model_config: ModelConfig):
    """Build the encoder, core and decoder named in ``model``."""
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
    core_type = str(core_config.type or "").strip().lower()
    core_input_dim = encoder.output_dim
    image_shape = None
    if core_type in IMAGE_INPUT_CORE_TYPES:
        image_shape = _image_input_shape(core_config, encoder_config, encoder)
        core_input_dim = image_shape[0]
    core_network = get_architecture(
        core_config.type, core_config, input_dim=core_input_dim
    )
    if core_type in FEATURE_MAP_CORE_TYPES:
        # The decoder reads vectors: flatten the core's feature maps.
        core_network = FlattenedSpatialCore(core_network, image_shape)
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
    _check_identity_decoder_output(decoder_config, core_network.output_dim)
    # Create decoder
    decoder_network = get_architecture(decoder_config.type, decoder_params)
    return encoder, core_network, decoder_network, encoder_network


def setup_environment(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    outputs_config,
    experiment_config,
    is_main: bool = True,
):
    """Setup training environment.

    Args:
        is_main: When False (non-rank-0 DDP processes), skip output
            directory creation and file-handler setup.
            The caller is responsible for broadcasting run_save_path
            from rank 0 to other ranks.
    """

    resolved_seeds = resolve_experiment_seeds(experiment_config, write_back=True)
    fast_mode = bool(_get_experiment_value(experiment_config, "fast_mode", False))
    performance = resolve_runtime_performance_settings(
        experiment_config, write_back=True
    )
    deterministic = performance["deterministic"]
    cudnn_benchmark = performance["cudnn_benchmark"]
    allow_tf32 = performance["allow_tf32"]
    float32_matmul_precision = performance["float32_matmul_precision"]
    strict_deterministic = resolved_seeds.strict_deterministic
    numerics = set_seed(
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

    if is_main:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        run_save_path = claim_run_directory(outputs_config)
        if run_save_path == output_dir:
            logger.info("Writing the run into %s", run_save_path)
        else:
            logger.info(f"Created run directory: {run_save_path}")

        # dendritic_modeling.log is the run's complete log; train.log keeps the
        # run-setup records of this module, replacing an earlier run's file.
        logger_manager.set_log_directory(run_save_path)
        for handler in [h for h in logger.handlers if h.get_name() == "train_log"]:
            logger.removeHandler(handler)
            handler.close()
        log_file = os.path.join(run_save_path, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.set_name("train_log")
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
            {**resolved_seeds.asdict(), "numerics": numerics},
            run_save_path,
            "resolved_seeds.json",
        )
        save_dict(runtime_environment(), run_save_path, RUNTIME_ENVIRONMENT_FILE)

    reject_unsupported_training_strategy(
        data_config.dataset_name,
        _get_experiment_value(training_config.main, "strategy", "standard"),
    )
    base_dir = resolve_data_root(data_config.base_dir)
    logger.info("Data root: %s", base_dir)
    dataset_specific_params = _get_dataset_specific_params(data_config)
    processing_params = set_processing_parameters(
        getattr(data_config, "processing", None)
    )

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
    _reject_empty_validation_selection(valid_ds, training_config, experiment_config)

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
    _check_classification_output_width(model_config, model, train_ds)

    initial_checkpoint = getattr(model_config, "initial_checkpoint", None)
    if initial_checkpoint:
        initial_checkpoint_record = load_initial_model_checkpoint(
            model,
            initial_checkpoint,
            expected_sha256=getattr(
                model_config,
                "initial_checkpoint_sha256",
                None,
            ),
            strict=bool(getattr(model_config, "initial_checkpoint_strict", True)),
        )
        logger.info(
            "Loaded initial model checkpoint %s (sha256=%s, strict=%s)",
            initial_checkpoint_record["path"],
            initial_checkpoint_record["sha256"],
            initial_checkpoint_record["strict"],
        )
        if is_main:
            save_dict(
                initial_checkpoint_record,
                run_save_path,
                "initial_checkpoint_provenance.json",
            )

    return run_save_path, train_ds, valid_ds, test_ds, model, encoder_network
