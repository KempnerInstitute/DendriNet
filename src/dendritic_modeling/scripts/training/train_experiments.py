#!/usr/bin/env python
"""
train_experiments.py
====================
Main training script for dendritic modeling experiments.

Supports both single-GPU and multi-GPU (DDP) training.  For DDP, launch with
``torchrun``::

    torchrun --nproc_per_node=4 train_experiments.py <config_path>

Single-GPU runs work exactly as before with plain ``python``.

For FSDP (very large models that don't fit on one GPU), use
``train_experiments_fsdp.py`` instead.
"""

from __future__ import annotations

# Suppress TensorFlow warnings for cleaner output
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "2"  # Suppress TensorFlow INFO and WARNING messages
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations warnings

# Core imports
import argparse
import logging
from dataclasses import asdict

import torch
import torch.distributed as dist

# External imports
import torchinfo
import wandb
from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config import load_config
from dendritic_modeling.config.training import OptimizerConfig, ParamGroupsConfig
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    BaseAutoencoder,
)
from dendritic_modeling.plotting.visualizations.performance_plots import (
    PerformancePlotter,
)
from dendritic_modeling.scripts.script_utils import config_utils as _config_utils
from dendritic_modeling.scripts.script_utils.init_diagnostics import (
    dump_init_gate_stats,
)
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.scripts.training.train_encoder_network import (
    load_train_encoder_network,
)
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.training.optimizers import create_optimizer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.training_stability_validator import (
    apply_automatic_safeguards,
    suggest_optimal_config_for_model_size,
    validate_training_stability,
)

logger = logging.getLogger(__name__)

prepare_trainer_config = _config_utils.prepare_trainer_config
normalize_param_groups_config = getattr(
    _config_utils, "normalize_param_groups_config", None
)
canonicalize_model_core_flags = getattr(
    _config_utils, "canonicalize_model_core_flags", None
)
canonicalize_rnn_core_flags = getattr(
    _config_utils, "canonicalize_rnn_core_flags", None
)


if normalize_param_groups_config is None:
    # Backward-compatible fallback for environments where config_utils lacks this helper.
    def normalize_param_groups_config(param_groups):
        if param_groups is None:
            return ParamGroupsConfig()

        if isinstance(param_groups, ParamGroupsConfig):
            return param_groups

        if isinstance(param_groups, DictConfig):
            param_groups = OmegaConf.to_container(param_groups, resolve=True)

        if hasattr(param_groups, "asdict") and callable(param_groups.asdict):
            param_groups = param_groups.asdict()
        elif not isinstance(param_groups, dict) and hasattr(param_groups, "__dict__"):
            param_groups = {
                key: value
                for key, value in vars(param_groups).items()
                if not key.startswith("_")
            }

        if not isinstance(param_groups, dict):
            return ParamGroupsConfig()

        allowed_fields = set(ParamGroupsConfig.__dataclass_fields__.keys())
        sanitized = {k: v for k, v in param_groups.items() if k in allowed_fields}
        return ParamGroupsConfig(**sanitized)


def _maybe_calibrate_data_driven_reactivation(
    config,
    model,
    train_dataset,
    is_main: bool = True,
    n_batches: int = 3,
    batch_size: int = 256,
    n_iterations: int = 3,
):
    """Run data-driven reactivation calibration when requested by init_policy."""
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
        calibrate_reactivation_from_data,
        is_data_driven_reactivation_policy,
        normalize_reactivation_init_policy,
        reactivation_policy_to_calibration_mode,
    )

    config_requested_policies = set()
    layer_requested_policies = set()
    saw_layer_policy = False

    try:
        reactivation_cfg = config.model.core.reactivation
        init_policy = normalize_reactivation_init_policy(
            getattr(reactivation_cfg, "init_policy", "analytical")
        )
        if is_data_driven_reactivation_policy(init_policy):
            config_requested_policies.add(init_policy)
    except AttributeError:
        # The reactivation config path is optional; fall back to collecting
        # any per-layer policies directly from the built modules below.
        pass

    for module in model.modules():
        layer_policy = getattr(module, "reactivation_init_policy", None)
        if layer_policy is None:
            continue
        saw_layer_policy = True
        layer_policy = normalize_reactivation_init_policy(layer_policy)
        if is_data_driven_reactivation_policy(layer_policy):
            layer_requested_policies.add(layer_policy)

    # Once the model is built, module-level policies are authoritative. This is
    # important for recurrent aliases where the factory may override stale YAML
    # policies for one population while preserving data-driven policies for the
    # other population.
    requested_policies = (
        layer_requested_policies if saw_layer_policy else config_requested_policies
    )
    if not requested_policies:
        return

    if saw_layer_policy:
        mode = None
        policy_msg = ", ".join(sorted(requested_policies))
    elif len(requested_policies) == 1:
        init_policy = next(iter(requested_policies))
        mode = reactivation_policy_to_calibration_mode(init_policy)
        policy_msg = init_policy
    else:
        mode = None
        policy_msg = ", ".join(sorted(requested_policies))

    print(f"[reactivation-calibration] init_policy={policy_msg}", flush=True)

    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    # Don't pass a global k — let each layer use its own
    # reactivation_sigma_aware_k (threaded by the factory from config).
    # This respects per-population overrides in unified_ei / recurrent configs.

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    is_recurrent = getattr(model, "_is_recurrent_core", False)
    batches = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            sample = next(it)
        except StopIteration:
            break
        if isinstance(sample, (list, tuple)):
            x = sample[0]
            x = x.to(device) if torch.is_tensor(x) else x
            if is_recurrent and len(sample) > 2:
                seq_lengths = sample[2]
                seq_lengths = (
                    seq_lengths.to(device)
                    if torch.is_tensor(seq_lengths)
                    else seq_lengths
                )
                batches.append((x, seq_lengths))
            else:
                batches.append(x)
        else:
            batches.append(sample.to(device))

    if not batches:
        if is_main:
            logger.warning(
                "reactivation calibration skipped: no training batches available"
            )
        return

    # Recurrent models expect [B, T, D]. Preserve full sequence inputs when the
    # dataset already provides them, and only synthesize a length-1 time axis
    # for flat [B, D] batches.
    if is_recurrent:
        normalized_batches = []
        added_time_axis = False
        for batch in batches:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                rest = list(batch[1:])
            else:
                x = batch
                rest = []

            if torch.is_tensor(x) and x.dim() == 2:
                x = x.unsqueeze(1)
                added_time_axis = True

            normalized_batches.append((x, *rest) if rest else x)

        batches = normalized_batches
        if is_main:
            shape_msg = "[B, 1, D]" if added_time_axis else "native [B, T, D]"
            print(
                f"[reactivation-calibration] recurrent model detected — using {shape_msg} batches",
                flush=True,
            )

    # Iterate: writing new (m, b) at layer i reshapes the V distribution at
    # every downstream layer, so a single-pass calibration is only
    # self-consistent at layer 0. Repeating 2-3 times converges the chain.
    # Median/MAD mode is fairly stable; quantile mode is very sensitive and
    # diverges without this iteration.
    diag = None
    quantile_aggregation = "chunk_median" if is_recurrent else "global"
    for iteration in range(max(1, int(n_iterations))):
        diag = calibrate_reactivation_from_data(
            model,
            batches,
            k=None,
            device=device,
            mode=mode,
            quantile_aggregation=quantile_aggregation,
        )
        if is_main:
            if not diag:
                print(
                    "[reactivation-calibration] found no DendriticBranchLayers with a non-Identity reactivation",
                    flush=True,
                )
                break
            m_values = [d["m"] for d in diag.values()]
            b_values = [d["b"] for d in diag.values()]
            sig_values = [d["V_std"] for d in diag.values()]
            print(
                f"[reactivation-calibration] iter {iteration + 1}/{n_iterations} "
                f"| mode={(mode if mode is not None else 'per-layer')} "
                f"| {len(diag)} layers calibrated "
                f"| m range [{min(m_values):.4f}, {max(m_values):.4f}] "
                f"| b range [{min(b_values):.4f}, {max(b_values):.4f}] "
                f"| sigma range [{min(sig_values):.4f}, {max(sig_values):.4f}] "
                f"| aggregation={quantile_aggregation} "
                f"| k=per-layer n_batches={n_batches}",
                flush=True,
            )
    if is_main and diag:
        for name, d in diag.items():
            extra = ""
            if d.get("occupancy_m_was_clamped"):
                extra += (
                    f" occ[dq_raw={d.get('occupancy_delta_q_raw', float('nan')):.3g}"
                    f" dq_used={d.get('occupancy_delta_q_used', float('nan')):.3g}"
                    f" m_raw={d.get('occupancy_m_raw', float('nan')):.3g} CLAMPED]"
                )
            if d.get("calibration_reverted"):
                extra += " REVERTED"
            print(
                f"[reactivation-calibration]   {name}: "
                f"m={d['m']:.4f} b={d['b']:.4f} "
                f"V[mean={d['V_mean']:.4f} std={d['V_std']:.4f} "
                f"mad_std={d.get('V_mad_std', float('nan')):.4f} "
                f"q10={d.get('V_q10', float('nan')):.3g} "
                f"q90={d.get('V_q90', float('nan')):.3g} "
                f"min={d.get('V_min', float('nan')):.3g} "
                f"max={d.get('V_max', float('nan')):.3g}] n={d['n']}{extra}",
                flush=True,
            )


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training if launched via ``torchrun``.

    ``torchrun`` sets ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``
    automatically.  When these are absent the function returns
    ``(0, 0, 1)`` and does nothing, preserving single-GPU behaviour.

    Returns:
        (rank, local_rank, world_size)
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def normalize_optimizer_config(optimizer_cfg) -> OptimizerConfig:
    """Normalize optimizer config to a typed OptimizerConfig."""
    if optimizer_cfg is None:
        return OptimizerConfig()

    if isinstance(optimizer_cfg, OptimizerConfig):
        return optimizer_cfg

    if isinstance(optimizer_cfg, DictConfig):
        optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)

    if hasattr(optimizer_cfg, "asdict") and callable(optimizer_cfg.asdict):
        optimizer_cfg = optimizer_cfg.asdict()
    elif not isinstance(optimizer_cfg, dict) and hasattr(optimizer_cfg, "__dict__"):
        optimizer_cfg = {
            key: value
            for key, value in vars(optimizer_cfg).items()
            if not key.startswith("_")
        }

    if not isinstance(optimizer_cfg, dict):
        return OptimizerConfig()

    allowed_fields = set(OptimizerConfig.__dataclass_fields__.keys())
    sanitized = {k: v for k, v in optimizer_cfg.items() if k in allowed_fields}
    return OptimizerConfig(**sanitized)


def run_analysis(
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    run_save_path: str,
    analysis_manager: AnalysisManager | None = None,
    training: bool = False,
    training_results: dict | None = None,
    training_config=None,
):
    """Run analysis using the AnalysisManager."""
    logger.info("Running analyses...")
    if analysis_manager is not None:
        # Set pruning results if available
        if training_results and (
            "pre_pruning_performance" in training_results
            or "post_pruning_performance" in training_results
        ):
            analysis_manager.set_pruning_results(training_results)

        analysis_manager.run_analysis(filename="final", training=training)

        # Generate performance evolution plots if this is final analysis and training tracking was enabled
        if not training and analysis_manager.config.performance_analysis.training:
            performance_dir = os.path.join(run_save_path, "performance", "epochs")
            logger.info(
                f"Checking for performance evolution data in: {performance_dir}"
            )
            if os.path.exists(performance_dir):
                logger.info(
                    "Performance directory exists. Generating performance evolution plots..."
                )
                try:
                    # Get final epoch from training config if available
                    final_epoch = None
                    if (
                        training_config is not None
                        and hasattr(training_config, "main")
                        and hasattr(training_config.main, "common")
                        and hasattr(training_config.main.common, "epochs")
                    ):
                        final_epoch = training_config.main.common.epochs

                    # Get pruning results if available
                    pruning_performance = getattr(
                        analysis_manager, "pruning_results", None
                    )
                    logger.info(
                        f"Pruning results available: {pruning_performance is not None}"
                    )

                    plotter = PerformancePlotter()
                    plotter.plot_performance_evolution(
                        performance_dir=performance_dir,
                        save_path=os.path.join(run_save_path, "performance"),
                        pruning_performance=pruning_performance,
                        final_epoch=final_epoch,
                    )
                    logger.info("Performance evolution plots generated successfully")
                except Exception as e:
                    logger.error(f"Failed to generate performance evolution plots: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
            else:
                logger.warning(
                    f"Performance directory does not exist: {performance_dir}"
                )

        logger.info("Analysis completed")


def main(config_path: str, args=None):
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    config = load_config(config_path)
    if is_main:
        logger.info(f"Loaded configuration from {config_path}")
        if world_size > 1:
            logger.info(f"DDP enabled: world_size={world_size}")

    if is_main:
        logger.info("Validating configuration...")
    validation_passed = validate_training_stability(config)
    if not validation_passed:
        logger.error(
            "Configuration validation failed. Please address the errors above."
        )
        raise ValueError("Invalid configuration - critical errors detected")

    apply_automatic_safeguards(config)
    if is_main:
        logger.info("Configuration safeguards applied")

    # Disable wandb on non-rank-0 processes to avoid duplicate logging.
    if not is_main and hasattr(config, "wandb"):
        config.wandb.use_wandb = False

    if args:
        if args.learning_strategy:
            config.training.main.strategy = args.learning_strategy
            logger.info(f"Override learning strategy: {args.learning_strategy}")
        if args.output_dir:
            config.outputs.results_dir = args.output_dir
            logger.info(f"Override output directory: {args.output_dir}")
        if args.run_name:
            config.outputs.run_name = args.run_name
            logger.info(f"Override run name: {args.run_name}")
        if getattr(args, "seed", None) is not None:
            config.experiment.seed = args.seed
            logger.info(f"Override seed: {args.seed}")

    # Extract config sections
    experiment_config = config.experiment
    data_config = config.data
    model_config = config.model
    training_config = config.training
    analysis_config = config.analysis
    wandb_config = config.wandb
    outputs_config = config.outputs

    # In DDP, only rank 0 creates output dirs, log file handlers, and wandb.
    # All ranks call setup_environment to load data + build the model, but
    # non-rank-0 processes suppress side-effects by disabling wandb (already
    # done above) and will adopt the run_save_path broadcast from rank 0.
    run_save_path, train_ds, valid_ds, test_ds, model, encoder_network = (
        setup_environment(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            wandb_config=wandb_config,
            outputs_config=outputs_config,
            experiment_config=experiment_config,
            is_main=is_main,
        )
    )

    # Broadcast run_save_path from rank 0 so all ranks use the same
    # directory (timestamps may differ across ranks).
    if world_size > 1:
        if is_main:
            path_bytes = run_save_path.encode("utf-8")
            path_len = torch.tensor([len(path_bytes)], dtype=torch.long, device="cuda")
        else:
            path_len = torch.tensor([0], dtype=torch.long, device="cuda")
        dist.broadcast(path_len, src=0)
        if is_main:
            path_tensor = torch.tensor(
                list(path_bytes), dtype=torch.uint8, device="cuda"
            )
        else:
            path_tensor = torch.empty(path_len.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(path_tensor, src=0)
        run_save_path = bytes(path_tensor.cpu().tolist()).decode("utf-8")

    # --- DDP wrapping ---
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
        if is_main:
            logger.info("Model wrapped in DistributedDataParallel")

    if is_main:
        logger.info("Network architecture summary before training:")
    # Use the unwrapped model for summary and param group extraction.
    unwrapped_model = model.module if hasattr(model, "module") else model

    # Post-build gate diagnostic: record per-layer (m, b) statistics to
    # init_gate_stats.json in the run directory. This is a generic health
    # check for runs that use learnable reactivation gates, especially when
    # data-driven calibration is enabled.
    if is_main:
        try:
            dump_init_gate_stats(unwrapped_model, run_save_path)
        except Exception as exc:  # diagnostic must not break training
            logger.warning("init_gate_stats diagnostic failed: %s", exc)
    if is_main:
        try:
            is_recurrent_core = getattr(
                unwrapped_model.core_network, "is_recurrent", False
            )
            if callable(is_recurrent_core):
                is_recurrent_core = is_recurrent_core()

            if is_recurrent_core:
                strategy = getattr(
                    getattr(config.training, "main", None), "strategy", "standard"
                )
                if strategy not in ("recurrent", "multi_stage"):
                    logger.warning(
                        "Recurrent model detected but training.main.strategy='%s'. "
                        "Use strategy='recurrent' for correct sequence handling.",
                        strategy,
                    )
                total_params = sum(p.numel() for p in unwrapped_model.parameters())
                trainable_params = sum(
                    p.numel() for p in unwrapped_model.parameters() if p.requires_grad
                )
                logger.info(
                    "Skipping torchinfo summary for recurrent model. "
                    f"total_params={total_params}, trainable_params={trainable_params}"
                )
                if wandb.run:
                    wandb.log(
                        {
                            "total_params": total_params,
                            "trainable_params": trainable_params,
                        }
                    )
            else:
                sample_input: torch.Tensor = train_ds[0][0]
                summary = torchinfo.summary(
                    unwrapped_model,
                    input_size=(100, *sample_input.shape),
                    col_names=["input_size", "output_size", "num_params"],
                    col_width=25,
                    row_settings=["depth"],
                    depth=7,
                )
                logger.info("Model summary generated (see console output)")
                if wandb.run:
                    wandb.log(
                        {
                            "total_params": summary.total_params,
                            "trainable_params": summary.trainable_params,
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to generate model summary before training: {e}")

    if is_main:
        config_payload = asdict(config)
        if canonicalize_model_core_flags is not None:
            config_payload = canonicalize_model_core_flags(config_payload)
        if canonicalize_rnn_core_flags is not None:
            config_payload = canonicalize_rnn_core_flags(config_payload)
        save_dict(config_payload, run_save_path, "config.json")

    # Under DDP, autoencoder pretraining must be serialized: rank 0
    # trains and saves the checkpoint, other ranks wait then load it.
    # Identity and non-autoencoder encoders are safe on all ranks.
    if world_size > 1 and isinstance(encoder_network, BaseAutoencoder):
        if is_main:
            encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds = (
                load_train_encoder_network(
                    encoder_network=encoder_network,
                    encoder_train_config=training_config.encoder,
                    train_ds=train_ds,
                    valid_ds=valid_ds,
                    test_ds=test_ds,
                    encoder_network_config=model_config.encoder,
                    task_config=data_config,
                )
            )
        dist.barrier()  # wait for rank 0 to finish training/saving
        if not is_main:
            # Rank 0 has saved a checkpoint; non-rank-0 will find it and
            # skip training, just load + encode.
            encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds = (
                load_train_encoder_network(
                    encoder_network=encoder_network,
                    encoder_train_config=training_config.encoder,
                    train_ds=train_ds,
                    valid_ds=valid_ds,
                    test_ds=test_ds,
                    encoder_network_config=model_config.encoder,
                    task_config=data_config,
                )
            )
    else:
        encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds = (
            load_train_encoder_network(
                encoder_network=encoder_network,
                encoder_train_config=training_config.encoder,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                encoder_network_config=model_config.encoder,
                task_config=data_config,
            )
        )

    encoded_data = {
        "train": encoded_train_ds,
        "valid": encoded_valid_ds,
        "test": encoded_test_ds,
    }
    analysis_manager = AnalysisManager(
        model=unwrapped_model,
        data=encoded_data,
        analysis_config=analysis_config,
        save_root=run_save_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Replace encoder on the unwrapped model so the change is visible
    # through the DDP wrapper (which delegates to model.module).
    if isinstance(encoder_network, Identity):
        unwrapped_model.encoder_network = encoder_network
    elif hasattr(encoder_network, "encoder"):
        unwrapped_model.encoder_network = Identity(encoder_network.encoder.output_dim)
    else:
        unwrapped_model.encoder_network = encoder_network

    # Optional post-hoc empirical calibration of the reactivation (m, b).
    # Triggered when model.core.reactivation.init_policy == "empirical".
    # Runs a handful of training batches through the model, measures the
    # per-layer voltage distribution before the reactivation, and sets
    # b = E[V], m = k / std(V).
    _maybe_calibrate_data_driven_reactivation(
        config=config,
        model=unwrapped_model,
        train_dataset=encoded_train_ds,
        is_main=is_main,
    )

    main_net_save_path = os.path.join(run_save_path, "main_network")
    os.makedirs(main_net_save_path, exist_ok=True)

    # Get main training config
    main_train_config = training_config.main

    # Prepare trainer configuration
    main_trainer_config_dict = prepare_trainer_config(
        main_train_config=main_train_config,
        experiment_config=experiment_config,
        save_path=main_net_save_path,
    )
    main_trainer_config_dict["seed"] = int(getattr(experiment_config, "seed", 0) or 0)

    # Extract config values (handle both dict from sweeps and object from single runs)
    if isinstance(main_train_config, dict):
        common_config = main_train_config.get("common", {})
        strategy = main_train_config.get("strategy", "standard")
        optimizer_config = normalize_optimizer_config(
            main_train_config.get("optimizer")
        )
    else:
        common_config = main_train_config.common
        strategy = main_train_config.strategy
        optimizer_config = normalize_optimizer_config(
            getattr(main_train_config, "optimizer", None)
        )

    # Extract from common_config (also handle dict vs object)
    if isinstance(common_config, dict):
        param_groups = common_config.get("param_groups", {})
        weight_decay_rate = common_config.get("weight_decay_rate", 0.0)
        weight_boosting = common_config.get("weight_boosting", False)
        common_epochs = common_config.get("epochs", 100)
    else:
        param_groups = common_config.param_groups
        weight_decay_rate = common_config.weight_decay_rate
        weight_boosting = getattr(common_config, "weight_boosting", False)
        common_epochs = common_config.epochs

    param_groups = normalize_param_groups_config(param_groups)
    model_param_groups = unwrapped_model.get_param_groups(param_groups)
    base_optimizer = create_optimizer(model_param_groups, optimizer_config)

    optimizer = CustomWeightDecayOptimizer(
        model=unwrapped_model,
        optimizer=base_optimizer,
        weight_decay=weight_decay_rate,
        weight_boosting=weight_boosting,
    )

    original_pruning_config = main_trainer_config_dict.get("pruning", {})
    main_trainer_config_dict["pruning"] = {}

    # Check if using multi-stage training
    if strategy == "multi_stage":
        if "multi_stage_config" not in main_trainer_config_dict:
            raise ValueError(
                "Learning strategy is 'multi_stage' but no 'multi_stage_config' found"
            )
        logger.info("Using multi-stage training strategy")

    # Inject task from model config for recurrent trainer
    main_trainer_config_dict["task"] = model_config.task

    main_trainer = get_trainer(
        strategy=strategy,
        optimizer=optimizer,
        trainer_config_dict=main_trainer_config_dict,
        analysis_manager=analysis_manager,
    )

    training_results = main_trainer.train(
        model=model, train_data=encoded_train_ds, valid_data=encoded_valid_ds
    )

    if is_main:
        nparams = sum(p.numel() for p in unwrapped_model.parameters())
        if wandb.run is not None:
            wandb.log({"nparams": nparams})
        save_dict({"nparams": nparams}, run_save_path, "nparams")

        suggestions = suggest_optimal_config_for_model_size(nparams)
        if suggestions:
            logger.info(
                f"=== MODEL SIZE RECOMMENDATIONS (Model has {nparams:,} parameters) ==="
            )
            for key, value in suggestions.items():
                current_value = getattr(config.training, key, "Not available")
                if current_value != value:
                    logger.info(
                        f"  Consider: {key} = {value} (current: {current_value})"
                    )
            logger.info("=" * 50)

    # Run analysis if enabled (rank 0 only)
    if is_main:
        any_analysis_enabled = any(
            [
                analysis_config.performance_analysis.enabled,
                analysis_config.information_analysis.enabled,
                analysis_config.synapse_turnover_analysis.enabled,
                analysis_config.noise_perturbation_analysis.enabled,
                analysis_config.single_layer_contribution.enabled,
                analysis_config.ablation_analysis.enabled,
                analysis_config.weight_analysis.enabled,
                analysis_config.synaptic_activation_analysis.enabled,
                analysis_config.branch_activation_analysis.enabled,
                hasattr(analysis_config, "correlation_analysis")
                and analysis_config.correlation_analysis.enabled,
                hasattr(analysis_config, "routing_analysis")
                and analysis_config.routing_analysis.enabled,
            ]
        )

        if strategy == "multi_stage":
            if analysis_config.performance_analysis.training:
                performance_dir = os.path.join(run_save_path, "performance", "epochs")
                if os.path.exists(performance_dir):
                    try:
                        multi_stage = main_train_config.strategies.multi_stage
                        stages = multi_stage.stages
                        total_epochs = sum(
                            getattr(stage, "epochs", common_epochs) for stage in stages
                        )
                        plotter = PerformancePlotter()
                        plotter.plot_performance_evolution(
                            performance_dir=performance_dir,
                            save_path=os.path.join(run_save_path, "performance"),
                            pruning_performance=getattr(
                                analysis_manager, "pruning_results", None
                            ),
                            final_epoch=total_epochs,
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate performance plots: {e}")

    # Now apply pruning if enabled (separate step)
    if original_pruning_config and original_pruning_config.get("enabled", False):
        if is_main:
            logger.info("Applying pruning as separate step...")
        main_trainer_config_dict["pruning"] = original_pruning_config
        main_trainer_config_dict["reporting"] = main_trainer_config_dict.get(
            "reporting", {}
        )

        pruning_trainer = get_trainer(
            strategy=strategy,
            optimizer=optimizer,
            trainer_config_dict=main_trainer_config_dict,
            analysis_manager=analysis_manager,
        )

        pruning_trainer._initialize_attributes(model, common_epochs)

        pruning_results = pruning_trainer._apply_pruning_and_evaluate(
            model=model, train_data=encoded_train_ds, valid_data=encoded_valid_ds
        )
        training_results.update(pruning_results)
        if is_main:
            logger.info("Pruning completed")

    # Save final model (rank 0 only).  Encoder was already swapped
    # to Identity before training started (see above).
    if is_main:
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(run_save_path, "final_model.pt"),
        )

        if any_analysis_enabled and strategy != "multi_stage":
            run_analysis(
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                run_save_path=run_save_path,
                analysis_manager=analysis_manager,
                training_config=training_config,
                training=False,
                training_results=training_results,
            )

    if (
        hasattr(wandb_config, "use_wandb")
        and wandb_config.use_wandb
        and wandb.run is not None
    ):
        wandb.finish()

    # Clean up distributed process group.
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

    if is_main:
        logger.info("Training experiment completed successfully")


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run dendritic modeling training and evaluation."
    )
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument(
        "--output_dir", help="Override output directory specified in config."
    )
    parser.add_argument("--run_name", help="Override run name specified in config.")
    parser.add_argument(
        "--learning_strategy",
        choices=[
            "standard",
            "freeze_layers",
            "freeze_branches",
            "fa",
            "dfa",
            "shunting_fa",
            "shunting_dfa",
            "local_ca",
            "freeze_branch_kl",
            "voltage_stabilization",
            "homeostatic_control",
            "recurrent",
            "multi_stage",
        ],
        help="Override learning strategy specified in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed (for running multiple seeds per config).",
    )

    return parser.parse_args()


def cli_main():
    """Console-script entry point for ``dendritic-train``."""
    args = _parse_cli_args()
    main(config_path=args.config, args=args)


if __name__ == "__main__":
    cli_main()
