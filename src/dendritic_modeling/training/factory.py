"""
Training Factory and Utilities

Factory functions for creating training strategies and utility functions
for MLP training and weight transfer.
"""

import logging
from typing import Any, Optional

import torch

from dendritic_modeling.config.multi_stage_training import (
    MultiStageTrainingConfig,
    TrainingStageConfig,
)
from dendritic_modeling.config.training import (
    OptimizerConfig,
    ParamGroupsConfig,
    PruningConfig,
    RegularizationConfig,
    ReportingConfig,
)
from dendritic_modeling.training.strategies import (
    FeedbackAlignmentTrainer,
    HomeostaticControlTrainer,
    LayerWiseTrainer,
    LocalCreditAssignment,
    MultiStageTrainer,
    RecurrentTrainer,
    ShuntingFeedbackAlignmentTrainer,
    Trainer,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
)

logger = logging.getLogger(__name__)


def _build_multi_stage_config(raw_config: Any) -> MultiStageTrainingConfig:
    """Normalize multi-stage config into a typed MultiStageTrainingConfig."""
    if isinstance(raw_config, MultiStageTrainingConfig):
        return raw_config

    if raw_config is None:
        raise ValueError(
            "strategy='multi_stage' requires 'multi_stage_config' in trainer config."
        )

    if hasattr(raw_config, "asdict") and callable(raw_config.asdict):
        raw_config = raw_config.asdict()
    elif hasattr(raw_config, "__dict__") and not isinstance(raw_config, dict):
        raw_config = {
            key: value
            for key, value in vars(raw_config).items()
            if not key.startswith("_")
        }

    if not isinstance(raw_config, dict):
        raise ValueError(
            "multi_stage_config must be a dict or MultiStageTrainingConfig."
        )

    config_dict = dict(raw_config)
    stages_raw = config_dict.get("stages", [])
    if not stages_raw:
        raise ValueError("multi_stage_config must contain at least one stage.")

    stage_configs: list[TrainingStageConfig] = []
    for idx, stage in enumerate(stages_raw):
        if isinstance(stage, TrainingStageConfig):
            stage_configs.append(stage)
            continue
        if hasattr(stage, "asdict") and callable(stage.asdict):
            stage = stage.asdict()
        elif hasattr(stage, "__dict__") and not isinstance(stage, dict):
            stage = {
                key: value
                for key, value in vars(stage).items()
                if not key.startswith("_")
            }
        if not isinstance(stage, dict):
            raise ValueError(
                f"Stage {idx} in multi_stage_config must be a dict or TrainingStageConfig."
            )
        stage_configs.append(TrainingStageConfig(**stage))

    config_dict["stages"] = stage_configs
    return MultiStageTrainingConfig(**config_dict)


def _extract_enhanced_configs(trainer_config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Extract enhanced configuration objects from trainer config dict.

    Args:
        trainer_config_dict: Configuration dictionary

    Returns:
        Dictionary containing enhanced config objects
    """
    enhanced_configs = {}

    # Extract optimizer config
    if "optimizer" in trainer_config_dict and isinstance(
        trainer_config_dict["optimizer"], dict
    ):
        enhanced_configs["optimizer_config"] = OptimizerConfig(
            **trainer_config_dict["optimizer"]
        )
        # Don't pass the dict version to avoid conflicts
        trainer_config_dict.pop("optimizer", None)

    # Extract regularization config
    if "regularization" in trainer_config_dict:
        reg_config = trainer_config_dict.pop("regularization")
        if isinstance(reg_config, dict):
            # Handle nested SelectiveConfig properly
            if "selective" in reg_config and isinstance(reg_config["selective"], dict):
                from dendritic_modeling.config.regularization import SelectiveConfig

                reg_config["selective"] = SelectiveConfig(**reg_config["selective"])
            enhanced_configs["regularization_config"] = RegularizationConfig(
                **reg_config
            )
        else:
            enhanced_configs["regularization_config"] = reg_config
    else:
        # Create default regularization config and check for E/I parameters
        reg_config_dict = {}

        # Check for E/I regularization parameters in trainer config
        ei_params = [
            "enforce_ei_weight_ratio",
            "target_ei_weight_ratio",
            "ei_ratio_loss_weight",
            "ei_ratio_scope",
            "ei_ratio_metric",
        ]
        for param in ei_params:
            if param in trainer_config_dict:
                reg_config_dict[param] = trainer_config_dict.pop(param)

        # Create regularization config with parameters if found
        if reg_config_dict:
            enhanced_configs["regularization_config"] = RegularizationConfig(
                **reg_config_dict
            )

    # Extract pruning config
    if "pruning" in trainer_config_dict:
        prune_config = trainer_config_dict.pop("pruning")
        if isinstance(prune_config, dict):
            # Handle nested SelectiveConfig properly
            if "selective" in prune_config and isinstance(
                prune_config["selective"], dict
            ):
                from dendritic_modeling.config.regularization import SelectiveConfig

                prune_config["selective"] = SelectiveConfig(**prune_config["selective"])
            enhanced_configs["pruning_config"] = PruningConfig(**prune_config)
        else:
            enhanced_configs["pruning_config"] = prune_config

    # Extract reporting config
    if "reporting" in trainer_config_dict:
        report_config = trainer_config_dict.pop("reporting")
        if isinstance(report_config, dict):
            enhanced_configs["reporting_config"] = ReportingConfig(**report_config)
        else:
            enhanced_configs["reporting_config"] = report_config

    # Extract param_groups config (preserve original parameter grouping behavior)
    if "param_groups" in trainer_config_dict:
        param_groups_config = trainer_config_dict.pop("param_groups")
        if isinstance(param_groups_config, dict):
            allowed_fields = set(ParamGroupsConfig.__dataclass_fields__.keys())
            sanitized = {
                key: value
                for key, value in param_groups_config.items()
                if key in allowed_fields
            }
            param_groups_config = ParamGroupsConfig(**sanitized)
        enhanced_configs["param_groups_config"] = param_groups_config

    # Extract strategy-specific nested configs and flatten into kwargs
    # Local credit assignment
    if "local_rule_config" in trainer_config_dict:
        enhanced_configs["local_rule_config"] = trainer_config_dict.pop(
            "local_rule_config"
        )

    # Voltage stabilization
    if "voltage_stabilization_config" in trainer_config_dict:
        vs_cfg = trainer_config_dict.pop("voltage_stabilization_config")
        if isinstance(vs_cfg, dict):
            enhanced_configs.update(vs_cfg)

    # Homeostatic control
    if "homeostatic_control_config" in trainer_config_dict:
        hc_cfg = trainer_config_dict.pop("homeostatic_control_config")
        if isinstance(hc_cfg, dict):
            enhanced_configs.update(hc_cfg)

    # Freeze layers
    if "freeze_layers_config" in trainer_config_dict:
        fl_cfg = trainer_config_dict.pop("freeze_layers_config")
        if isinstance(fl_cfg, dict):
            enhanced_configs.update(fl_cfg)

    # Freeze branches
    if "freeze_branches_config" in trainer_config_dict:
        fb_cfg = trainer_config_dict.pop("freeze_branches_config")
        if isinstance(fb_cfg, dict):
            enhanced_configs.update(fb_cfg)

    # Feedback alignment
    if "fa_config" in trainer_config_dict:
        fa_cfg = trainer_config_dict.pop("fa_config")
        if isinstance(fa_cfg, dict):
            enhanced_configs.update(fa_cfg)

    # Freeze branch with KL
    if "freeze_branch_kl_config" in trainer_config_dict:
        fbkl_cfg = trainer_config_dict.pop("freeze_branch_kl_config")
        if isinstance(fbkl_cfg, dict):
            enhanced_configs.update(fbkl_cfg)

    return enhanced_configs


def get_trainer(
    strategy: str,
    optimizer: torch.optim.Optimizer,
    trainer_config_dict: dict[str, Any],
    analysis_manager: Optional[object] = None,
) -> Trainer:
    """
    Factory function to create training strategies.

    Args:
        strategy: String specifying the training strategy
        optimizer: The optimizer to use
        trainer_config_dict: The trainer configuration dict
        analysis_manager: Optional analysis manager

    Returns:
        An instance of the requested training strategy
    """
    strategy = strategy.lower()
    trainer_config_dict["analysis_manager"] = analysis_manager
    trainer_config_dict["optimizer"] = optimizer

    if strategy == "standard":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)

        return Trainer(**trainer_config_dict)

    elif strategy == "enhanced":
        # Redirect to standard strategy
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return Trainer(**trainer_config_dict)

    elif strategy == "freeze_layers":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return LayerWiseTrainer(**trainer_config_dict)
    elif strategy == "freeze_branches":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return TwoStepTrainer(**trainer_config_dict)
    elif strategy == "fa":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        trainer_config_dict["mode"] = "fa"
        return FeedbackAlignmentTrainer(**trainer_config_dict)
    elif strategy == "dfa":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        trainer_config_dict["mode"] = "dfa"
        return FeedbackAlignmentTrainer(**trainer_config_dict)
    elif strategy == "shunting_fa":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        trainer_config_dict["mode"] = "fa"
        e_rev = trainer_config_dict.pop("e_rev_exc", 1.0)
        return ShuntingFeedbackAlignmentTrainer(**trainer_config_dict, e_rev_exc=e_rev)
    elif strategy == "shunting_dfa":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        trainer_config_dict["mode"] = "dfa"
        e_rev = trainer_config_dict.pop("e_rev_exc", 1.0)
        return ShuntingFeedbackAlignmentTrainer(**trainer_config_dict, e_rev_exc=e_rev)
    elif strategy == "local_ca":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return LocalCreditAssignment(**trainer_config_dict)
    elif strategy == "freeze_branch_kl":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return TwoStepTrainerWithKL(**trainer_config_dict)
    elif strategy == "voltage_stabilization":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return VoltageStabilizationTrainer(**trainer_config_dict)
    elif strategy == "homeostatic_control":
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return HomeostaticControlTrainer(**trainer_config_dict)
    elif strategy == "recurrent":
        task = trainer_config_dict.pop("task", "classification")
        grad_clip = trainer_config_dict.pop(
            "grad_clip", trainer_config_dict.pop("grad_clip_value", 1.0)
        )
        enable_amp = trainer_config_dict.pop(
            "enable_amp", trainer_config_dict.pop("use_amp", False)
        )
        lr_schedule = trainer_config_dict.pop("lr_schedule", "none")
        lr_warmup_epochs = trainer_config_dict.pop("lr_warmup_epochs", 0)
        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)
        return RecurrentTrainer(
            task=task,
            grad_clip=grad_clip,
            enable_amp=enable_amp,
            lr_schedule=lr_schedule,
            lr_warmup_epochs=lr_warmup_epochs,
            **trainer_config_dict,
        )
    elif strategy == "multi_stage":
        raw_multi_stage_config = trainer_config_dict.pop("multi_stage_config", None)
        multi_stage_config = _build_multi_stage_config(raw_multi_stage_config)

        enhanced_configs = _extract_enhanced_configs(trainer_config_dict)
        trainer_config_dict.update(enhanced_configs)

        base_trainer_config = dict(trainer_config_dict)
        return MultiStageTrainer(
            multi_stage_config=multi_stage_config,
            base_optimizer=optimizer,
            base_trainer_config=base_trainer_config,
            analysis_manager=analysis_manager,
        )
    elif strategy == "fsdp" or strategy == "fsdp_standard":
        # FSDP requires additional setup (distributed init, special config)
        from dendritic_modeling.training.fsdp_utils import get_fsdp_config
        from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer

        if "fsdp_config" not in trainer_config_dict:
            # Generate default FSDP config if not provided
            trainer_config_dict["fsdp_config"] = get_fsdp_config()
            logger.info("Generated default FSDP config for trainer")
        return FSDPTrainer(**trainer_config_dict)
    else:
        supported = [
            "standard",
            "enhanced",
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
            "fsdp",
            "fsdp_standard",
        ]
        raise ValueError(
            f"Unknown training strategy '{strategy}'. Supported strategies: {supported}"
        )


def transfer_mlp_weights_to_input_net(mlp_net, input_net, head_type="excitatory"):
    """
    Transfer weights from a trained MLP to the corresponding part of an InputNetTransform.

    Args:
        mlp_net: The trained MLP network
        input_net: The InputNetTransform to transfer weights to
        head_type: Either "excitatory" or "inhibitory" to specify which head to update
    """
    logger.info(
        f"Transferring weights from MLP to {head_type} head of InputNetTransform"
    )

    # Extract the MLP layers
    if hasattr(mlp_net, "layers"):
        mlp_layers = mlp_net.layers
    elif hasattr(mlp_net, "net") and hasattr(mlp_net.net, "layers"):
        mlp_layers = mlp_net.net.layers
    else:
        logger.warning(
            f"Could not extract layers from MLP model. Using random initialization for {head_type} head."
        )
        return

    # Get the shared backbone and target head of the InputNetTransform
    shared_layers = []
    for module in input_net.shared:
        if isinstance(module, torch.nn.Linear):
            shared_layers.append(module)

    target_head = (
        input_net.head_exc if head_type == "excitatory" else input_net.head_inh
    )
    head_layers = []
    for module in target_head:
        if isinstance(module, torch.nn.Linear):
            head_layers.append(module)

    # Transfer weights for the shared backbone (only once, typically from excitatory head)
    if head_type == "excitatory":
        for i, shared_layer in enumerate(shared_layers):
            if (
                i < len(mlp_layers) - 1
            ):  # All but the last MLP layer go to shared backbone
                mlp_layer = mlp_layers[i]
                if hasattr(mlp_layer, "excit_pre_w"):
                    # MLPExcInhLayer has a different structure
                    shared_layer.weight.data = mlp_layer.excit_pre_w.exp().t()
                    shared_layer.bias.data = mlp_layer.bias.data
                else:
                    logger.warning(
                        f"Unexpected MLP layer structure. Using random initialization for shared layer {i}."
                    )

    # Transfer weights for the head-specific part
    if len(head_layers) > 0 and len(mlp_layers) > 0:
        # The last layer of MLP goes to the head
        last_mlp_layer = mlp_layers[-1]
        if hasattr(last_mlp_layer, "excit_pre_w"):
            head_layers[0].weight.data = last_mlp_layer.excit_pre_w.exp().t()
            head_layers[0].bias.data = last_mlp_layer.bias.data
        else:
            logger.warning(
                f"Unexpected MLP layer structure. Using random initialization for {head_type} head."
            )

    logger.info(f"Weight transfer completed for {head_type} head")


def extract_weights_from_simple_mlp(simple_mlp, input_net_transform):
    """
    Extract weights from a SimpleMLP model and transfer them to an InputNetTransform.

    Args:
        simple_mlp: The trained SimpleMLP model
        input_net_transform: The InputNetTransform to transfer weights to
    """
    logger.info("Transferring weights from SimpleMLP to InputNetTransform")

    # Get network modules
    mlp_modules = list(simple_mlp.net.network.children())
    mlp_linear_modules = [m for m in mlp_modules if isinstance(m, torch.nn.Linear)]

    # Get shared backbone modules of InputNetTransform
    shared_modules = []
    for module in input_net_transform.shared:
        if isinstance(module, torch.nn.Linear):
            shared_modules.append(module)

    # Get head modules
    exc_head_modules = []
    for module in input_net_transform.head_exc:
        if isinstance(module, torch.nn.Linear):
            exc_head_modules.append(module)

    inh_head_modules = []
    for module in input_net_transform.head_inh:
        if isinstance(module, torch.nn.Linear):
            inh_head_modules.append(module)

    # Transfer weights for the shared backbone (from early layers of MLP)
    for i, shared_module in enumerate(shared_modules):
        if i < len(mlp_linear_modules) - 1:  # All but the last MLP layer
            mlp_layer = mlp_linear_modules[i]
            shared_module.weight.data = mlp_layer.weight.data.clone()
            shared_module.bias.data = mlp_layer.bias.data.clone()
            logger.info(f"Transferred weights for shared layer {i}")

    # Transfer weights for heads (from last layer of MLP)
    if len(mlp_linear_modules) > 0 and len(exc_head_modules) > 0:
        # Use the second-to-last layer for the heads (since the last layer is the classifier)
        if len(mlp_linear_modules) >= 2:
            source_layer = mlp_linear_modules[-2]
        else:
            source_layer = mlp_linear_modules[0]

        # Transfer to excitatory head
        exc_head_modules[0].weight.data = source_layer.weight.data[
            : exc_head_modules[0].weight.shape[0], :
        ].clone()
        exc_head_modules[0].bias.data = source_layer.bias.data[
            : exc_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to excitatory head")

        # Transfer to inhibitory head
        inh_head_modules[0].weight.data = source_layer.weight.data[
            : inh_head_modules[0].weight.shape[0], :
        ].clone()
        inh_head_modules[0].bias.data = source_layer.bias.data[
            : inh_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to inhibitory head")

    logger.info("Weight transfer from SimpleMLP to InputNetTransform completed")
