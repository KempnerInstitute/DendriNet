"""
Training Factory and Utilities
===============================

This module contains factory functions for creating training strategies
and utility functions for training simple MLPs and weight transfer.
"""

import torch
from typing import Optional

from dendritic_modeling import logger
from dendritic_modeling.training.strategies import (
    Trainer,
    LocalCreditAssignment,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
    LayerWiseTrainer
)

def get_trainer(
    strategy: str,
    optimizer: torch.optim.Optimizer,
    trainer_config_dict: dict,
    analysis_manager: Optional[object] = None
):
    """
    Factory function to create training strategies.

    Args:
        strategy: String specifying the training strategy
        optimizer: The optimizer to use
        loss_function: The loss function to use
        trainer_config: The trainer configuration

    Returns:
        An instance of the requested training strategy
    """
    strategy = strategy.lower()
    trainer_config_dict["optimizer"] = optimizer
    trainer_config_dict["analysis_manager"] = analysis_manager

    if strategy == "standard":
        return Trainer(**trainer_config_dict)
    elif strategy == "freeze_layers":
        return LayerWiseTrainer(**trainer_config_dict)
    elif strategy == "freeze_branches":
        return LocalCreditAssignment(**trainer_config_dict)
    elif strategy == "voltage_stabilization":
        return VoltageStabilizationTrainer(**trainer_config_dict)
    elif strategy == "two_step":
        return TwoStepTrainer(**trainer_config_dict)
    elif strategy == "two_step_kl":
        return TwoStepTrainerWithKL(**trainer_config_dict)
    else:
        logger.info(f"Unrecognized strategy '{strategy}', defaulting to 'standard'.")
        return Trainer(**trainer_config_dict)

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
