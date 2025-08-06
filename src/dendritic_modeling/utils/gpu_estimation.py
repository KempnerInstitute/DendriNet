#!/usr/bin/env python3
"""
GPU Memory Estimation for Dendritic Models.

This module provides comprehensive GPU memory estimation for dendritic neural networks,
including support for FSDP distributed training.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from dendritic_modeling import logger


@dataclass
class MemoryBreakdown:
    """Memory breakdown for model training."""

    parameters: float  # GB
    gradients: float  # GB
    optimizer_states: float  # GB
    activations: float  # GB
    fsdp_overhead: float  # GB
    total: float  # GB

    def __str__(self):
        lines = [
            "Memory Breakdown:",
            f"  Parameters:        {self.parameters:.2f} GB",
            f"  Gradients:         {self.gradients:.2f} GB",
            f"  Optimizer states:  {self.optimizer_states:.2f} GB",
            f"  Activations:       {self.activations:.2f} GB",
            f"  FSDP overhead:     {self.fsdp_overhead:.2f} GB",
            f"  Total:             {self.total:.2f} GB",
        ]
        return "\n".join(lines)


@dataclass
class GPURecommendation:
    """GPU recommendations for training."""

    gpu_model: str
    gpu_memory_gb: int
    min_gpus_needed: int
    recommended_gpus: int  # Power of 2 for efficiency
    memory_per_gpu_gb: float
    utilization_percent: float
    fits: bool

    def __str__(self):
        status = "[FITS]" if self.fits else "[NO FIT]"
        return (
            f"{status} {self.gpu_model} ({self.gpu_memory_gb}GB): "
            f"min {self.min_gpus_needed} GPUs, "
            f"recommended {self.recommended_gpus} GPUs, "
            f"{self.memory_per_gpu_gb:.2f}GB/GPU ({self.utilization_percent:.1f}% utilization)"
        )


class GPUMemoryEstimator:
    """Estimate GPU memory requirements for dendritic models."""

    GPU_SPECS = {
        "A100-40GB": {"memory_gb": 40, "compute": "high"},
        "H100": {"memory_gb": 80, "compute": "high"},
    }

    def __init__(self, dtype: str = "fp16", optimizer: str = "adam"):
        """
        Initialize the estimator.

        Args:
            dtype: Data type ('fp16', 'fp32', 'bf16')
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
        """
        self.dtype = dtype
        self.optimizer = optimizer
        self.bytes_per_param = self._get_bytes_per_param(dtype)
        self.optimizer_state_multiplier = self._get_optimizer_multiplier(optimizer)

    def _get_bytes_per_param(self, dtype: str) -> int:
        """Get bytes per parameter for given dtype."""
        dtype_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
        }
        return dtype_bytes.get(dtype, 2)

    def _get_optimizer_multiplier(self, optimizer: str) -> float:
        """Get memory multiplier for optimizer states."""
        optimizer_multipliers = {
            "sgd": 1.0,  # Only momentum
            "adam": 2.0,  # Momentum + variance
            "adamw": 2.0,  # Same as Adam
            "lamb": 2.0,  # Similar to Adam
            "rmsprop": 1.0,  # Running average
        }
        return optimizer_multipliers.get(optimizer.lower(), 2.0)

    def estimate_model_parameters(self, model: torch.nn.Module) -> dict[str, int]:
        """
        Count parameters by layer type.

        Args:
            model: PyTorch model

        Returns:
            Dictionary mapping layer types to parameter counts
        """
        param_counts = {}
        total_params = 0

        for _name, module in model.named_modules():
            # Skip compound modules
            if len(list(module.children())) > 0:
                continue

            # Count parameters for this module
            module_params = sum(p.numel() for p in module.parameters())
            if module_params == 0:
                continue

            # Categorize by module type
            module_type = type(module).__name__

            # Special handling for dendritic layers
            if "TopK" in module_type:
                category = "TopK Layers"
            elif "BlockLinear" in module_type:
                category = "Block Linear Layers"
            elif "Linear" in module_type:
                category = "Linear Layers"
            elif "Conv" in module_type:
                category = "Convolutional Layers"
            elif "Reactivation" in module_type or "ParametricTanh" in module_type:
                category = "Reactivation Layers"
            elif "BatchNorm" in module_type or "LayerNorm" in module_type:
                category = "Normalization Layers"
            else:
                category = "Other Layers"

            param_counts[category] = param_counts.get(category, 0) + module_params
            total_params += module_params

        param_counts["Total"] = total_params
        return param_counts

    def estimate_activation_memory(
        self, model: torch.nn.Module, batch_size: int, input_shape: tuple[int, ...]
    ) -> float:
        """
        Estimate activation memory for dendritic networks.

        Args:
            model: PyTorch model
            batch_size: Batch size
            input_shape: Input shape (excluding batch dimension)

        Returns:
            Activation memory in GB
        """
        activation_memory_bytes = 0

        # For dendritic networks (EINet), we need to analyze the structure more carefully
        # The network has multiple layers with branch factors creating massive intermediate activations

        # First, check if this is an EINet model
        is_einet = False
        einet_config = None

        for _name, module in model.named_modules():
            if "ExcitationInhibitionNetwork" in type(module).__name__:
                is_einet = True
                # Extract configuration from the network
                if hasattr(module, "excitatory_layer_sizes"):
                    einet_config = {
                        "e_layers": module.excitatory_layer_sizes,
                        "i_layers": module.inhibitory_layer_sizes,
                        "e_branches": module.excitatory_branch_factors,
                        "i_branches": module.inhibitory_branch_factors,
                        "ee_synapses": module.ee_synapses_per_branch_per_layer,
                        "ei_synapses": module.ei_synapses_per_branch_per_layer,
                        "ie_synapses": module.ie_synapses_per_branch_per_layer,
                        "ii_synapses": module.ii_synapses_per_branch_per_layer,
                    }
                break

        if is_einet and einet_config:
            # Calculate activation memory for EINet architecture
            input_dim = np.prod(input_shape)

            # Transfer layer activations
            activation_memory_bytes += batch_size * input_dim * self.bytes_per_param

            # Calculate branch factors product for each layer
            e_branch_product = 1
            for bf in einet_config["e_branches"]:
                e_branch_product *= bf

            i_branch_product = 1
            if einet_config["i_branches"]:
                for bf in einet_config["i_branches"]:
                    i_branch_product *= bf

            # For each excitatory layer
            for i, layer_size in enumerate(einet_config["e_layers"]):
                # Dendritic branch activations
                # Each neuron has branch_product branches
                branch_activations = (
                    batch_size * layer_size * e_branch_product * self.bytes_per_param
                )
                activation_memory_bytes += branch_activations

                # TopK connectivity activations (sparse weight indices and values)
                if i < len(einet_config["ee_synapses"]):
                    synapses = einet_config["ee_synapses"][i]
                    # Each branch has K synapses, storing indices and values
                    topk_memory = (
                        batch_size
                        * layer_size
                        * e_branch_product
                        * synapses
                        * self.bytes_per_param
                        * 2
                    )
                    activation_memory_bytes += topk_memory

            # For each inhibitory layer (if present)
            if einet_config["i_layers"]:
                for i, layer_size in enumerate(einet_config["i_layers"]):
                    if layer_size:  # Skip None entries
                        # Similar calculation for inhibitory neurons
                        branch_activations = (
                            batch_size
                            * layer_size
                            * i_branch_product
                            * self.bytes_per_param
                        )
                        activation_memory_bytes += branch_activations

                        # TopK connectivity for inhibitory neurons
                        if i < len(einet_config["ii_synapses"]):
                            synapses = einet_config["ii_synapses"][i]
                            topk_memory = (
                                batch_size
                                * layer_size
                                * i_branch_product
                                * synapses
                                * self.bytes_per_param
                                * 2
                            )
                            activation_memory_bytes += topk_memory

            # Add substantial overhead for gradient computation graphs in dendritic networks
            # Dendritic networks have complex backward passes with branch-wise operations
            activation_memory_bytes *= (
                10.0  # Increased for dendritic branching overhead
            )

        else:
            # Fallback: Analyze layer by layer
            for _name, module in model.named_modules():
                # Skip compound modules
                if len(list(module.children())) > 0:
                    continue

                # TopK layers - these are the main memory consumers in dendritic networks
                if "TopK" in type(module).__name__ or hasattr(module, "K"):
                    in_features = getattr(module, "in_features", 0)
                    out_features = getattr(module, "out_features", 0)
                    K = getattr(module, "K", 100)  # Number of synapses per output

                    # Memory for sparse connectivity pattern and gradients
                    # Store: indices (K ints per output) + values (K floats per output) + gradients
                    layer_activation = (
                        batch_size * out_features * K * self.bytes_per_param * 4
                    )
                    activation_memory_bytes += layer_activation

                elif "BlockLinear" in type(module).__name__:
                    # BlockLinear combines branches
                    if hasattr(module, "weight"):
                        weight_shape = module.weight.shape
                        layer_activation = (
                            batch_size
                            * np.prod(weight_shape)
                            * self.bytes_per_param
                            * 2
                        )
                        activation_memory_bytes += layer_activation

                elif isinstance(module, torch.nn.Linear):
                    in_features = getattr(module, "in_features", 0)
                    out_features = getattr(module, "out_features", 0)
                    layer_activation = (
                        batch_size
                        * (in_features + out_features)
                        * self.bytes_per_param
                        * 2
                    )
                    activation_memory_bytes += layer_activation

            # If we couldn't get a good estimate, use parameter-based estimation
            if activation_memory_bytes == 0:
                total_params = sum(p.numel() for p in model.parameters())
                # For dendritic models, activations can be 10-20x parameters due to branching
                activation_memory_bytes = (
                    total_params * self.bytes_per_param * 60
                )  # Multiplier for dendritic models

            # Add gradient graph overhead
            activation_memory_bytes *= 2.5

        return activation_memory_bytes / (1024**3)  # Convert to GB

    def estimate_memory_breakdown(
        self,
        model: torch.nn.Module,
        batch_size: int,
        input_shape: tuple[int, ...],
        use_fsdp: bool = False,
        num_gpus: int = 1,
    ) -> MemoryBreakdown:
        """
        Estimate complete memory breakdown for training.

        Args:
            model: PyTorch model
            batch_size: Batch size
            input_shape: Input shape (excluding batch dimension)
            use_fsdp: Whether using FSDP
            num_gpus: Number of GPUs for FSDP

        Returns:
            MemoryBreakdown object
        """
        # Count parameters
        param_counts = self.estimate_model_parameters(model)
        total_params = param_counts["Total"]

        # Parameter memory
        param_memory_gb = (total_params * self.bytes_per_param) / (1024**3)

        # Gradient memory (same as parameters)
        gradient_memory_gb = param_memory_gb

        # Optimizer state memory
        optimizer_memory_gb = param_memory_gb * self.optimizer_state_multiplier

        # Activation memory
        activation_memory_gb = self.estimate_activation_memory(
            model, batch_size, input_shape
        )

        # FSDP overhead
        fsdp_overhead_gb = 0.0
        if use_fsdp:
            # FSDP adds communication buffers and sharding metadata
            # Approximately 10-20% overhead
            base_memory = param_memory_gb + gradient_memory_gb + optimizer_memory_gb
            fsdp_overhead_gb = (
                base_memory * 0.5
            )  # Increased overhead for communication/sharding

            # With FSDP, memory is sharded across GPUs
            param_memory_gb /= num_gpus
            gradient_memory_gb /= num_gpus
            optimizer_memory_gb /= num_gpus
            # Activations are not sharded (each GPU computes full forward pass)
            # But FSDP can use gradient checkpointing to reduce activation memory
            activation_memory_gb *= 0.7  # Assume some gradient checkpointing

        # Total memory
        total_memory_gb = (
            param_memory_gb
            + gradient_memory_gb
            + optimizer_memory_gb
            + activation_memory_gb
            + fsdp_overhead_gb
        )

        return MemoryBreakdown(
            parameters=param_memory_gb,
            gradients=gradient_memory_gb,
            optimizer_states=optimizer_memory_gb,
            activations=activation_memory_gb,
            fsdp_overhead=fsdp_overhead_gb,
            total=total_memory_gb,
        )

    def get_gpu_recommendations(
        self, memory_breakdown: MemoryBreakdown, safety_factor: float = 0.85
    ) -> list[GPURecommendation]:
        """
        Get GPU recommendations based on memory requirements.

        Args:
            memory_breakdown: Memory breakdown from estimate_memory_breakdown
            safety_factor: Safety factor for memory (0.85 = use 85% of GPU memory)

        Returns:
            List of GPU recommendations sorted by efficiency
        """
        recommendations = []

        for gpu_name, gpu_spec in self.GPU_SPECS.items():
            gpu_memory = gpu_spec["memory_gb"]
            usable_memory = gpu_memory * safety_factor

            # Single GPU scenario
            single_gpu_fits = memory_breakdown.total <= usable_memory

            # Multi-GPU scenario (with FSDP)
            if usable_memory <= 0 or memory_breakdown.total <= 0:
                print(
                    f" Invalid memory values: total={memory_breakdown.total:.2f}GB, usable={usable_memory:.2f}GB"
                )
                min_gpus_needed = 1  # Default to 1 GPU if calculation fails
            else:
                min_gpus_needed = math.ceil(memory_breakdown.total / usable_memory)

            # Recommend power of 2 for communication efficiency
            recommended_gpus = 1
            while recommended_gpus < min_gpus_needed:
                recommended_gpus *= 2

            # Memory per GPU with FSDP
            if recommended_gpus > 0:
                memory_per_gpu = memory_breakdown.total / recommended_gpus
                utilization = (
                    (memory_per_gpu / gpu_memory) * 100 if gpu_memory > 0 else 0
                )
            else:
                memory_per_gpu = memory_breakdown.total
                utilization = 0

            recommendation = GPURecommendation(
                gpu_model=gpu_name,
                gpu_memory_gb=gpu_memory,
                min_gpus_needed=min_gpus_needed,
                recommended_gpus=recommended_gpus,
                memory_per_gpu_gb=memory_per_gpu,
                utilization_percent=utilization,
                fits=single_gpu_fits or recommended_gpus <= 16,  # Practical limit
            )

            recommendations.append(recommendation)

        recommendations.sort(key=lambda r: (r.recommended_gpus, -r.utilization_percent))

        return recommendations

    def get_optimal_gpu_config(self, single_gpu_breakdown, h100_memory, a100_memory):
        """
        Determine optimal GPU configuration for FSDP training.

        Returns:
            dict: Optimal configuration with nodes, gpus_per_node, total_gpus, etc.
        """

        if single_gpu_breakdown.total <= h100_memory:
            return {
                "single_gpu_fits": True,
                "gpu_type": "H100",
                "nodes": 1,
                "gpus_per_node": 1,
                "total_gpus": 1,
                "memory_per_gpu": single_gpu_breakdown.total,
            }
        elif single_gpu_breakdown.total <= a100_memory:
            return {
                "single_gpu_fits": True,
                "gpu_type": "A100-40GB",
                "nodes": 1,
                "gpus_per_node": 1,
                "total_gpus": 1,
                "memory_per_gpu": single_gpu_breakdown.total,
            }

        configurations = [
            # (total_gpus, nodes, gpus_per_node)
            (2, 1, 2),  # 2 GPUs, 1 node
            (4, 1, 4),  # 4 GPUs, 1 node
            (8, 2, 4),  # 8 GPUs, 2 nodes
            (16, 4, 4),  # 16 GPUs, 4 nodes
            (32, 8, 4),  # 32 GPUs, 8 nodes
        ]

        for total_gpus, nodes, gpus_per_node in configurations:

            param_memory = single_gpu_breakdown.parameters / total_gpus
            grad_memory = single_gpu_breakdown.gradients / total_gpus
            optimizer_memory = single_gpu_breakdown.optimizer_states / total_gpus
            activation_memory = single_gpu_breakdown.activations * 0.8
            fsdp_overhead = 0.1 * (param_memory + grad_memory)

            total_memory = (
                param_memory
                + grad_memory
                + optimizer_memory
                + activation_memory
                + fsdp_overhead
            )

            if total_memory <= a100_memory:
                return {
                    "single_gpu_fits": False,
                    "gpu_type": "A100-40GB",
                    "nodes": nodes,
                    "gpus_per_node": gpus_per_node,
                    "total_gpus": total_gpus,
                    "memory_per_gpu": total_memory,
                }

            elif total_memory <= h100_memory:
                return {
                    "single_gpu_fits": False,
                    "gpu_type": "H100",
                    "nodes": nodes,
                    "gpus_per_node": gpus_per_node,
                    "total_gpus": total_gpus,
                    "memory_per_gpu": total_memory,
                }

        return {
            "single_gpu_fits": False,
            "gpu_type": "H100",
            "nodes": 8,
            "gpus_per_node": 4,
            "total_gpus": 32,
            "memory_per_gpu": total_memory,
            "warning": "Model may be too large - consider optimization",
        }

    def print_estimation_report(
        self,
        model: torch.nn.Module,
        batch_size: int,
        input_shape: tuple[int, ...],
        config: Optional[Any] = None,
    ):
        """
        Print a simplified GPU memory estimation report.

        Args:
            model: PyTorch model
            batch_size: Batch size
            input_shape: Input shape (excluding batch dimension)
            config: Optional configuration object
        """
        # Get total parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Single GPU estimation
        single_gpu_breakdown = self.estimate_memory_breakdown(
            model, batch_size, input_shape, use_fsdp=False
        )

        logger.info("=" * 80)
        logger.info("GPU MEMORY ESTIMATION")
        logger.info("=" * 80)
        logger.info(f"Model: {total_params:,} parameters")
        logger.info(f"Single GPU memory: {single_gpu_breakdown.total:.2f} GB")

        # Test 4-GPU FSDP (common setup)
        fsdp_4gpu = self.estimate_memory_breakdown(
            model, batch_size, input_shape, use_fsdp=True, num_gpus=4
        )
        logger.info(f"4-GPU FSDP memory per GPU: {fsdp_4gpu.total:.2f} GB")

        a100_memory = self.GPU_SPECS["A100-40GB"]["memory_gb"] * 0.85
        h100_memory = self.GPU_SPECS["H100"]["memory_gb"] * 0.85

        if single_gpu_breakdown.total <= a100_memory:
            logger.info("[FITS] Fits on single A100-40GB (40GB)")
        elif single_gpu_breakdown.total <= h100_memory:
            logger.info("[FITS] Fits on single H100 (80GB)")
        else:
            logger.info("[NO FIT] Requires multi-GPU setup")
        optimal_config = self.get_optimal_gpu_config(
            single_gpu_breakdown, h100_memory, a100_memory
        )

        if optimal_config["single_gpu_fits"]:
            logger.info(f"[FITS] Single {optimal_config['gpu_type']} sufficient")
        else:
            logger.info(
                f"[RECOMMENDED] {optimal_config['total_gpus']}x {optimal_config['gpu_type']} FSDP"
            )
            logger.info(
                f"[CONFIG] {optimal_config['nodes']} nodes × {optimal_config['gpus_per_node']} GPUs/node"
            )

        logger.info("=" * 80)

        return optimal_config


def estimate_from_config(config_path: str, batch_size: Optional[int] = None):
    """
    Estimate GPU requirements from a configuration file.
    Uses real dataset samples for accurate shape inference, similar to train_experiments.py.

    Args:
        config_path: Path to configuration file
        batch_size: Override batch size (uses config value if None)
    """
    from dendritic_modeling.config import load_config
    from dendritic_modeling.datasets import get_unified_datasets
    from dendritic_modeling.models import Classifier, Regressor
    from dendritic_modeling.networks import Identity
    from dendritic_modeling.networks.architectures.factory import \
        get_architecture

    # Load configuration
    config = load_config(config_path)

    # Get batch size
    if batch_size is None:
        trainer_params = (
            config.train.main.trainer.__dict__
            if hasattr(config.train.main, "trainer")
            else {}
        )
        batch_size = trainer_params.get("batch_size", 32)

    # Initialize task_params for later use
    task_params = config.task.parameters or {}

    # Load real dataset to get accurate input shape (like train_experiments.py)
    try:
        # Get datasets using the same method as training scripts
        train_ds, valid_ds, test_ds = get_unified_datasets(
            task_cfg=config.task,
            train_cfg=config.train,
            output_dir=config.outputs.dir,
        )

        # Get sample input shape from real data
        sample_input = train_ds[0][0]  # First sample, input only
        input_shape = sample_input.shape  # This is the real shape from dataset

        print(f"Using real dataset input shape: {input_shape}")

    except Exception as e:
        print(f"Failed to load dataset, falling back to hardcoded shapes: {e}")

        # Fallback to hardcoded input dimensions if dataset loading fails
        if config.task.dataset == "mnist":
            input_shape = (784,)  # Flattened
        elif config.task.dataset == "cifar10":
            input_shape = (3072,)  # Flattened
        elif config.task.dataset == "imagenet":
            if task_params.get("flatten", False):
                input_shape = (150528,)  # Flattened
            else:
                input_shape = (3, 224, 224)
        else:
            input_dim = task_params.get("input_dim", 784)
            input_shape = (input_dim,)

    # Create model (similar to train scripts)
    model_config = config.model

    # Create encoder
    encoder_config = model_config.encoder_network
    if encoder_config.type.lower() == "identity":
        encoder = Identity(
            input_shape[0] if len(input_shape) == 1 else np.prod(input_shape)
        )
    else:
        encoder_params = (
            dict(encoder_config.parameters) if encoder_config.parameters else {}
        )
        encoder_params["input_dim"] = (
            input_shape[0] if len(input_shape) == 1 else np.prod(input_shape)
        )
        encoder = get_architecture(encoder_config.type, encoder_params)

    # Create core network
    core_config = model_config.core_network
    core_params = dict(core_config.parameters) if core_config.parameters else {}
    core_params["input_dim"] = encoder.output_dim
    core_network = get_architecture(core_config.type, core_params)

    # Get task type and number of classes
    task_type = model_config.task  # task is on model config, not task config
    if task_type == "classification":
        n_classes_map = {
            "mnist": 10,
            "cifar10": 10,
            "imagenet": 1000,
        }
        n_classes = n_classes_map.get(
            config.task.dataset, task_params.get("n_classes", 10)
        )
    else:
        n_classes = 1

    # Create decoder
    decoder_config = model_config.decoder_network
    if decoder_config.type.lower() == "identity":
        decoder = Identity(core_network.output_dim)
    else:
        decoder_params = (
            dict(decoder_config.parameters) if decoder_config.parameters else {}
        )
        decoder_params["input_dim"] = core_network.output_dim
        decoder_params["output_dim"] = n_classes
        decoder = get_architecture(decoder_config.type, decoder_params)

    # Create full model
    if task_type == "classification":
        model = Classifier(
            encoder_network=encoder,
            core_network=core_network,
            decoder_network=decoder,
        )
    else:
        model = Regressor(
            encoder_network=encoder,
            core_network=core_network,
            decoder_network=decoder,
        )

    # Get precision from config
    trainer_params = (
        config.train.main.trainer.__dict__
        if hasattr(config.train.main, "trainer")
        else {}
    )
    dtype = trainer_params.get("dtype", "fp16")

    # Get optimizer type - it's not directly specified in the config, default to adam
    optimizer_type = "adam"  # Default optimizer for dendritic models

    # Create estimator and print report
    estimator = GPUMemoryEstimator(dtype=dtype, optimizer=optimizer_type)
    estimator.print_estimation_report(model, batch_size, input_shape, config)

    return estimator, model
