#!/usr/bin/env python3
"""
GPU Memory Estimation for Dendritic Models.

This module provides comprehensive GPU memory estimation for dendritic neural networks,
including support for FSDP distributed training.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1024**3
DEFAULT_DTYPE_BYTES = 2
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}
DEFAULT_OPTIMIZER_STATE_MULTIPLIER = 2.0
OPTIMIZER_STATE_MULTIPLIERS = {
    "sgd": 1.0,  # Only momentum
    "adam": 2.0,  # Momentum + variance
    "adamw": 2.0,  # Same as Adam
    "lamb": 2.0,  # Similar to Adam
    "rmsprop": 1.0,  # Running average
}
DENDRITIC_BRANCHING_OVERHEAD = 10.0
TOPK_INDEX_VALUE_FACTOR = 2
TOPK_SPARSE_GRADIENT_FACTOR = 4
DEFAULT_TOPK_SYNAPSES = 100
BLOCKLINEAR_ACTIVATION_FACTOR = 2
LINEAR_ACTIVATION_FACTOR = 2
PARAMETER_FALLBACK_ACTIVATION_MULTIPLIER = 60
GRADIENT_GRAPH_OVERHEAD = 2.5
FSDP_COMMUNICATION_OVERHEAD = 0.5
FSDP_ACTIVATION_CHECKPOINT_FACTOR = 0.7
FSDP_CONFIG_ACTIVATION_FACTOR = 0.8
FSDP_CONFIG_OVERHEAD_FACTOR = 0.1
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAINING_DTYPE = "fp16"
DEFAULT_CLASS_COUNT = 10
REGRESSION_OUTPUT_DIM = 1
DATASET_CLASS_COUNTS = {
    "mnist": 10,
    "cifar10": 10,
    "imagenet": 1000,
}
DEFAULT_TRAIN_VALID_SPLIT = 0.8
DEFAULT_INPUT_DIM = 784
MNIST_FLAT_INPUT_SHAPE = (784,)
CIFAR10_FLAT_INPUT_SHAPE = (3072,)
IMAGENET_FLAT_INPUT_SHAPE = (150528,)
IMAGENET_IMAGE_INPUT_SHAPE = (3, 224, 224)
GPU_MEMORY_SAFETY_FACTOR = 0.85
PRACTICAL_FSDP_GPU_LIMIT = 16


def _bytes_to_gib(num_bytes: float) -> float:
    """Convert bytes to GiB using the estimator's historical convention."""
    return num_bytes / BYTES_PER_GIB


def _resolve_batch_size(config: Any, batch_size: Optional[int] = None) -> int:
    """Resolve the batch size from explicit, current, or legacy config paths."""
    if batch_size is not None:
        return batch_size

    resolved_batch_size = DEFAULT_BATCH_SIZE

    if hasattr(config, "training") and hasattr(config.training, "main"):
        if hasattr(config.training.main, "common"):
            resolved_batch_size = getattr(
                config.training.main.common, "batch_size", resolved_batch_size
            )
        elif hasattr(config.training.main, "trainer"):
            trainer_params = config.training.main.trainer.__dict__
            resolved_batch_size = trainer_params.get("batch_size", resolved_batch_size)
    elif hasattr(config, "train") and hasattr(config.train, "main"):
        if hasattr(config.train.main, "trainer"):
            trainer_params = config.train.main.trainer.__dict__
            resolved_batch_size = trainer_params.get("batch_size", resolved_batch_size)

    return resolved_batch_size


def _resolve_task_config(config: Any) -> tuple[Any, Any]:
    """Resolve legacy task config or synthesize one from the data config."""
    if hasattr(config, "task"):
        task_cfg = config.task
        task_params = getattr(task_cfg, "parameters", {}) or {}
        return task_cfg, task_params

    task_cfg = type("TaskConfig", (), {})()
    task_cfg.dataset = config.data.dataset_name
    task_cfg.data_path = getattr(config.data, "base_dir", None)
    task_cfg.parameters = getattr(config.data, "processing", {})
    task_cfg.train_valid_split = getattr(
        config.experiment, "train_valid_split", DEFAULT_TRAIN_VALID_SPLIT
    )
    return task_cfg, task_cfg.parameters or {}


def _fallback_input_shape(task_cfg: Any, task_params: Any) -> tuple[int, ...]:
    """Return the historical hardcoded input shape when dataset loading fails."""
    dataset_name = getattr(task_cfg, "dataset", "cifar10")
    if dataset_name == "mnist":
        return MNIST_FLAT_INPUT_SHAPE
    if dataset_name == "cifar10":
        return CIFAR10_FLAT_INPUT_SHAPE
    if dataset_name == "imagenet":
        if task_params.get("flatten", False):
            return IMAGENET_FLAT_INPUT_SHAPE
        return IMAGENET_IMAGE_INPUT_SHAPE

    input_dim = task_params.get("input_dim", DEFAULT_INPUT_DIM)
    return (input_dim,)


def _flat_input_dim(input_shape: tuple[int, ...]) -> int:
    """Return the flattened input dimension used by flat architectures."""
    return input_shape[0] if len(input_shape) == 1 else int(np.prod(input_shape))


def _config_params(config_node: Any) -> dict[str, Any]:
    """Return explicit legacy/current params from a config node."""
    return dict(
        getattr(config_node, "parameters", {}) or getattr(config_node, "params", {})
    )


def _core_params_from_config(core_config: Any) -> dict[str, Any]:
    """Merge non-structured core parameter sections into one parameter dict."""
    if hasattr(core_config, "parameters"):
        return dict(core_config.parameters)

    core_params = {}
    for section_name in ("architecture", "connectivity", "morphology"):
        if hasattr(core_config, section_name):
            core_params.update(dict(getattr(core_config, section_name).__dict__))
    return core_params


def _resolve_n_classes(task_type: str, task_cfg: Any, task_params: Any) -> int:
    """Resolve decoder output size from task type and dataset metadata."""
    if task_type != "classification":
        return REGRESSION_OUTPUT_DIM

    dataset_name = getattr(task_cfg, "dataset", "cifar10")
    return DATASET_CLASS_COUNTS.get(
        dataset_name, task_params.get("n_classes", DEFAULT_CLASS_COUNT)
    )


def _resolve_trainer_params(config: Any) -> dict[str, Any]:
    """Resolve trainer params from current or legacy config paths."""
    if hasattr(config, "training") and hasattr(config.training, "main"):
        if hasattr(config.training.main, "common"):
            return config.training.main.common.__dict__
        if hasattr(config.training.main, "trainer"):
            return config.training.main.trainer.__dict__
    elif hasattr(config, "train") and hasattr(config.train, "main"):
        if hasattr(config.train.main, "trainer"):
            return config.train.main.trainer.__dict__
    return {}


def _resolve_training_dtype(config: Any) -> str:
    """Resolve training dtype used by the memory estimator."""
    return _resolve_trainer_params(config).get("dtype", DEFAULT_TRAINING_DTYPE)


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

    GPU_SPECS: ClassVar[dict[str, dict[str, Any]]] = {
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
        return DTYPE_BYTES.get(dtype, DEFAULT_DTYPE_BYTES)

    def _get_optimizer_multiplier(self, optimizer: str) -> float:
        """Get memory multiplier for optimizer states."""
        return OPTIMIZER_STATE_MULTIPLIERS.get(
            optimizer.lower(), DEFAULT_OPTIMIZER_STATE_MULTIPLIER
        )

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
                        * TOPK_INDEX_VALUE_FACTOR
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
                                * TOPK_INDEX_VALUE_FACTOR
                            )
                            activation_memory_bytes += topk_memory

            # Add substantial overhead for gradient computation graphs in dendritic networks
            # Dendritic networks have complex backward passes with branch-wise operations
            activation_memory_bytes *= DENDRITIC_BRANCHING_OVERHEAD

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
                    K = getattr(module, "K", DEFAULT_TOPK_SYNAPSES)

                    # Memory for sparse connectivity pattern and gradients
                    # Store: indices (K ints per output) + values (K floats per output) + gradients
                    layer_activation = (
                        batch_size
                        * out_features
                        * K
                        * self.bytes_per_param
                        * TOPK_SPARSE_GRADIENT_FACTOR
                    )
                    activation_memory_bytes += layer_activation

                elif "BlockLinear" in type(module).__name__:
                    # BlockLinear combines branches
                    if hasattr(module, "weight"):
                        weight = module.weight
                        # Handle case where weight might be a property/function
                        if callable(weight):
                            continue  # Skip if weight is a function
                        if hasattr(weight, "shape"):
                            weight_shape = weight.shape
                            layer_activation = (
                                batch_size
                                * np.prod(weight_shape)
                                * self.bytes_per_param
                                * BLOCKLINEAR_ACTIVATION_FACTOR
                            )
                            activation_memory_bytes += layer_activation

                elif isinstance(module, torch.nn.Linear):
                    in_features = getattr(module, "in_features", 0)
                    out_features = getattr(module, "out_features", 0)
                    layer_activation = (
                        batch_size
                        * (in_features + out_features)
                        * self.bytes_per_param
                        * LINEAR_ACTIVATION_FACTOR
                    )
                    activation_memory_bytes += layer_activation

            # If we couldn't get a good estimate, use parameter-based estimation
            if activation_memory_bytes == 0:
                total_params = sum(p.numel() for p in model.parameters())
                # For dendritic models, activations can be 10-20x parameters due to branching
                activation_memory_bytes = (
                    total_params
                    * self.bytes_per_param
                    * PARAMETER_FALLBACK_ACTIVATION_MULTIPLIER
                )  # Multiplier for dendritic models

            # Add gradient graph overhead
            activation_memory_bytes *= GRADIENT_GRAPH_OVERHEAD

        return _bytes_to_gib(activation_memory_bytes)

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
        param_memory_gb = _bytes_to_gib(total_params * self.bytes_per_param)

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
                base_memory * FSDP_COMMUNICATION_OVERHEAD
            )  # Increased overhead for communication/sharding

            # With FSDP, memory is sharded across GPUs
            param_memory_gb /= num_gpus
            gradient_memory_gb /= num_gpus
            optimizer_memory_gb /= num_gpus
            # Activations are not sharded (each GPU computes full forward pass)
            # But FSDP can use gradient checkpointing to reduce activation memory
            activation_memory_gb *= FSDP_ACTIVATION_CHECKPOINT_FACTOR

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
                logger.warning(
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
                fits=single_gpu_fits or recommended_gpus <= PRACTICAL_FSDP_GPU_LIMIT,
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
            activation_memory = (
                single_gpu_breakdown.activations * FSDP_CONFIG_ACTIVATION_FACTOR
            )
            fsdp_overhead = FSDP_CONFIG_OVERHEAD_FACTOR * (param_memory + grad_memory)

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

        a100_memory = (
            self.GPU_SPECS["A100-40GB"]["memory_gb"] * GPU_MEMORY_SAFETY_FACTOR
        )
        h100_memory = self.GPU_SPECS["H100"]["memory_gb"] * GPU_MEMORY_SAFETY_FACTOR

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
                f"[CONFIG] {optimal_config['nodes']} nodes x {optimal_config['gpus_per_node']} GPUs/node"
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
    from dendritic_modeling.networks.architectures.factory import get_architecture

    # Load configuration
    config = load_config(config_path)

    batch_size = _resolve_batch_size(config, batch_size)
    task_cfg, task_params = _resolve_task_config(config)

    # Load real dataset to get accurate input shape (like train_experiments.py)
    try:
        # Get datasets using the same method as training scripts
        train_ds, _, _ = get_unified_datasets(
            task_cfg=task_cfg,
        )

        # Get sample input shape from real data
        sample_input = train_ds[0][0]  # First sample, input only
        input_shape = sample_input.shape  # This is the real shape from dataset

        logger.info(f"Using real dataset input shape: {input_shape}")

    except Exception as e:
        logger.warning(f"Failed to load dataset, falling back to hardcoded shapes: {e}")

        input_shape = _fallback_input_shape(task_cfg, task_params)

    # Create model (similar to train scripts)
    model_config = config.model

    # Create encoder - handle both old (encoder_network) and new (encoder) structure
    encoder_config = (
        getattr(model_config, "encoder_network", None) or model_config.encoder
    )
    if encoder_config.type.lower() == "identity":
        encoder = Identity(_flat_input_dim(input_shape))
    else:
        encoder_params = _config_params(encoder_config)
        encoder_params["input_dim"] = _flat_input_dim(input_shape)
        encoder = get_architecture(encoder_config.type, encoder_params)

    # Create core network - handle both old (core_network) and new (core) structure
    core_config = getattr(model_config, "core_network", None) or model_config.core

    if core_config.type.lower() == "einet" and hasattr(core_config, "architecture"):
        # For EINet with structured config, pass the entire config
        core_network = get_architecture(
            core_config.type, core_config, encoder.output_dim
        )
    else:
        # Handle parameter structure - combine architecture, connectivity, etc.
        core_params = _core_params_from_config(core_config)
        core_params["input_dim"] = encoder.output_dim
        core_network = get_architecture(core_config.type, core_params)

    # Get task type and number of classes
    task_type = model_config.task  # task is on model config, not task config
    n_classes = _resolve_n_classes(task_type, task_cfg, task_params)

    # Create decoder - handle both old (decoder_network) and new (decoder) structure
    decoder_config = (
        getattr(model_config, "decoder_network", None) or model_config.decoder
    )
    if decoder_config.type.lower() == "identity":
        decoder = Identity(core_network.output_dim)
    else:
        decoder_params = _config_params(decoder_config)
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

    dtype = _resolve_training_dtype(config)

    # Get optimizer type - it's not directly specified in the config, default to adam
    optimizer_type = "adam"  # Default optimizer for dendritic models

    # Create estimator and print report
    estimator = GPUMemoryEstimator(dtype=dtype, optimizer=optimizer_type)
    estimator.print_estimation_report(model, batch_size, input_shape, config)

    return estimator, model
