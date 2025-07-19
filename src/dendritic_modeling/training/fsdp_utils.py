"""
FSDP Utilities for Large Dendritic Models
=========================================

This module provides utilities for training extremely large dendritic models
that cannot fit on a single GPU using PyTorch's Fully Sharded Data Parallel (FSDP).
"""

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
    CPUOffload,
)
from typing import Optional, Dict, Any, List

from dendritic_modeling import logger
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    DendriticBranchLayer,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
)


def create_fsdp_wrap_policy(min_num_params: int = 1e6):
    """
    Create an auto-wrap policy for FSDP that intelligently wraps dendritic layers.

    Args:
        min_num_params: Minimum number of parameters for a layer to be wrapped

    Returns:
        Auto-wrap policy function
    """

    def dendritic_auto_wrap_policy(module, recurse, nonwrapped_numel):
        # Always wrap these large layers
        if isinstance(module, (DendriticBranchLayer, ExcitationInhibitionLayer)):
            return True

        # Wrap based on parameter count
        if nonwrapped_numel >= min_num_params:
            return True

        # Don't wrap small modules
        return False

    return dendritic_auto_wrap_policy


def get_fsdp_config(
    model_size: str = "large",
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD",
) -> Dict[str, Any]:
    """
    Get FSDP configuration based on model size and requirements.

    Args:
        model_size: "small", "large", or "xlarge"
        mixed_precision: Whether to use mixed precision training
        cpu_offload: Whether to offload parameters to CPU
        sharding_strategy: FSDP sharding strategy

    Returns:
        FSDP configuration dictionary
    """
    # Sharding strategies
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }

    # Mixed precision configuration
    mp_config = None
    if mixed_precision:
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # CPU offload configuration
    cpu_offload_config = None
    if cpu_offload:
        cpu_offload_config = CPUOffload(offload_params=True)

    # Auto-wrap policy based on model size
    if model_size == "xlarge":
        min_params = 100_000  # Wrap layers with >100K params
    elif model_size == "large":
        min_params = 1_000_000  # Wrap layers with >1M params
    else:
        min_params = 10_000_000  # Wrap layers with >10M params

    auto_wrap_policy = create_fsdp_wrap_policy(min_params)

    return {
        "sharding_strategy": strategy_map[sharding_strategy],
        "mixed_precision": mp_config,
        "cpu_offload": cpu_offload_config,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": torch.cuda.current_device(),
        "sync_module_states": True,
        "use_orig_params": True,  # Important for dendritic models
        "limit_all_gathers": True,  # Reduce memory pressure during communication
        "forward_prefetch": True,  # Prefetch parameters for forward pass
    }


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    fsdp_config: Optional[Dict[str, Any]] = None,
) -> FSDP:
    """
    Wrap a dendritic model with FSDP.

    Args:
        model: The model to wrap
        fsdp_config: FSDP configuration (uses defaults if None)

    Returns:
        FSDP-wrapped model
    """
    if fsdp_config is None:
        fsdp_config = get_fsdp_config()

    # Special handling for dendritic models
    if hasattr(model, "net") and isinstance(model.net, ExcitationInhibitionNetwork):
        logger.info(
            "Detected E/I Network - applying FSDP with dendritic-aware wrapping"
        )

        # Estimate model size
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")

        if total_params > 1e9:  # > 1B parameters
            logger.info("Large model detected - using aggressive sharding")
            fsdp_config["sharding_strategy"] = ShardingStrategy.FULL_SHARD

    # Wrap the model
    fsdp_model = FSDP(model, **fsdp_config)

    return fsdp_model


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    rank: int = 0,
):
    """
    Save FSDP model checkpoint.

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        epoch: Current epoch
        save_path: Path to save checkpoint
        rank: Process rank
    """
    # Configure state dict saving
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()

        # Only save on rank 0
        if rank == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Saved FSDP checkpoint to {save_path}")


def load_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
):
    """
    Load FSDP model checkpoint.

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint

    Returns:
        Epoch number from checkpoint
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Configure state dict loading
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"]


def estimate_memory_requirements(
    num_neurons: List[int],
    num_synapses_per_neuron: List[int],
    num_branches: List[int],
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    Estimate memory requirements for a dendritic model.

    Args:
        num_neurons: Number of neurons per layer
        num_synapses_per_neuron: Synapses per neuron per layer
        num_branches: Number of branches per layer
        dtype: Data type for parameters

    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = 4 if dtype == torch.float32 else 2

    total_params = 0
    layer_params = []

    for i in range(len(num_neurons) - 1):
        # TopKLinear parameters
        n_in = num_neurons[i]
        n_out = num_neurons[i + 1]
        n_branches_out = num_branches[i + 1] if i + 1 < len(num_branches) else 1

        # Each branch has its own TopKLinear layers
        topk_params = n_in * n_out * n_branches_out

        # Branch aggregation parameters
        if i > 0:
            branch_agg_params = num_branches[i] * n_branches_out
        else:
            branch_agg_params = 0

        layer_total = topk_params + branch_agg_params
        layer_params.append(layer_total)
        total_params += layer_total

    # Convert to GB
    param_memory_gb = (total_params * bytes_per_param) / 1e9

    # Estimate additional memory needs
    # Gradients: same as parameters
    gradient_memory_gb = param_memory_gb

    # Optimizer states (Adam): 2x parameters
    optimizer_memory_gb = 2 * param_memory_gb

    # Activations (rough estimate): 10% of parameters
    activation_memory_gb = 0.1 * param_memory_gb

    total_memory_gb = (
        param_memory_gb
        + gradient_memory_gb
        + optimizer_memory_gb
        + activation_memory_gb
    )

    return {
        "parameters_gb": param_memory_gb,
        "gradients_gb": gradient_memory_gb,
        "optimizer_gb": optimizer_memory_gb,
        "activations_gb": activation_memory_gb,
        "total_gb": total_memory_gb,
        "total_params": total_params,
        "layer_params": layer_params,
    }


def get_recommended_gpu_count(memory_per_gpu_gb: float, total_memory_gb: float) -> int:
    """
    Get recommended number of GPUs based on memory requirements.

    Args:
        memory_per_gpu_gb: Available memory per GPU (e.g., 80 for H100)
        total_memory_gb: Total memory required

    Returns:
        Recommended number of GPUs
    """
    # Leave 10% headroom
    usable_memory_per_gpu = memory_per_gpu_gb * 0.9

    # Calculate minimum GPUs needed
    min_gpus = int(total_memory_gb / usable_memory_per_gpu) + 1

    # Round up to power of 2 for better communication efficiency
    if min_gpus <= 2:
        return 2
    elif min_gpus <= 4:
        return 4
    elif min_gpus <= 8:
        return 8
    elif min_gpus <= 16:
        return 16
    else:
        return ((min_gpus + 15) // 16) * 16  # Round up to nearest 16
