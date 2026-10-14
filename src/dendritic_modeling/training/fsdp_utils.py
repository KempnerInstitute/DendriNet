"""
FullyShardedDataParallel Utilities for Large Dendritic Models
=========================================

This module provides utilities for training extremely large dendritic models
that cannot fit on a single GPU using PyTorch's Fully Sharded Data Parallel (FullyShardedDataParallel).
"""

import functools
import logging
import random
from collections.abc import Mapping
from types import MethodType
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    DendriticBranchLayer,
    ExcitationInhibitionLayer,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.recurrent.ei_layer import EILayer
from dendritic_modeling.networks.architectures.recurrent.ei_network import EINetwork
from dendritic_modeling.networks.checkpoints import atomic_torch_save
from dendritic_modeling.networks.utils.sparse_maintenance import (
    forward_stateful_modules,
    mark_rank_reduced_statistics,
)
from dendritic_modeling.utils.hooks import iter_named_modules_matching

logger = logging.getLogger(__name__)


_FSDP_DTYPE_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


# buffer_dtype value that leaves every buffer in its own dtype (no cast).
KEEP_BUFFER_DTYPE = "keep"


def _resolve_buffer_dtype(value: Any) -> Optional[torch.dtype]:
    if str(value).strip().lower() == KEEP_BUFFER_DTYPE:
        return None
    return _resolve_fsdp_dtype(value, field_name="buffer_dtype")


def mixed_precision_record(fsdp_config: dict[str, Any]) -> Optional[dict[str, str]]:
    """The resolved FSDP mixed-precision dtypes, or None without mixed precision."""
    policy = fsdp_config.get("mixed_precision")
    if policy is None:
        return None
    return {
        "param_dtype": str(policy.param_dtype),
        "reduce_dtype": str(policy.reduce_dtype),
        "buffer_dtype": (
            KEEP_BUFFER_DTYPE
            if policy.buffer_dtype is None
            else str(policy.buffer_dtype)
        ),
    }


def frozen_decay_buffers(model: torch.nn.Module, dtype: torch.dtype) -> list[str]:
    """``decays`` buffers with factors that ``dtype`` rounds to exactly 1.

    Recurrent layers cast their decay factors exp(-dt/tau) to the activation
    dtype in every forward. A factor of 1 leaves that level's trace at its
    initial value, so the level stops responding to input (in bf16, every
    tau above about 511 steps).
    """
    findings = []
    for name, buffer in model.named_buffers():
        if name.rsplit(".", 1)[-1] != "decays" or not buffer.is_floating_point():
            continue
        exact = buffer.detach().double()
        frozen = (buffer.detach().to(dtype).double() == 1.0) & (exact < 1.0)
        if bool(frozen.any()):
            findings.append(
                f"{name}: {int(frozen.sum())} of {buffer.numel()} decay factors"
            )
    return findings


def _resolve_fsdp_dtype(value: Any, *, field_name: str) -> torch.dtype:
    """Resolve a config dtype alias while preserving old fp16 defaults."""
    if isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    if key not in _FSDP_DTYPE_ALIASES:
        raise ValueError(
            f"Unknown FSDP {field_name}={value!r}. "
            f"Expected one of {sorted(_FSDP_DTYPE_ALIASES)}."
        )
    return _FSDP_DTYPE_ALIASES[key]


def _is_dendritic_fsdp_module(module: torch.nn.Module) -> bool:
    """Return whether a module should be treated as an FSDP dendritic unit."""
    return isinstance(
        module,
        (DendriticBranchLayer, ExcitationInhibitionLayer, EINetwork, EILayer),
    )


def _is_large_linear_checkpoint_candidate(module: torch.nn.Module) -> bool:
    """Return whether a Linear module meets the checkpointing size rule."""
    return isinstance(module, torch.nn.Linear) and module.in_features > 1000


def _is_gradient_checkpoint_candidate(_name: str, module: torch.nn.Module) -> bool:
    """Return whether a module should have activation checkpointing applied."""
    return _is_dendritic_fsdp_module(module) or _is_large_linear_checkpoint_candidate(
        module
    )


FSDP_AUTO_WRAP_POLICIES = ("none", "indexed_synapses", "size")


def get_fsdp_config(
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD",
    fsdp_config_dict: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return FullyShardedDataParallel keyword arguments for a dendritic model.

    Args:
        mixed_precision: Use FSDP mixed precision.
        cpu_offload: Offload parameters to CPU.
        sharding_strategy: FSDP sharding strategy name.
        fsdp_config_dict: The ``distributed.fsdp`` settings; its values take
            precedence over the arguments above.

    Returns:
        Keyword arguments for ``FullyShardedDataParallel`` plus the private
        ``_gradient_checkpointing`` flag.

    ``mixed_precision_dtype`` sets the parameter, gradient-reduction and buffer
    dtypes together (``param_dtype``, ``reduce_dtype`` and ``buffer_dtype``
    override it one by one); the default is bf16. ``buffer_dtype: keep``
    leaves buffers in their own dtype: FSDP otherwise casts every floating
    buffer for the rest of the run, with no full-precision copy, so in-place
    statistics lose precision and checkpoints store the rounded values.
    ``auto_wrap_policy`` is
    ``none`` (the whole model is one unit), ``indexed_synapses`` (one unit per
    indexed sparse synapse module) or ``size`` (PyTorch's size-based policy
    with ``min_num_params``).
    """
    settings = dict(fsdp_config_dict or {})
    mixed_precision = settings.get("mixed_precision", mixed_precision)
    cpu_offload = settings.get("cpu_offload", cpu_offload)
    sharding_strategy = str(
        settings.get("sharding_strategy", sharding_strategy)
    ).upper()
    auto_wrap_policy_name = str(settings.get("auto_wrap_policy", "none")).lower()
    # `or` (not a .get default) so explicit None granular fields fall back to
    # the single mixed_precision_dtype knob.
    mp_dtype_knob = settings.get("mixed_precision_dtype") or "bf16"
    mp_param_dtype = settings.get("param_dtype") or mp_dtype_knob
    mp_reduce_dtype = settings.get("reduce_dtype") or mp_param_dtype
    mp_buffer_dtype = settings.get("buffer_dtype") or mp_param_dtype

    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }

    mp_config = None
    if mixed_precision:
        mp_config = MixedPrecision(
            param_dtype=_resolve_fsdp_dtype(mp_param_dtype, field_name="param_dtype"),
            reduce_dtype=_resolve_fsdp_dtype(
                mp_reduce_dtype, field_name="reduce_dtype"
            ),
            buffer_dtype=_resolve_buffer_dtype(mp_buffer_dtype),
        )

    if auto_wrap_policy_name == "none":
        auto_wrap_policy = None
    elif auto_wrap_policy_name == "indexed_synapses":
        # One unit per leaf sparse-synapse module bounds parameter all-gathers
        # for very large recurrent networks.
        auto_wrap_policy = ModuleWrapPolicy(
            {IndexedSparseLinear, IndexedDynamicTopKLinear}
        )
    elif auto_wrap_policy_name == "size":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(settings.get("min_num_params", 1_000_000)),
        )
    else:
        raise ValueError(
            f"FSDP auto_wrap_policy must be one of {list(FSDP_AUTO_WRAP_POLICIES)}, "
            f"got {auto_wrap_policy_name!r}"
        )

    backward_prefetch = settings.get("backward_prefetch", "BACKWARD_PRE")
    if backward_prefetch is not None:
        backward_prefetch = str(backward_prefetch).upper()
    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        None: BackwardPrefetch.BACKWARD_PRE,
    }
    if backward_prefetch not in backward_prefetch_map:
        raise ValueError(
            "FSDP backward_prefetch must be BACKWARD_PRE or BACKWARD_POST, "
            f"got {backward_prefetch!r}"
        )

    if sharding_strategy not in strategy_map:
        raise ValueError(
            f"FSDP sharding_strategy must be one of {list(strategy_map)}, "
            f"got {sharding_strategy!r}"
        )
    return {
        "sharding_strategy": strategy_map[sharding_strategy],
        "mixed_precision": mp_config,
        "cpu_offload": CPUOffload(offload_params=True) if cpu_offload else None,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": backward_prefetch_map[backward_prefetch],
        "device_id": torch.cuda.current_device(),
        "sync_module_states": settings.get("sync_module_states", True),
        "use_orig_params": settings.get("use_orig_params", True),
        "limit_all_gathers": settings.get("limit_all_gathers", True),
        "forward_prefetch": settings.get("forward_prefetch", True),
        # Private flag consumed by wrap_model_with_fsdp, not an FSDP argument.
        "_gradient_checkpointing": settings.get("gradient_checkpointing", False),
    }


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    fsdp_config: Optional[dict[str, Any]] = None,
) -> FullyShardedDataParallel:
    """
    Wrap a dendritic model with FullyShardedDataParallel.

    Args:
        model: The model to wrap
        fsdp_config: FullyShardedDataParallel configuration (uses defaults if None)

    Returns:
        FullyShardedDataParallel-wrapped model
    """
    if fsdp_config is None:
        fsdp_config = get_fsdp_config()
    else:
        # Private strategy flags must not reach FSDP, but callers may reuse the
        # same configuration while wrapping multiple model components.
        fsdp_config = dict(fsdp_config)

    # Extract private flags (not FullyShardedDataParallel parameters)
    gradient_checkpointing = fsdp_config.pop("_gradient_checkpointing", False)

    # Apply gradient checkpointing if enabled
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for memory savings")
        # Apply gradient checkpointing to the model before FullyShardedDataParallel wrapping

        # For dendritic models, apply checkpointing to major components
        if hasattr(model, "net"):
            # Apply to the core network
            _apply_gradient_checkpointing(model.net)
        else:
            # Apply to the entire model
            _apply_gradient_checkpointing(model)

    # Every rank trains on its own batch; synapses that keep batch statistics
    # reduce them over the ranks so that all ranks select alike.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        mark_rank_reduced_statistics(model)

    # Wrap the model with FullyShardedDataParallel
    fsdp_model = FullyShardedDataParallel(model, **fsdp_config)

    return fsdp_model


def _checkpoint_module_forward(module: torch.nn.Module) -> None:
    """Patch a module forward method to run through activation checkpointing."""
    from torch.utils.checkpoint import checkpoint

    original_forward = module.forward

    def checkpointed_forward(self, *args, orig_forward=original_forward, **kwargs):
        return checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

    module.forward = MethodType(checkpointed_forward, module)


def _apply_gradient_checkpointing(model: torch.nn.Module):
    """
    Apply gradient checkpointing to suitable modules in the model.

    Args:
        model: Model to apply checkpointing to
    """
    # Checkpoint the outermost eligible module only. Nested checkpoint regions
    # recompute the same dendritic operations more than once and can make
    # non-reentrant checkpointing observe different saved-tensor metadata.
    checkpointed_names: list[str] = []
    for name, module in iter_named_modules_matching(
        model,
        _is_gradient_checkpoint_candidate,
    ):
        is_nested = any(
            bool(name) and (not parent_name or name.startswith(f"{parent_name}."))
            for parent_name in checkpointed_names
        )
        if is_nested:
            logger.debug(
                "Skipping nested gradient checkpoint target %s",
                name,
            )
            continue

        stateful = forward_stateful_modules(module)
        if stateful:
            raise ValueError(
                f"gradient_checkpointing would recompute {name or 'the model'} in "
                "backward, replaying training-forward state updates of "
                f"{', '.join(stateful[:3])}"
                + (f" and {len(stateful) - 3} more" if len(stateful) > 3 else "")
                + " (dense-to-sparse schedules, variance statistics or release "
                "impact) so that gradients follow a different mask; set "
                "distributed.fsdp.gradient_checkpointing: false"
            )
        if _is_dendritic_fsdp_module(module):
            _checkpoint_module_forward(module)
            checkpointed_names.append(name)
            logger.debug("Applied gradient checkpointing to %s", name)
        elif _is_large_linear_checkpoint_candidate(module):
            _checkpoint_module_forward(module)
            checkpointed_names.append(name)
            logger.debug(
                "Applied gradient checkpointing to large linear layer %s",
                name,
            )


def gather_full_state_dict(model: FullyShardedDataParallel) -> dict[str, Any]:
    """Return the model's full (unsharded) state dict on rank 0, in CPU memory.

    Every rank must call this; ranks other than 0 receive an empty dict. The
    keys are those of the unwrapped model.
    """
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        return model.state_dict()


def _rank_random_state() -> dict[str, Any]:
    """This rank's Python, NumPy, CPU and current-device CUDA generators.

    The NumPy key array is stored as a tensor so the checkpoint loads with
    ``torch.load(weights_only=True)``.
    """
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (
            name,
            torch.from_numpy(keys.astype(np.int64)),
            int(position),
            int(has_gauss),
            float(cached_gaussian),
        ),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return state


def _restore_rank_random_state(state: Mapping[str, Any]) -> None:
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state(
        (name, keys.numpy().astype(np.uint32), position, has_gauss, cached_gaussian)
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), torch.cuda.current_device())


def _world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def save_fsdp_checkpoint(
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    rank: int = 0,
):
    """Save the full model, optimizer and random-generator state on rank 0.

    Every rank must call this. Rank 0 writes the unsharded model and optimizer
    state and the random generators of every rank (Python, NumPy, CPU and the
    rank's CUDA device), so a run resumed with :func:`load_fsdp_checkpoint`
    on the same number of ranks draws the same stochastic selections, dropout
    masks and samples as the uninterrupted run.

    Args:
        model: FullyShardedDataParallel-wrapped model
        optimizer: Optimizer
        epoch: Current epoch
        save_path: Path to save checkpoint
        rank: Process rank
    """
    # Captured before the state-dict gathers, which draw no random numbers.
    random_state = _rank_random_state()
    global_rank, world_size = _world()
    random_states = [random_state]
    if world_size > 1:
        random_states = [None] * world_size if global_rank == 0 else None
        dist.gather_object(random_state, random_states, dst=0)

    # Rank 0 gathers the full model and optimizer state on the CPU; the other
    # ranks keep no copy.
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()
        optimizer_state_dict = FullyShardedDataParallel.optim_state_dict(
            model, optimizer
        )

        # Only save on rank 0
        if rank == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "random_states": random_states,
            }
            atomic_torch_save(checkpoint, save_path)
            logger.info(f"Saved FullyShardedDataParallel checkpoint to {save_path}")


def load_fsdp_checkpoint(
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    *,
    restore_random_state: bool = True,
):
    """Load a checkpoint written by :func:`save_fsdp_checkpoint`.

    Every rank loads the full model and optimizer state and, with
    ``restore_random_state`` (the default), its own random generators. The
    generators map rank by rank, so a checkpoint written by a different number
    of ranks raises; pass ``restore_random_state=False`` to resume the weights
    and optimizer with this process's generators instead.

    Args:
        model: FullyShardedDataParallel-wrapped model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint
        restore_random_state: Restore this rank's random generators.

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    random_states = None
    if restore_random_state:
        rank, world_size = _world()
        random_states = checkpoint.get("random_states")
        if random_states is None:
            raise ValueError(
                f"{checkpoint_path} holds no random-generator state, so a resumed "
                "run would draw different stochastic selections; pass "
                "restore_random_state=False to resume without it"
            )
        if len(random_states) != world_size:
            raise ValueError(
                f"{checkpoint_path} holds the random generators of "
                f"{len(random_states)} ranks, but this run has {world_size}; "
                "pass restore_random_state=False to resume with fresh generators"
            )

    # Configure state dict loading
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint["model_state_dict"])

    # A rank-local optimizer state would give every rank rank 0's shard, so a
    # failed remapping raises.
    optimizer_state_dict = FullyShardedDataParallel.optim_state_dict_to_load(
        model, optimizer, checkpoint["optimizer_state_dict"]
    )
    optimizer.load_state_dict(optimizer_state_dict)

    if random_states is not None:
        _restore_rank_random_state(random_states[rank])
    return checkpoint["epoch"]


def estimate_memory_requirements(
    num_neurons: list[int],
    num_synapses_per_neuron: list[int],
    num_branches: list[int],
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
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
