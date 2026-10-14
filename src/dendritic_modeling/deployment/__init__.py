"""Deployment and post-training model-compaction utilities."""

from dendritic_modeling.deployment.compression import (
    COMPACT_CHECKPOINT_FORMAT,
    COMPACT_CHECKPOINT_VERSION,
    bitmask_slot_order_keys,
    compact_model_checkpoint,
    freeze_sparse_topology_,
    load_compact_model_checkpoint,
    prepare_model_for_compact_state_,
)
from dendritic_modeling.deployment.torchao import optimize_dense_runtime_

__all__ = [
    "COMPACT_CHECKPOINT_FORMAT",
    "COMPACT_CHECKPOINT_VERSION",
    "bitmask_slot_order_keys",
    "compact_model_checkpoint",
    "freeze_sparse_topology_",
    "load_compact_model_checkpoint",
    "optimize_dense_runtime_",
    "prepare_model_for_compact_state_",
]
