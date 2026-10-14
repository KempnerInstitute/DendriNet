"""Lossless compaction of trained dendritic sparse topology."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config.compression import ModelCompressionConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DeepstLinear,
    DenseToSparseLinear,
    IndexedDynamicTopKLinear,
    IndexedRewireLinear,
    IndexedSparseLinear,
    TopKLinear,
)
from dendritic_modeling.networks.checkpoints import (
    BITMASK_ASCENDING_ENCODING,
    atomic_torch_save,
    compact_model_state_dict,
    decode_sparse_bitmask_state_dict,
    extract_model_state_dict,
    per_contact_tensors,
    sha256_file,
)

logger = logging.getLogger(__name__)

COMPACT_CHECKPOINT_FORMAT = "dendritic_modeling.compact_checkpoint"
COMPACT_CHECKPOINT_VERSION = 1
_INDICES = "connection_indices"


@dataclass(frozen=True)
class FrozenSparseModuleRecord:
    """Reconstruction record for one fixed indexed sparse module."""

    path: str
    source_type: str
    target_type: str
    in_features: int
    out_features: int
    synapses_per_output: int
    param_space: str
    init_method: str
    init_gain: float
    weight_transform: str
    weight_norm_order: int | None
    gamma: float
    output_chunk_size: int
    workspace_mb: float | None
    cache_transformed_weights: bool
    projection_backend: str
    index_dtype: str = "int64"


def _module_device_and_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.get_default_dtype()
    return parameter.device, parameter.dtype


def _sorted_selected_topology(
    indices: torch.Tensor,
    raw_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_indices, order = indices.to(dtype=torch.long).sort(dim=1)
    sorted_weights = raw_weights.gather(1, order.to(raw_weights.device))
    return sorted_indices, sorted_weights


def _dense_topk_support(module: TopKLinear) -> tuple[torch.Tensor, torch.Tensor]:
    mask = module.weight_mask().to(dtype=torch.bool)
    counts = mask.sum(dim=1)
    if counts.numel() == 0 or bool((counts != counts[0]).any()):
        raise RuntimeError("TopK support must have a constant row-wise synapse count")
    k = int(counts[0].item())
    coordinates = torch.nonzero(mask, as_tuple=False)
    indices = coordinates[:, 1].reshape(module.out_features, k)
    raw_weights = module.pre_w.gather(1, indices.to(module.pre_w.device))
    return _sorted_selected_topology(indices, raw_weights)


def _dynamic_indexed_support(
    module: IndexedDynamicTopKLinear,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = module.active_sparse_mask().to(dtype=torch.bool)
    counts = active.sum(dim=1)
    if bool((counts != module.K).any()):
        raise RuntimeError(
            "Indexed dynamic support must select exactly K candidates per row"
        )
    slots = torch.nonzero(active, as_tuple=False)[:, 1].reshape(
        module.out_features, module.K
    )
    indices = module.connection_indices.gather(
        1, slots.to(module.connection_indices.device)
    )
    raw_weights = module.pre_w.gather(1, slots.to(module.pre_w.device))
    return _sorted_selected_topology(indices, raw_weights)


def _fixed_indexed_support(
    module: IndexedSparseLinear,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sorted_selected_topology(module.connection_indices, module.pre_w)


def _replacement_options(module: nn.Module) -> dict[str, Any]:
    return {
        "param_space": str(getattr(module, "param_space", "log")),
        "init_method": str(getattr(module, "init_method", "xavier_normal")),
        "init_gain": float(getattr(module, "init_gain", 1.0)),
        "weight_transform": str(getattr(module, "weight_transform", "exp")),
        "weight_norm_order": getattr(module, "weight_norm_order", None),
        "gamma": float(getattr(module, "gamma", 1.0)),
        "output_chunk_size": int(getattr(module, "output_chunk_size", 2048)),
        "workspace_mb": getattr(module, "workspace_mb", None),
        "cache_transformed_weights": bool(
            getattr(module, "cache_transformed_weights", False)
        ),
        "projection_backend": str(getattr(module, "projection_backend", "eager")),
    }


def _make_indexed_replacement(
    source: nn.Module,
    indices: torch.Tensor,
    raw_weights: torch.Tensor,
    *,
    options: Mapping[str, Any] | None = None,
    index_dtype: str = "int64",
) -> IndexedSparseLinear:
    replacement_options = dict(options or _replacement_options(source))
    device, dtype = _module_device_and_dtype(source)
    replacement = IndexedSparseLinear(
        in_features=int(source.in_features),
        out_features=int(source.out_features),
        K=int(indices.shape[1]),
        connection_indices=indices.detach().cpu(),
        index_dtype=index_dtype,
        persistent_indices=True,
        **replacement_options,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        replacement.connection_indices.copy_(
            indices.to(device=replacement.connection_indices.device)
        )
        replacement.pre_w.copy_(raw_weights.to(device=device, dtype=dtype))
    replacement.pre_w.requires_grad_(source.pre_w.requires_grad)
    replacement.train(source.training)
    return replacement


def _verify_layer_equivalence(
    source: nn.Module,
    replacement: nn.Module,
    *,
    atol: float,
    rtol: float,
) -> None:
    device, dtype = _module_device_and_dtype(source)
    if not (dtype.is_floating_point or dtype.is_complex):
        return
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    sample = torch.randn(
        2,
        int(source.in_features),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    source_training = source.training
    replacement_training = replacement.training
    try:
        source.eval()
        replacement.eval()
        with torch.no_grad():
            expected = source(sample)
            observed = replacement(sample)
    finally:
        source.train(source_training)
        replacement.train(replacement_training)
    dtype_atol = 5e-3 if dtype in {torch.float16, torch.bfloat16} else atol
    dtype_rtol = 5e-3 if dtype in {torch.float16, torch.bfloat16} else rtol
    if not torch.allclose(expected, observed, atol=dtype_atol, rtol=dtype_rtol):
        maximum_error = float((expected - observed).abs().max().item())
        raise RuntimeError(
            "Sparse topology conversion changed layer outputs; "
            f"maximum absolute error={maximum_error:.6g}"
        )


def _convertible_support(
    module: nn.Module,
    config: ModelCompressionConfig,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    sparse = config.sparse_topology
    if type(module) is DenseToSparseLinear:
        if module.noise_level != 0:
            raise ValueError("Cannot losslessly freeze noisy DenseToSparseLinear")
        if sparse.require_final_dense_to_sparse_step and (
            module.current_k != module.target_k
        ):
            raise ValueError(
                "DenseToSparseLinear has not reached target K; finish its pruning "
                "schedule or explicitly disable the final-step requirement"
            )
        return _dense_topk_support(module)
    if type(module) is TopKLinear:
        if module.noise_level != 0:
            raise ValueError("Cannot losslessly freeze noisy TopKLinear")
        return _dense_topk_support(module)
    if type(module) is IndexedDynamicTopKLinear:
        if not sparse.allow_indexed_dynamic:
            return None
        if module.selection != "standard" or module.noise_level != 0:
            raise ValueError(
                "IndexedDynamicTopKLinear must use deterministic standard "
                "selection with noise_level=0 before topology freezing"
            )
        return _dynamic_indexed_support(module)
    if type(module) is IndexedRewireLinear:
        if not sparse.allow_indexed_rewire:
            return None
        return _fixed_indexed_support(module)
    return None


def _record_for(
    path: str,
    source: nn.Module,
    replacement: IndexedSparseLinear,
) -> FrozenSparseModuleRecord:
    options = _replacement_options(replacement)
    return FrozenSparseModuleRecord(
        path=path,
        source_type=type(source).__name__,
        target_type=type(replacement).__name__,
        in_features=replacement.in_features,
        out_features=replacement.out_features,
        synapses_per_output=replacement.K,
        index_dtype=str(replacement.connection_indices.dtype).removeprefix("torch."),
        **options,
    )


def freeze_sparse_topology_(
    model: nn.Module,
    config: ModelCompressionConfig | None = None,
) -> list[FrozenSparseModuleRecord]:
    """Replace deterministic dynamic sparse layers with exact fixed topology.

    The operation mutates ``model`` and is intended for inference/export after
    training. Existing :class:`IndexedSparseLinear` modules are already compact
    and are left unchanged.
    """

    policy = config or ModelCompressionConfig()
    policy.validate()
    if not policy.sparse_topology.enabled:
        return []

    candidates: list[
        tuple[
            nn.Module,
            str,
            str,
            nn.Module,
            tuple[torch.Tensor, torch.Tensor],
        ]
    ] = []
    converted_paths: dict[int, str] = {}

    def collect(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent._modules.items()):
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            support = _convertible_support(child, policy)
            if support is None:
                collect(child, path)
                continue
            previous_path = converted_paths.get(id(child))
            if previous_path is not None:
                raise ValueError(
                    "Shared dynamic sparse modules cannot be frozen safely: "
                    f"{previous_path!r} and {path!r} reference the same module"
                )
            converted_paths[id(child)] = path
            candidates.append((parent, name, path, child, support))

    collect(model, "")
    unconverted = [
        f"{path} ({type(module).__name__})"
        for path, module in model.named_modules()
        if isinstance(module, (TopKLinear, DeepstLinear))
        and id(module) not in converted_paths
    ]
    if unconverted and not policy.sparse_topology.allow_unconverted:
        raise ValueError(
            "sparse_topology cannot freeze these sparse layers losslessly: "
            + ", ".join(unconverted)
            + ". Set sparse_topology.allow_unconverted: true to export them "
            "unchanged, with their dense candidate weights."
        )
    prepared: list[
        tuple[nn.Module, str, IndexedSparseLinear, FrozenSparseModuleRecord]
    ] = []
    for parent, name, path, child, support in candidates:
        indices, raw_weights = support
        options = _replacement_options(child)
        for key in ("projection_backend", "workspace_mb"):
            value = getattr(policy.sparse_topology, key)
            if value is not None:
                options[key] = value
        replacement = _make_indexed_replacement(
            child,
            indices,
            raw_weights,
            options=options,
            index_dtype=policy.sparse_topology.index_dtype,
        )
        if policy.verification.enabled:
            _verify_layer_equivalence(
                child,
                replacement,
                atol=policy.verification.atol,
                rtol=policy.verification.rtol,
            )
        prepared.append(
            (parent, name, replacement, _record_for(path, child, replacement))
        )

    for parent, name, replacement, _ in prepared:
        parent._modules[name] = replacement
    return [record for _, _, _, record in prepared]


def _module_at_path(model: nn.Module, path: str) -> tuple[nn.Module, str, nn.Module]:
    parts = path.split(".")
    if not parts or not path:
        raise ValueError("Compact module paths must be non-empty")
    parent = model
    for part in parts[:-1]:
        child = parent._modules.get(part)
        if child is None:
            raise KeyError(f"Model has no module at compact path {path!r}")
        parent = child
    name = parts[-1]
    child = parent._modules.get(name)
    if child is None:
        raise KeyError(f"Model has no module at compact path {path!r}")
    return parent, name, child


def prepare_model_for_compact_state_(
    model: nn.Module,
    manifest: list[Mapping[str, Any]],
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Reconstruct fixed indexed modules before strict compact-state loading."""

    for item in manifest:
        path = str(item["path"])
        parent, name, source = _module_at_path(model, path)
        index_key = f"{path}.connection_indices"
        if index_key not in state_dict:
            raise KeyError(f"Compact state is missing {index_key}")
        indices = state_dict[index_key].to(dtype=torch.long)
        expected_shape = (
            int(item["out_features"]),
            int(item["synapses_per_output"]),
        )
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Compact topology shape for {path} is {tuple(indices.shape)}, "
                f"expected {expected_shape}"
            )
        weight_key = f"{path}.pre_w"
        if weight_key not in state_dict:
            raise KeyError(f"Compact state is missing {weight_key}")
        replacement = _make_indexed_replacement(
            source,
            indices,
            state_dict[weight_key],
            index_dtype=str(item.get("index_dtype", "int64")),
            options={
                key: item[key]
                for key in (
                    "param_space",
                    "init_method",
                    "init_gain",
                    "weight_transform",
                    "weight_norm_order",
                    "gamma",
                    "output_chunk_size",
                    "workspace_mb",
                    "cache_transformed_weights",
                    "projection_backend",
                )
            },
        )
        parent._modules[name] = replacement


def _module_of_indices(
    modules: Mapping[str, nn.Module], key: str
) -> tuple[str, nn.Module | None]:
    path = key[: -len(_INDICES)]
    if path and not path.endswith("."):
        return path, None  # e.g. ``base_connection_indices``: not an indexed layer
    path = path[:-1]
    return path, modules.get(path)


def _contacts_may_be_reordered(
    module: nn.Module | None,
    key: str,
    state_dict: Mapping[str, torch.Tensor],
) -> bool:
    cls = type(module)
    return (
        module is not None
        and vars(cls).get("SLOT_ORDER_MATTERS", True) is False
        and tuple(getattr(cls, "CONTACT_SLOT_STATE", ())) == ("pre_w",)
        and not per_contact_tensors(state_dict, key)
    )


def bitmask_slot_order_keys(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> set[str]:
    """``connection_indices`` keys whose contacts keep their slot order.

    The bitmask encoding of a compact checkpoint may reorder a layer's
    contacts into ascending input order only when that changes nothing but
    the order in which they are summed: the layer's class itself (not only a
    parent class) declares ``SLOT_ORDER_MATTERS = False`` with ``pre_w`` as
    its only per-contact tensor (``CONTACT_SLOT_STATE``), and its state holds
    no other tensor with one entry per contact. Of the library's layers only
    the fixed ``IndexedSparseLinear`` qualifies. Every other layer keeps its
    slot order (see :func:`~dendritic_modeling.networks.checkpoints.compact_sparse_bitmask_state_dict`):
    ``indexed_release`` (per-contact impact and ages; release draws and
    cellwise rewiring follow slot positions), ``indexed_rewire`` (pruning
    ties, pairing of new inputs with pruned slots), ``indexed_dynamic`` (Top-K
    ties, selection noise) and layer types defined elsewhere.
    """
    state = model.state_dict() if state_dict is None else state_dict
    modules = dict(model.named_modules())
    keep = set()
    for key in state:
        if not key.endswith(_INDICES):
            continue
        _, module = _module_of_indices(modules, key)
        if not _contacts_may_be_reordered(module, key, state):
            keep.add(key)
    return keep


def _reordered_contact_warnings(
    model: nn.Module, encodings: Mapping[str, Any] | None
) -> list[str]:
    """Layers whose contacts a checkpoint stores reordered although they matter.

    Bitmask encodings before 2026-10-13 reordered every layer's contacts and
    weights into ascending input order and left other per-contact tensors in
    their slots. The stored label (``"bitmask"``) cannot tell whether a row
    was already ascending, so each such layer is reported.
    """
    if not isinstance(encodings, Mapping):
        return []
    state = model.state_dict()
    modules = dict(model.named_modules())
    warnings = []
    for key, label in sorted(encodings.items()):
        if label != BITMASK_ASCENDING_ENCODING or key not in state:
            continue
        path, module = _module_of_indices(modules, key)
        if module is None or _contacts_may_be_reordered(module, key, state):
            continue
        message = (
            f"{path or '<root>'} ({type(module).__name__}): this checkpoint stores "
            "its contacts reordered into ascending input order, as bitmask "
            "encodings written before 2026-10-13 did for every layer"
        )
        others = per_contact_tensors(state, key)
        if others:
            message += (
                f"; {', '.join(name.rsplit('.', 1)[-1] for name in others)} kept "
                "their slots and may be attached to other contacts"
            )
        if getattr(type(module), "SLOT_ORDER_MATTERS", True):
            message += (
                "; training updates that follow the slot order can differ from "
                "the source model's"
            )
        warnings.append(
            message + ". Export it again from the source checkpoint to keep the "
            "contact order."
        )
    return warnings


def compact_model_checkpoint(
    model: nn.Module,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    config: ModelCompressionConfig | None = None,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze native sparse topology and save a self-describing checkpoint.

    With the ``bitmask`` topology encoding, layers listed by
    :func:`bitmask_slot_order_keys` keep their contact order exactly; the
    others are stored with their contacts in ascending input order.
    """

    policy = config or ModelCompressionConfig()
    source = Path(source_checkpoint).expanduser().resolve()
    output = Path(output_checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Compact output must differ from the source checkpoint")
    if output.exists():
        raise FileExistsError(output)

    records = freeze_sparse_topology_(model, policy)
    state = model.state_dict()
    compact_state, encodings = compact_model_state_dict(
        state,
        topology_encoding=policy.sparse_topology.topology_encoding,
        keep_slot_order=bitmask_slot_order_keys(model, state),
    )
    source_bytes = source.stat().st_size
    payload = {
        "format": COMPACT_CHECKPOINT_FORMAT,
        "format_version": COMPACT_CHECKPOINT_VERSION,
        "state_dict": compact_state,
        "sparse_topology_manifest": [asdict(record) for record in records],
        "sparse_index_encoding": encodings,
        "compression_config": asdict(policy),
        "provenance": dict(provenance or {}),
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "bytes": source_bytes,
        },
    }
    atomic_torch_save(payload, output)
    output_bytes = output.stat().st_size
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_bytes": output_bytes,
        "source": str(source),
        "source_bytes": source_bytes,
        "size_reduction_factor": source_bytes / max(output_bytes, 1),
        "converted_modules": len(records),
        "topology_encoding": policy.sparse_topology.topology_encoding,
    }


def load_compact_model_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = True,
    apply_dense_runtime: bool = False,
) -> dict[str, Any]:
    """Load a compact checkpoint into a model built from its training config.

    The report's ``contact_order_warnings`` lists layers whose contacts the
    checkpoint stores reordered although their per-contact state or training
    updates depend on the slot order (bitmask files written before
    2026-10-13); each is also logged as a warning.
    """

    resolved = Path(checkpoint).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != (
        COMPACT_CHECKPOINT_FORMAT
    ):
        raise ValueError("Checkpoint is not a dendritic compact checkpoint")
    if payload.get("format_version") != COMPACT_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported compact checkpoint version {payload.get('format_version')!r}"
        )
    state = extract_model_state_dict(payload)
    state = decode_sparse_bitmask_state_dict(state)
    manifest = payload.get("sparse_topology_manifest", [])
    if not isinstance(manifest, list):
        raise TypeError("sparse_topology_manifest must be a list")
    prepare_model_for_compact_state_(model, manifest, state)
    incompatible = model.load_state_dict(state, strict=strict)
    contact_order_warnings = _reordered_contact_warnings(
        model, payload.get("sparse_index_encoding")
    )
    for warning in contact_order_warnings:
        logger.warning(warning)
    runtime_report: dict[str, Any] | None = None
    if apply_dense_runtime:
        compression_payload = payload.get("compression_config", {})
        if not isinstance(compression_payload, dict):
            raise TypeError("compression_config must be a mapping")
        policy = ModelCompressionConfig(**compression_payload)
        from dendritic_modeling.deployment.torchao import optimize_dense_runtime_

        model.eval()
        runtime_report = optimize_dense_runtime_(model, policy.dense_runtime)
    return {
        "checkpoint": str(resolved),
        "sha256": sha256_file(resolved),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "converted_modules": len(manifest),
        "contact_order_warnings": contact_order_warnings,
        "dense_runtime": runtime_report,
    }


__all__ = [
    "COMPACT_CHECKPOINT_FORMAT",
    "COMPACT_CHECKPOINT_VERSION",
    "FrozenSparseModuleRecord",
    "bitmask_slot_order_keys",
    "compact_model_checkpoint",
    "freeze_sparse_topology_",
    "load_compact_model_checkpoint",
    "prepare_model_for_compact_state_",
]
