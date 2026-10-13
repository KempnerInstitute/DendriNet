"""Safe helpers for strict model-state checkpoint loading."""

from __future__ import annotations

import hashlib
import os
import secrets
from collections import OrderedDict
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.state_versions import (
    clear_legacy_state,
    collect_legacy_state,
)

_BITMASK_SUFFIX = ".__bitmask"
_BITMASK_METADATA_SUFFIX = ".__bitmask_metadata"
_BITMASK_SLOT_ORDER_SUFFIX = ".__bitmask_slot_order"
# Fourth bitmask metadata entry: the rows keep their stored contact order, and
# ``<key>.__bitmask_slot_order`` holds the slot of each ascending contact.
# Decoders before 2026-10-13 accept only three entries, so they refuse such a
# file instead of misreading it.
_BITMASK_SLOT_ORDER_LAYOUT = 1
# ``sparse_index_encoding`` labels of the two bitmask layouts.
BITMASK_ASCENDING_ENCODING = "bitmask"
BITMASK_SLOT_ORDER_ENCODING = "bitmask_slot_order"


def create_temporary_file(directory: str | Path, prefix: str) -> Path:
    """Create an empty temporary file whose mode follows the process umask.

    ``tempfile.mkstemp`` creates owner-only files, which ``os.replace`` would
    carry onto shared checkpoints.
    """
    directory = Path(directory)
    for _ in range(100):
        candidate = directory / f".{prefix}.{secrets.token_hex(6)}.tmp"
        try:
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue
        os.close(descriptor)
        return candidate
    raise FileExistsError(f"could not create a temporary file in {directory}")


def _fsync(path: Path, *, directory: bool = False) -> None:
    flags = os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if directory else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    except OSError:
        if not directory:  # some filesystems cannot sync a directory
            raise
    finally:
        os.close(descriptor)


def commit_temporary_file(temporary: str | Path, destination: str | Path) -> None:
    """Flush ``temporary`` to disk and rename it onto ``destination``."""
    temporary, destination = Path(temporary), Path(destination)
    _fsync(temporary)
    os.replace(temporary, destination)
    _fsync(destination.parent, directory=True)


def atomic_torch_save(payload: Any, destination: str | Path) -> None:
    """Save a PyTorch payload so ``destination`` is never partial.

    The payload is written to a temporary file next to ``destination``,
    flushed to disk and renamed over it, so an interrupted save leaves the
    previous file intact. The file mode follows the umask.
    """

    resolved = Path(destination).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = create_temporary_file(resolved.parent, resolved.name)
    try:
        torch.save(payload, temporary)
        commit_temporary_file(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def _same_state(state: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    if list(state) != list(other):
        return False
    if getattr(state, "_metadata", None) != getattr(other, "_metadata", None):
        return False
    for key, value in state.items():
        stored = other[key]
        if torch.is_tensor(value) != torch.is_tensor(stored):
            return False
        if not torch.is_tensor(value):
            if value != stored:
                return False
        elif (
            value.dtype != stored.dtype
            or value.shape != stored.shape
            or not torch.equal(value.detach().cpu(), stored)
        ):
            return False
    return True


def save_state_dict(
    state_dict: Mapping[str, Any],
    destination: str | Path,
    *,
    identical_to: str | Path | None = None,
) -> str:
    """Atomically save ``state_dict``, or link a file that already holds it.

    ``identical_to`` names a checkpoint that may hold the same state, such as
    the best-validation checkpoint after training restored it. When its
    tensors, keys and metadata equal ``state_dict``, ``destination`` becomes a
    hard link to it, so the run keeps one copy on disk under both names.
    Otherwise (or where the filesystem refuses the link) the state is saved.
    Returns ``"linked"`` or ``"saved"``.
    """
    resolved = Path(destination).expanduser().resolve()
    if identical_to is not None and Path(identical_to).is_file():
        existing = torch.load(identical_to, map_location="cpu", weights_only=True)
        if isinstance(existing, Mapping) and _same_state(state_dict, existing):
            resolved.parent.mkdir(parents=True, exist_ok=True)
            temporary = resolved.parent / f".{resolved.name}.{secrets.token_hex(6)}"
            try:
                os.link(identical_to, temporary)
                os.replace(temporary, resolved)
                _fsync(resolved.parent, directory=True)
                return "linked"
            except OSError:
                temporary.unlink(missing_ok=True)
    atomic_torch_save(state_dict, resolved)
    return "saved"


def atomic_torch_save_candidates(
    payloads: Mapping[str, Any],
    destination: str | Path,
) -> dict[str, Any]:
    """Serialize candidate payloads and atomically keep the smallest one.

    Topology encodings have different break-even points: integer indices are
    compact at low fan-in, while row-wise bitmasks can win for denser rows.
    Comparing tensor ``numel`` is insufficient because ``torch.save`` adds
    storage and metadata overhead.  This helper serializes every complete
    candidate payload in the destination directory, compares the actual file
    sizes, and moves the smallest byte-for-byte candidate into place.

    Ties are resolved by sorted candidate name so repeated exports are
    deterministic.  Candidate sizes describe the exact payloads considered;
    the selected file is not reserialized after selection.
    """

    if not payloads:
        raise ValueError("payloads must contain at least one candidate")
    resolved = Path(destination).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporaries: dict[str, Path] = {}
    candidate_bytes: dict[str, int] = {}
    try:
        for candidate in sorted(payloads):
            temporary = create_temporary_file(
                resolved.parent, f"{resolved.name}.{candidate}"
            )
            temporaries[candidate] = temporary
            torch.save(payloads[candidate], temporary)
            candidate_bytes[candidate] = int(temporary.stat().st_size)
        selected = min(
            candidate_bytes,
            key=lambda candidate: (candidate_bytes[candidate], candidate),
        )
        commit_temporary_file(temporaries[selected], resolved)
        temporaries.pop(selected)
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)
    return {
        "selection_basis": "minimum_complete_torch_serialized_payload_bytes",
        "selected": selected,
        "candidate_bytes": candidate_bytes,
        "output_bytes": int(resolved.stat().st_size),
        "output_sha256": sha256_file(resolved),
    }


def compact_sparse_index_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Pack non-negative sparse topology indices for checkpoint storage.

    Runtime indexed layers use signed 32- or 64-bit integer buffers for broad
    operator compatibility. Checkpoints do not need that overhead: non-negative
    topologies can be stored as unsigned 16- or 32-bit integers and are cast
    back to the destination buffer dtype by ``load_state_dict``. Both signed
    input dtypes normalize to the same packed output for equal index values.
    """

    compact: dict[str, torch.Tensor] = {}
    encoding: dict[str, str] = {}
    for key, value in state_dict.items():
        if key.endswith("._last_forward_param_tensor") or key == (
            "_last_forward_param_tensor"
        ):
            continue
        if key.endswith("connection_indices") and value.dtype in (
            torch.int32,
            torch.int64,
            torch.uint16,
            torch.uint32,
        ):
            # Re-normalize already compact unsigned checkpoints as well as
            # runtime signed buffers. This makes the codec idempotent and lets
            # validators prove that stored uint topology is canonical.
            indices = value.to(dtype=torch.int64)
            max_index = int(indices.max().item()) if indices.numel() else 0
            min_index = int(indices.min().item()) if indices.numel() else 0
            if min_index >= 0 and max_index <= 65_535:
                value = indices.to(dtype=torch.uint16)
            elif min_index >= 0 and max_index <= 4_294_967_295:
                value = indices.to(dtype=torch.uint32)
            encoding[key] = str(value.dtype).removeprefix("torch.")
        compact[key] = value
    return compact, encoding


def _slot_order_dtype(k: int) -> torch.dtype:
    """Narrowest integer dtype that holds the slot numbers ``0 .. k - 1``."""
    if k <= 1 << 8:
        return torch.uint8
    if k <= 1 << 16:
        return torch.uint16
    if k <= 1 << 32:
        return torch.uint32
    return torch.int64


def per_contact_tensors(
    state_dict: Mapping[str, torch.Tensor],
    key: str,
) -> list[str]:
    """Other tensors of the module owning ``key`` with one entry per contact.

    ``key`` names a ``connection_indices`` tensor of shape ``[rows,
    contacts]``. The result lists the module's direct entries (no further dot
    after its prefix), other than the indices and ``pre_w``, whose first two
    dimensions are ``[rows, contacts]``. The bitmask encoding uses it only to
    refuse a reordering that would leave such a tensor behind.
    """
    prefix = key[: -len("connection_indices")]
    shape = tuple(state_dict[key].shape[:2])
    return sorted(
        other
        for other, value in state_dict.items()
        if other.startswith(prefix)
        and "." not in other[len(prefix) :]
        and other not in (key, f"{prefix}pre_w")
        and torch.is_tensor(value)
        and value.ndim >= 2
        and tuple(value.shape[:2]) == shape
    )


def compact_sparse_bitmask_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    keep_slot_order: bool | Collection[str] = False,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Encode row-wise topology as bitmasks instead of integer indices.

    This is useful for moderately dense topology, where even uint16 costs two
    index bytes per retained weight while a bitmask costs one bit per possible
    connection. Every ``connection_indices`` tensor with a ``pre_w`` of the
    same shape is encoded. A bitmask records which inputs a row contacts, not
    in which slot each contact is stored, so each key uses one of two
    layouts:

    ``ascending`` (default; label ``"bitmask"``)
       The contacts of each row and the matching ``pre_w`` entries are
       reordered into ascending input order, so the mask and the weights
       reconstruct the layer. This suits a layer whose only per-contact tensor
       is ``pre_w`` and whose training updates do not depend on the slot
       order, such as a fixed ``IndexedSparseLinear``; its outputs can differ
       in the last bits because the contacts are summed in another order. A
       row that is not already ascending raises ``ValueError`` when the module
       holds another tensor with one entry per contact (``indexed_release``
       impact and ages, for example), which the reordering would misalign.
    ``slot order`` (keys in ``keep_slot_order``; label ``"bitmask_slot_order"``)
       The contacts keep their slots and no other tensor changes. A key with a
       row that is not ascending also stores, under
       ``<key>.__bitmask_slot_order``, the slot of each ascending contact
       (one byte per contact up to 256 contacts per row, two up to 65,536),
       and its metadata gains a fourth entry. Decoding restores the indices
       exactly. Library versions before 2026-10-13 cannot decode such keys and
       raise.

    ``keep_slot_order`` is ``True`` (every key), ``False`` (none) or the
    ``connection_indices`` keys that keep their slot order.
    """

    compact = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("._last_forward_param_tensor")
        and key != "_last_forward_param_tensor"
    }
    original = dict(compact)
    if isinstance(keep_slot_order, bool):
        keep_all, keep_keys = keep_slot_order, frozenset()
    elif isinstance(keep_slot_order, str):
        raise TypeError("keep_slot_order must be a bool or a collection of keys")
    else:
        keep_all, keep_keys = False, frozenset(keep_slot_order)
    encoding: dict[str, str] = {}
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
    for key, value in tuple(compact.items()):
        if not key.endswith("connection_indices"):
            continue
        if value.ndim != 2 or value.dtype not in integer_dtypes:
            continue

        weight_key = f"{key[: -len('connection_indices')]}pre_w"
        weight = compact.get(weight_key)
        if weight is None or tuple(weight.shape) != tuple(value.shape):
            continue

        indices = value.to(dtype=torch.long)
        if indices.numel() and bool((indices < 0).any()):
            continue
        sorted_indices, order = indices.sort(dim=1)
        if sorted_indices.shape[1] > 1 and bool(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
        ):
            raise ValueError(f"Topology indices must be unique within each row: {key}")

        n_rows, k = sorted_indices.shape
        n_inputs = int(sorted_indices.max().item()) + 1 if k else 0
        n_bytes = (n_inputs + 7) // 8
        packed = torch.zeros(
            (n_rows, n_bytes),
            dtype=torch.int16,
            device=sorted_indices.device,
        )
        if k:
            byte_indices = torch.div(sorted_indices, 8, rounding_mode="floor")
            bit_values = torch.bitwise_left_shift(
                torch.ones_like(sorted_indices, dtype=torch.int16),
                torch.remainder(sorted_indices, 8),
            ).to(dtype=torch.int16)
            packed.scatter_add_(1, byte_indices, bit_values)

        ascending = bool(torch.equal(sorted_indices, indices))
        metadata = [n_rows, k, n_inputs]
        if keep_all or key in keep_keys:
            if not ascending:
                compact[f"{key}{_BITMASK_SLOT_ORDER_SUFFIX}"] = order.to(
                    dtype=_slot_order_dtype(k)
                )
                metadata.append(_BITMASK_SLOT_ORDER_LAYOUT)
            encoding[key] = BITMASK_SLOT_ORDER_ENCODING
        else:
            if not ascending:
                companions = per_contact_tensors(original, key)
                if companions:
                    raise ValueError(
                        f"Reordering the contacts of {key} into ascending input "
                        f"order would misalign {', '.join(companions)}, which "
                        "hold one entry per contact; encode it with "
                        "keep_slot_order"
                    )
            compact[weight_key] = weight.gather(1, order.to(weight.device))
            encoding[key] = BITMASK_ASCENDING_ENCODING
        del compact[key]
        compact[f"{key}{_BITMASK_SUFFIX}"] = packed.to(dtype=torch.uint8)
        compact[f"{key}{_BITMASK_METADATA_SUFFIX}"] = torch.tensor(
            metadata,
            dtype=torch.int64,
            device=indices.device,
        )
    return compact, encoding


def decode_sparse_bitmask_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Restore row-wise indices from :func:`compact_sparse_bitmask_state_dict`.

    Keys in the ascending layout decode to ascending indices, aligned with the
    stored (reordered) ``pre_w``; keys with a stored slot order decode to the
    indices in their original slots. Malformed or inconsistent entries raise
    ``RuntimeError``.
    """

    decoded: dict[str, torch.Tensor] = {}
    consumed_slot_orders: set[str] = set()
    for key, value in state_dict.items():
        if key.endswith((_BITMASK_METADATA_SUFFIX, _BITMASK_SLOT_ORDER_SUFFIX)):
            continue
        if not key.endswith(_BITMASK_SUFFIX):
            decoded[key] = value
            continue

        original_key = key.removesuffix(_BITMASK_SUFFIX)
        metadata_key = f"{original_key}{_BITMASK_METADATA_SUFFIX}"
        if metadata_key not in state_dict:
            raise RuntimeError(f"Missing bitmask metadata for {original_key}")
        metadata = state_dict[metadata_key].to(dtype=torch.long).reshape(-1)
        if metadata.numel() not in (3, 4) or (
            metadata.numel() == 4
            and int(metadata[3].item()) != _BITMASK_SLOT_ORDER_LAYOUT
        ):
            raise RuntimeError(f"Invalid bitmask metadata for {original_key}")
        n_rows, k, n_inputs = (int(item) for item in metadata[:3].tolist())
        if tuple(value.shape) != (n_rows, (n_inputs + 7) // 8):
            raise RuntimeError(f"Invalid packed bitmask shape for {original_key}")

        bit_offsets = torch.arange(8, device=value.device, dtype=torch.uint8)
        present = torch.bitwise_and(
            torch.bitwise_right_shift(value.unsqueeze(-1), bit_offsets),
            1,
        ).reshape(n_rows, -1)[:, :n_inputs]
        counts = present.sum(dim=1)
        if bool((counts != k).any()):
            raise RuntimeError(
                f"Packed bitmask row counts do not match K for {original_key}"
            )
        coordinates = torch.nonzero(present, as_tuple=False)
        indices = coordinates[:, 1].reshape(n_rows, k).to(torch.long)

        slot_order_key = f"{original_key}{_BITMASK_SLOT_ORDER_SUFFIX}"
        if metadata.numel() == 4:
            if slot_order_key not in state_dict:
                raise RuntimeError(f"Missing bitmask slot order for {original_key}")
            slots = state_dict[slot_order_key].to(
                device=indices.device, dtype=torch.long
            )
            if tuple(slots.shape) != (n_rows, k) or not torch.equal(
                slots.sort(dim=1).values,
                torch.arange(k, device=slots.device).expand(n_rows, k),
            ):
                raise RuntimeError(
                    f"Bitmask slot order of {original_key} is not a permutation "
                    "of the contact slots in every row"
                )
            indices = torch.empty_like(indices).scatter_(1, slots, indices)
            consumed_slot_orders.add(slot_order_key)
        elif slot_order_key in state_dict:
            raise RuntimeError(
                f"Bitmask slot order of {original_key} without slot-order metadata"
            )
        decoded[original_key] = indices
    orphans = sorted(
        key
        for key in state_dict
        if key.endswith(_BITMASK_SLOT_ORDER_SUFFIX) and key not in consumed_slot_orders
    )
    if orphans:
        raise RuntimeError(f"Bitmask slot orders without a bitmask: {orphans}")
    return decoded


def compact_model_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    topology_encoding: str = "uint",
    keep_slot_order: bool | Collection[str] = False,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return a deployment state dictionary with compact sparse topology.

    ``keep_slot_order`` applies to the ``bitmask`` encoding (see
    :func:`compact_sparse_bitmask_state_dict`); ``uint`` keeps every slot.
    """

    normalized = str(topology_encoding).strip().lower()
    if normalized == "uint":
        return compact_sparse_index_state_dict(state_dict)
    if normalized == "bitmask":
        return compact_sparse_bitmask_state_dict(
            state_dict, keep_slot_order=keep_slot_order
        )
    raise ValueError("topology_encoding must be 'uint' or 'bitmask'")


def topology_encoding_candidates(requested: str) -> tuple[str, ...]:
    """Resolve an explicit or artifact-size-selected topology policy."""

    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return ("uint", "bitmask")
    if normalized in {"uint", "bitmask"}:
        return (normalized,)
    raise ValueError("topology_encoding must be 'auto', 'uint', or 'bitmask'")


def compact_model_state_dict_candidates(
    state_dict: Mapping[str, torch.Tensor],
    *,
    topology_encoding: str = "auto",
    keep_slot_order: bool | Collection[str] = False,
) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, str]]]:
    """Build every state candidate required by a topology encoding policy."""

    return {
        encoding: compact_model_state_dict(
            state_dict,
            topology_encoding=encoding,
            keep_slot_order=keep_slot_order,
        )
        for encoding in topology_encoding_candidates(topology_encoding)
    }


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_ddp_prefix(key: str) -> str:
    """Remove a DistributedDataParallel ``module.`` prefix from a state key."""
    return key[7:] if key.startswith("module.") else key


def copy_state_metadata(
    source: Mapping[str, Any],
    target: OrderedDict,
    *,
    strip_prefix: bool = False,
) -> OrderedDict:
    """Give ``target`` the per-module ``_metadata`` (versions) of ``source``.

    Module load hooks read their saved ``_version`` from it; a state dict
    rebuilt without it loads every module as if it predated all versions.
    ``strip_prefix`` maps a DDP wrapper's ``module`` entries to the bare
    model's module names.
    """
    metadata = getattr(source, "_metadata", None)
    if metadata is None:
        return target
    copied: OrderedDict = OrderedDict()
    for key, value in metadata.items():
        if strip_prefix and key == "module":
            key = ""
        elif strip_prefix:
            key = strip_ddp_prefix(key)
        copied[key] = dict(value)
    target._metadata = copied
    return target


def extract_model_state_dict(payload: Any) -> OrderedDict:
    """Extract a tensor-only model state from a standard checkpoint payload.

    A DDP ``module.`` prefix is removed, and the per-module version metadata
    is kept.
    """

    candidate = payload
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                candidate = nested
                break
    if not isinstance(candidate, Mapping) or not candidate:
        raise TypeError("Checkpoint does not contain a non-empty model state mapping")
    state: OrderedDict = OrderedDict()
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Model state must map string keys to tensors only")
        normalized = strip_ddp_prefix(key)
        if normalized in state:
            raise ValueError(
                f"Checkpoint contains duplicate normalized key {normalized!r}"
            )
        state[normalized] = value
    return copy_state_metadata(candidate, state, strip_prefix=True)


def load_state_dict_with_record(
    model: nn.Module, state: Mapping[str, Any], *, strict: bool = True
) -> dict[str, Any]:
    """Load ``state`` into ``model`` and return what the load did.

    The record lists missing and unexpected keys and ``legacy_state``: keys a
    module filled or interpreted under a legacy rule because the file predates
    them (see :mod:`dendritic_modeling.networks.state_versions`).
    """
    clear_legacy_state(model)
    incompatibilities = model.load_state_dict(state, strict=bool(strict))
    return {
        "missing_keys": list(incompatibilities.missing_keys),
        "unexpected_keys": list(incompatibilities.unexpected_keys),
        "legacy_state": collect_legacy_state(model),
    }


def load_model_state_checkpoint(path: str | Path) -> OrderedDict:
    """Load a state dictionary with PyTorch's restricted weights-only loader."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    return extract_model_state_dict(payload)


def load_initial_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Strictly load a model warm start and return an auditable record."""

    resolved = Path(path).expanduser().resolve()
    observed_sha256 = sha256_file(resolved)
    if expected_sha256 is not None:
        normalized_expected = str(expected_sha256).strip().lower()
        if observed_sha256 != normalized_expected:
            raise ValueError(
                "Initial checkpoint SHA-256 mismatch: "
                f"expected {normalized_expected}, observed {observed_sha256}"
            )
    state = load_model_state_checkpoint(resolved)
    return {
        "schema_version": "initial_model_checkpoint_v1",
        "path": str(resolved),
        "sha256": observed_sha256,
        "strict": bool(strict),
        "state_dict_key_count": len(state),
        **load_state_dict_with_record(model, state, strict=strict),
    }
