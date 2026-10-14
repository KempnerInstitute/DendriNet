"""Lower bound on the device memory of a forward pass over a whole split.

A materialized split is evaluated in one forward pass over all its examples.
The split's storage size does not bound that pass: activations grow with the
number of examples and can be far larger than the inputs. This module bounds
the pass's memory from below by measuring small batches, so that a pass which
cannot fit on the device is streamed at once instead of first failing with an
out-of-memory error.

The bound only proves failure. A pass is sent to streaming only when the
tensors it must hold at one time exceed what the device can still provide;
a pass that fits is never redirected, so its batch partition and its numbers
do not change.

How the bound is measured: after one untraced pass, the model runs on the
first 2, 4 and 8 examples under a dispatch mode that records every tensor
the dispatched operations allocate on the device and when the tensors are
released. The bytes held after each operation must grow linearly with the
batch size (same operations, sizes affine in the batch); the held bytes are
then extrapolated to the whole split, and their maximum over the pass is the
bound. Allocations PyTorch makes outside dispatched operations (library
workspaces, allocator rounding) are not counted, which only makes the bound
smaller. The probes run on a copy of the model, in eval mode without
gradients, and restore every global random number generator, so the model and
the pass that follows are unchanged.

No bound is given, and the pass runs as before, when the device is not a CUDA
device, when the split has no more examples than the largest probe, when a
probe fails or does not grow linearly, when forward hooks are registered (a
probe would run them), or when the model holds a module that chooses its chunk
sizes from the batch size (a ``workspace_mb`` budget, or a module that sets
``batch_adaptive_memory = True``), since its allocations at the full batch are
smaller than an extrapolation. A probe is reused for the same model, input
shape and device until a parameter or buffer of the model changes.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import logging
import weakref
from collections.abc import Sequence
from typing import Any

import torch

from dendritic_modeling.utils.reproducibility import preserved_random_state

try:  # PyTorch's dispatch-mode hooks; without them no bound is given
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
except ImportError:  # pragma: no cover - PyTorch without Python dispatch modes
    TorchDispatchMode = object  # type: ignore[assignment,misc]
    tree_flatten = None

logger = logging.getLogger(__name__)

#: Batch sizes of the probes; the bound extrapolates from the last two.
PROBE_BATCH_SIZES = (2, 4, 8)


def _storage_key(tensor: torch.Tensor) -> int | None:
    try:
        pointer = tensor.untyped_storage().data_ptr()
    except Exception:  # sparse, nested or other storage-less tensors
        return None
    return pointer or None


def _on(tensor: torch.Tensor, device: torch.device) -> bool:
    if tensor.device.type != device.type:
        return False
    return device.index is None or tensor.device.index == device.index


def _storage_bytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return 0


class _HeldTensorBytes(TorchDispatchMode):
    """Record the bytes the dispatched operations allocate and still hold.

    ``held`` lists, after every operation, the bytes of the tensors on
    ``device`` that operations allocated during the mode and that are still
    alive. A tensor that shares the storage of an input (a view or an in-place
    result) adds nothing; a storage counts until the tensor that first held it
    is freed.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.live = 0
        self.held: list[int] = []
        self.operations: list[str] = []
        self._tracked: dict[int, int] = {}

    def _release(self, key: int) -> None:
        self.live -= self._tracked.pop(key, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        inputs = {
            _storage_key(value)
            for value in tree_flatten((args, kwargs))[0]
            if isinstance(value, torch.Tensor)
        }
        for value in tree_flatten(out)[0]:
            if not isinstance(value, torch.Tensor) or not _on(value, self.device):
                continue
            key = _storage_key(value)
            if key is None or key in inputs or key in self._tracked:
                continue
            size = _storage_bytes(value)
            if size <= 0:
                continue
            self._tracked[key] = size
            self.live += size
            weakref.finalize(value, self._release, key)
        self.operations.append(str(func))
        self.held.append(self.live)
        return out


def batch_adaptive_modules(model: torch.nn.Module) -> list[str]:
    """Names of modules whose allocations depend non-linearly on the batch.

    These modules cap a workspace with ``workspace_mb`` (choosing smaller
    chunks for larger batches) or declare ``batch_adaptive_memory = True``,
    which a module defined outside the library sets when its allocations do
    not grow linearly with the batch size.
    """
    names = []
    for name, module in model.named_modules():
        adaptive = getattr(module, "batch_adaptive_memory", False) is True or any(
            getattr(module, attribute, None) is not None
            for attribute in ("workspace_mb", "indexed_workspace_mb")
        )
        if adaptive:
            names.append(name or type(module).__name__)
    return names


def has_forward_hooks(model: torch.nn.Module) -> bool:
    """Whether a forward or forward pre-hook would run during ``model(...)``."""
    from torch.nn.modules import module as module_lib

    for registry in ("_global_forward_hooks", "_global_forward_pre_hooks"):
        if getattr(module_lib, registry, None):
            return True
    return any(
        getattr(module, "_forward_hooks", None)
        or getattr(module, "_forward_pre_hooks", None)
        for module in model.modules()
    )


def _model_state_token(model: torch.nn.Module) -> tuple:
    """Changes whenever a parameter or buffer is replaced or modified in place."""
    return tuple(
        (id(tensor), tensor._version)
        for tensor in itertools.chain(model.parameters(), model.buffers())
    )


# model -> (state token, {(example shape, dtype, device, lengths): growth or None})
_PROBES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _accepts_seq_lengths(model: torch.nn.Module) -> bool:
    try:
        return "seq_lengths" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _probe(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    device: torch.device,
) -> _HeldTensorBytes:
    batch = inputs.to(device)
    kwargs = {}
    if seq_lengths is not None and _accepts_seq_lengths(model):
        kwargs["seq_lengths"] = seq_lengths.to(device)
    recorder = _HeldTensorBytes(device)
    with recorder:
        output = model(batch, **kwargs)
    del output
    return recorder


def _warm_up(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    device: torch.device,
) -> None:
    kwargs = {}
    if seq_lengths is not None and _accepts_seq_lengths(model):
        kwargs["seq_lengths"] = seq_lengths[: inputs.shape[0]].to(device)
    model(inputs.to(device), **kwargs)


def forward_peak_lower_bound(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    device: str | torch.device,
    seq_lengths: torch.Tensor | None = None,
    n_examples: int | None = None,
) -> int | None:
    """Lower bound, in bytes, on the tensor memory one forward pass holds at once.

    The pass is ``model`` on ``n_examples`` examples (default: all of
    ``inputs``); ``inputs`` must hold at least the first
    ``PROBE_BATCH_SIZES[-1]`` of them, on its first axis. The inputs
    themselves are not included. ``model`` must already be on ``device``. The
    probes run a copy of the model in eval mode without gradients on the first
    examples only, with every global random number generator restored
    afterwards. Returns None when no bound can be given (see the module
    documentation).
    """
    if n_examples is None:
        n_examples = int(inputs.shape[0]) if inputs.dim() > 0 else 0
    sizes = PROBE_BATCH_SIZES
    target = torch.device(device)
    if (
        tree_flatten is None
        or target.type != "cuda"
        or n_examples <= sizes[-1]
        or inputs.dim() == 0
        or inputs.shape[0] < sizes[-1]
    ):
        return None
    adaptive = batch_adaptive_modules(model)
    if adaptive:
        logger.debug(
            "No activation-memory bound: %s choose chunk sizes from the batch",
            ", ".join(adaptive[:3]),
        )
        return None
    if has_forward_hooks(model):
        logger.debug("No activation-memory bound: forward hooks are registered")
        return None
    key = (
        tuple(inputs.shape[1:]),
        str(inputs.dtype),
        str(target),
        None if seq_lengths is None else tuple(seq_lengths.shape[1:]),
    )
    token = _model_state_token(model)
    cached_token, growth_by_key = _PROBES.get(model, (None, {}))
    if cached_token != token:
        growth_by_key = {}
    if key not in growth_by_key:
        growth_by_key[key] = _probe_growth(model, inputs, seq_lengths, target)
        _PROBES[model] = (token, growth_by_key)
    growth = growth_by_key[key]
    if growth is None:
        return None
    return max(
        (held + per_step * (n_examples - sizes[-1]) // (sizes[-1] - sizes[-2]))
        for held, per_step in growth
    )


def _probe_growth(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    target: torch.device,
) -> list[tuple[int, int]] | None:
    """Held bytes after each operation at the largest probe, with their growth
    between the two largest probes; None when they are not affine in the batch.

    The probes run on a copy of the model, so state a forward pass changes
    (buffers, caches, counters) stays as it was in ``model``.
    """
    sizes = PROBE_BATCH_SIZES
    traces = []
    try:
        with torch.no_grad(), preserved_random_state():
            probe_model = copy.deepcopy(model).eval()
            # A first, untraced pass fills caches a model builds lazily, which
            # would otherwise make the first probe differ from the others.
            _warm_up(probe_model, inputs[: sizes[0]], seq_lengths, target)
            for size in sizes:
                traces.append(
                    _probe(
                        probe_model,
                        inputs[:size],
                        None if seq_lengths is None else seq_lengths[:size],
                        target,
                    )
                )
            del probe_model
    except Exception as error:  # the full pass reports its own errors
        logger.debug("No activation-memory bound: the probe failed (%s)", error)
        return None
    return _affine_growth(traces, sizes)


def _affine_growth(
    traces: Sequence[_HeldTensorBytes], sizes: Sequence[int]
) -> list[tuple[int, int]] | None:
    """Per operation: held bytes at the largest probe and their growth from the
    previous probe, when the held bytes are affine in the batch size."""
    first, middle, last = traces
    if not (first.operations == middle.operations == last.operations):
        return None
    step_a, step_b = sizes[1] - sizes[0], sizes[2] - sizes[1]
    growth = []
    for held_a, held_b, held_c in zip(first.held, middle.held, last.held):
        per_step = held_c - held_b
        # Equal growth per example between the probes, never shrinking.
        if (held_b - held_a) * step_b != per_step * step_a or per_step < 0:
            return None
        growth.append((held_c, per_step))
    return growth


def available_device_bytes(device: str | torch.device) -> int | None:
    """Memory the CUDA caching allocator can still provide on ``device``.

    The free device memory plus the memory PyTorch has reserved but not
    allocated, and no more than a per-process memory fraction
    (``torch.cuda.set_per_process_memory_fraction``) leaves. None for devices
    other than CUDA, where no bound is used.
    """
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return None
    index = target.index if target.index is not None else torch.cuda.current_device()
    try:
        free, total = torch.cuda.mem_get_info(index)
        allocated = torch.cuda.memory_allocated(index)
        cached = torch.cuda.memory_reserved(index) - allocated
        available = int(free) + max(0, int(cached))
        get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        if callable(get_fraction):
            limit = int(float(get_fraction(index)) * int(total))
            available = min(available, limit - int(allocated))
    except Exception:  # pragma: no cover - driver without memory queries
        return None
    return max(0, available)


def full_pass_cannot_fit(
    model: torch.nn.Module,
    items: Sequence[Any],
    *,
    device: str | torch.device,
    n_examples: int | None = None,
) -> tuple[bool, str]:
    """Whether one forward pass over a whole split certainly exceeds the device.

    ``items`` holds the split's tensors (inputs, labels, optional sequence
    lengths), complete or only its first examples; ``n_examples`` is the size
    of the split (default: the length of ``items``). ``model`` must already be
    on ``device``. The bound of :func:`forward_peak_lower_bound`, plus the
    bytes the split's tensors take once copied to the device, is compared
    with :func:`available_device_bytes`. Returns the decision and a short
    description for logs.
    """
    available = available_device_bytes(device)
    if available is None or not items or not isinstance(items[0], torch.Tensor):
        return False, "no device memory bound applies"
    inputs = items[0]
    if n_examples is None:
        n_examples = int(inputs.shape[0]) if inputs.dim() > 0 else 0
    seq_lengths = items[2] if len(items) > 2 else None
    if seq_lengths is not None and not isinstance(seq_lengths, torch.Tensor):
        seq_lengths = None
    activations = forward_peak_lower_bound(
        model,
        inputs,
        device=device,
        seq_lengths=seq_lengths,
        n_examples=n_examples,
    )
    if activations is None:
        return False, "the forward pass could not be bounded"
    target = torch.device(device)
    copied = 0
    for item in items:
        if isinstance(item, torch.Tensor) and item.dim() > 0 and item.shape[0]:
            if not _on(item, target):
                per_example = item.nelement() * item.element_size() // item.shape[0]
                copied += per_example * n_examples
    needed = activations + copied
    description = (
        f"the pass over {n_examples} examples needs at least "
        f"{needed / 2**30:.2f} GiB of device memory and {available / 2**30:.2f} "
        "GiB are available"
    )
    return needed > available, description


__all__ = [
    "PROBE_BATCH_SIZES",
    "available_device_bytes",
    "batch_adaptive_modules",
    "forward_peak_lower_bound",
    "full_pass_cannot_fit",
    "has_forward_hooks",
]
