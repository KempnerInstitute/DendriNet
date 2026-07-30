#!/usr/bin/env python
"""Standalone forward+backward profiler for DendriNet population-network cores.

MEASUREMENT ONLY. This script does not modify, fuse, or "fix" any kernel. It
instantiates the *core* network (the module holding every custom dendritic op)
directly from a real repo config, feeds it synthetic input of the correct shape,
and measures where forward + backward time and memory actually go.

Why the core (not the full encoder->core->decoder Model): every custom op the
profiling brief cares about -- (a) top-k synapse gather, (b) branch-local
shunting / normalized-additive divisor, (c) axial child->parent coupling
(BlockLinear), (d) E/I population pooling -- lives inside the core. The encoder
is identity for MNIST/CIFAR and the decoder is a tiny MLP; profiling the core in
isolation removes the dataloader and keeps the signal on the dendritic ops.

Two independent measurement mechanisms, cross-checked:
  1. cuda.Event around the whole forward and the whole backward  -> authoritative
     wall-clock, per iter, so we get mean + coefficient of variation (CV).
  2. cuda.Event around every submodule's forward AND backward (via hooks),
     each module classified into one of the four op families -> per-family
     fwd/bwd GPU time attribution, per iter (for CV).
  3. one torch.profiler pass (CUDA+CPU, profile_memory, with_flops, record_shapes)
     with record_function() family scopes -> self-CUDA time, memory alloc/free,
     and theoretical FLOPs per op, exported to disk.

Guardrails honored: measured wall-clock/memory are ground truth; FLOPs are
secondary. Nothing is optimized. If a config cannot be built or run (OOM, shape,
missing data) it is reported verbatim and the model is NOT shrunk silently.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext

import torch

# Dynamo config for the optional --compile pass (mirrors ei_network.py's guard).
try:
    import torch._dynamo.config as _dynamo_config
except ImportError:
    _dynamo_config = None

# --------------------------------------------------------------------------- #
# Repo imports (require PYTHONPATH=src)                                        #
# --------------------------------------------------------------------------- #
from dendritic_modeling.config import load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.factory import get_architecture


# --------------------------------------------------------------------------- #
# Op-family classification (matches the Phase-0 code audit).                   #
# A module is assigned to a family purely by its class name so we do not have  #
# to import every synapse variant.                                            #
# --------------------------------------------------------------------------- #
FAMILY_BY_CLASSNAME = {
    # (a) top-k excitatory/inhibitory synapse gather / sparse selection
    "TopKLinear": "a_topk_synapse_gather",
    "IndexedSparseLinear": "a_topk_synapse_gather",
    "IndexedDynamicLinear": "a_topk_synapse_gather",
    "IndexedRewireLinear": "a_topk_synapse_gather",
    "DenseToSparseLinear": "a_topk_synapse_gather",
    "StochasticLinear": "a_topk_synapse_gather",
    "VarianceLinear": "a_topk_synapse_gather",
    "DeepSTLinear": "a_topk_synapse_gather",
    # (c) axial child->parent coupling / branch convergence
    "BlockLinear": "c_axial_blocklinear",
    "EfficientBlockLinear": "c_axial_blocklinear",
    # (b) branch-local shunting / normalized-additive divisor
    "DendriticBranchLayer": "b_branch_shunting_divisor",
    "DendriNet": "b_branch_shunting_divisor",
    "StatefulDendriNet": "b_branch_shunting_divisor",
    # (d) E/I population pooling
    "PopulationLayer": "d_ei_pooling",
    "ExcitationInhibitionLayer": "d_ei_pooling",
}

# Substrings used as a fallback so subclasses / renamed variants still land in a
# family instead of "other".
FAMILY_BY_SUBSTR = [
    ("topk", "a_topk_synapse_gather"),
    ("indexed", "a_topk_synapse_gather"),
    ("sparse", "a_topk_synapse_gather"),
    ("synapse", "a_topk_synapse_gather"),
    ("blocklinear", "c_axial_blocklinear"),
    ("branch", "b_branch_shunting_divisor"),
    ("population", "d_ei_pooling"),
    ("excitationinhibition", "d_ei_pooling"),
]

FAMILY_LABELS = {
    "a_topk_synapse_gather": "(a) top-k synapse gather  [eager: topk+scatter+matmul]",
    "b_branch_shunting_divisor": "(b) branch shunting/additive divisor  [eager]",
    "c_axial_blocklinear": "(c) axial child->parent (BlockLinear)  [eager]",
    "d_ei_pooling": "(d) E/I population pooling  [eager: cat/sum]",
    "other": "(other modules)",
}


def classify(module: torch.nn.Module) -> str:
    name = type(module).__name__
    if name in FAMILY_BY_CLASSNAME:
        return FAMILY_BY_CLASSNAME[name]
    low = name.lower()
    for sub, fam in FAMILY_BY_SUBSTR:
        if sub in low:
            return fam
    return "other"


# --------------------------------------------------------------------------- #
# Config -> core builder                                                       #
# --------------------------------------------------------------------------- #
# input_dim is injected by the training pipeline from the (identity) encoder;
# for these configs it is just the flattened sample dimension.
DATASET_INPUT_DIM = {
    "mnist": 784,
    "fashion_mnist": 784,
    "kmnist": 784,
    "cifar10": 3072,
    "cifar100": 3072,
}


def build_core_from_config(config_path: str):
    """Return (core_module, input_dim, meta) built exactly as the factory does."""
    config = load_config(config_path)
    core_cfg = to_plain_dict(config.model.core)
    core_type = (core_cfg.get("type") or "").lower()

    # Resolve input_dim: prefer an explicit value in the config, else dataset map.
    input_dim = core_cfg.get("input_dim")
    if input_dim is None:
        pn = core_cfg.get("population_network") or {}
        input_dim = pn.get("input_dim")
    if input_dim is None:
        ds = None
        try:
            ds = to_plain_dict(config.data).get("dataset_name")
        except Exception:
            ds = None
        if ds is None:
            raise ValueError(
                f"Could not resolve input_dim for {config_path}: no input_dim in "
                f"config.model.core and no data.dataset_name to infer it from."
            )
        if ds not in DATASET_INPUT_DIM:
            raise ValueError(
                f"Unknown dataset '{ds}' for {config_path}; add it to "
                f"DATASET_INPUT_DIM (flattened input dimension)."
            )
        input_dim = DATASET_INPUT_DIM[ds]

    core = get_architecture(
        core_type, core_cfg, input_dim=int(input_dim), suffix_input_dim=None
    )
    meta = {
        "config_path": config_path,
        "core_type": core_type,
        "input_dim": int(input_dim),
        "is_recurrent": bool(getattr(core, "is_recurrent", False)),
        "n_params": int(sum(p.numel() for p in core.parameters())),
    }
    return core, int(input_dim), meta


# --------------------------------------------------------------------------- #
# Kernel toggle (MEASUREMENT-ONLY, default OFF).                               #
# The shipped default is every gather flag False (original advanced-index      #
# loop). Passing --kernels on flips the three sparse-gather flags to True on   #
# every layer that has them, so the profiler measures the OPTIMIZED path       #
# (precedence cuda -> triton -> fused). Detected by attribute presence, not by #
# importing the layer class, so it is robust to subclasses/renames. Returns    #
# the number of layers toggled, for visibility.                                #
# --------------------------------------------------------------------------- #
_GATHER_FLAGS = ("use_cuda_gather", "use_triton_gather", "use_fused_gather")

# Exactly one flag per path. `auto` reproduces the old `--kernels on` behaviour
# (all three True, so indexed_sparse's cuda -> triton -> fused precedence picks
# the fastest AVAILABLE path) -- useful for "best case" but useless for
# attributing a result to a specific implementation, which is what the A/B
# figures need. Prefer an explicit path.
_PATH_FLAGS = {
    "loop": (),
    "fused": ("use_fused_gather",),
    "triton": ("use_triton_gather",),
    "cuda": ("use_cuda_gather",),
    "auto": _GATHER_FLAGS,
}


def assert_path_available(path: str) -> None:
    """Fail loudly if the requested path would silently degrade to another.

    ``indexed_sparse.forward`` guards each accelerated path on a runtime
    availability check and falls through to the next-best one when it fails --
    great for robustness, fatal for benchmarking, because an unbuilt CUDA
    extension yields a run that is labelled "cuda" but is really the eager loop.
    (That exact mistake produced an invalid CUDA profile earlier in this work.)
    """
    if path == "cuda":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.cuda_gather import (
            cuda_gather_available,
        )

        if not cuda_gather_available():
            raise RuntimeError(
                "--path cuda requested but cuda_gather_available() is False, so "
                "the run would silently fall back to the eager loop and be "
                "mislabelled. Build the extension first:\n"
                "  source profilings/weight_transform_opt/setup_cuda_env.sh"
            )
    elif path == "triton":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.triton_gather import (
            triton_gather_available,
        )

        if not triton_gather_available():
            raise RuntimeError(
                "--path triton requested but triton_gather_available() is False, "
                "so the run would silently fall back and be mislabelled."
            )


def apply_path(core, path: str) -> dict:
    """Force exactly one gather path on every sparse layer.

    Every flag is set explicitly (not just the selected one) so the measured
    path is unambiguous regardless of what the config asked for.
    """
    on = set(_PATH_FLAGS[path])
    n = 0
    for m in core.modules():
        if all(hasattr(m, f) for f in _GATHER_FLAGS):
            for f in _GATHER_FLAGS:
                setattr(m, f, f in on)
            n += 1
    return {"path": path, "layers": n}


def freeze_rewiring(core) -> int:
    """Pin structural plasticity off so every path measures the same topology.

    The largest feedforward config uses ``topk_type: indexed_rewire``. Rewiring
    only fires from an explicit ``apply_rewiring()`` call (never from forward),
    so a raw fwd/bwd harness should not trigger it -- but if it ever did fire
    unevenly across paths the A/B would be comparing different models. Setting
    ``freeze_connectivity`` makes that guarantee explicit rather than assumed;
    ``verify_rewiring_frozen`` re-checks it after the timed loop.
    """
    n = 0
    for m in core.modules():
        if hasattr(m, "freeze_connectivity"):
            m.freeze_connectivity = True
            n += 1
    return n


def verify_rewiring_frozen(core) -> dict:
    """Confirm no rewiring happened during measurement (rewire_step still 0)."""
    steps = [
        int(m.rewire_step.item())
        for m in core.modules()
        if hasattr(m, "rewire_step")
    ]
    return {
        "rewire_layers": len(steps),
        "max_rewire_step": max(steps) if steps else 0,
        "rewiring_occurred": bool(steps and max(steps) > 0),
    }


# --------------------------------------------------------------------------- #
# torch.compile toggle (MEASUREMENT-ONLY, default OFF).                        #
# Wraps core.forward in torch.compile AFTER the gather flags are set, so the   #
# compiled graph observes the optimized path and Inductor can fuse the         #
# elementwise/reduction tail (shunting divisor, E/I pooling) between the       #
# custom gather kernels (which dynamo treats as opaque graph breaks). Mirrors  #
# the opt-in wrap in ei_network.py. Returns True if compilation was applied.   #
# --------------------------------------------------------------------------- #
def apply_compile(core, mode: str) -> bool:
    if mode != "on":
        return False
    if _dynamo_config is not None:
        _dynamo_config.recompile_limit = 100
        _dynamo_config.force_parameter_static_shapes = False
    core.forward = torch.compile(core.forward, dynamic=True)
    return True


# --------------------------------------------------------------------------- #
# Per-module cuda.Event timing via hooks.                                      #
# Events are recorded on the default stream during the timed loop; we sync ONCE #
# at the end and reduce elapsed_time. Each event pair is tagged (iter, family). #
# --------------------------------------------------------------------------- #
class ModuleTimer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.cur_iter = -1
        self.recording = False
        # events[iter][family][phase] -> list[(start_event, end_event)]
        self.events = defaultdict(
            lambda: defaultdict(lambda: {"fwd": [], "bwd": []})
        )
        self._fwd_start = {}  # id(module) -> start event (this call)
        self._bwd_start = {}
        self._handles = []
        self._families = {}
        self._register(model)

    def _new_event(self):
        return torch.cuda.Event(enable_timing=True)

    def _register(self, model):
        for m in model.modules():
            fam = classify(m)
            if fam == "other":
                # Still time "other" leaf-ish modules? Skip container-only noise:
                # we only time modules that actually do compute in a family, plus
                # keep "other" for anything unclassified that has parameters.
                if not any(True for _ in m.parameters(recurse=False)):
                    continue
            self._families[id(m)] = fam
            self._handles.append(
                m.register_forward_pre_hook(self._fwd_pre)
            )
            self._handles.append(m.register_forward_hook(self._fwd_post))
            self._handles.append(
                m.register_full_backward_pre_hook(self._bwd_pre)
            )
            self._handles.append(
                m.register_full_backward_hook(self._bwd_post)
            )

    # --- forward ---
    def _fwd_pre(self, module, inputs):
        if not self.recording:
            return
        ev = self._new_event()
        ev.record()
        self._fwd_start[id(module)] = ev

    def _fwd_post(self, module, inputs, output):
        if not self.recording:
            return
        start = self._fwd_start.pop(id(module), None)
        if start is None:
            return
        end = self._new_event()
        end.record()
        fam = self._families.get(id(module), "other")
        self.events[self.cur_iter][fam]["fwd"].append((start, end))

    # --- backward ---
    def _bwd_pre(self, module, grad_output):
        if not self.recording:
            return
        ev = self._new_event()
        ev.record()
        self._bwd_start[id(module)] = ev

    def _bwd_post(self, module, grad_input, grad_output):
        if not self.recording:
            return
        start = self._bwd_start.pop(id(module), None)
        if start is None:
            return
        end = self._new_event()
        end.record()
        fam = self._families.get(id(module), "other")
        self.events[self.cur_iter][fam]["bwd"].append((start, end))

    def remove(self):
        for h in self._handles:
            h.remove()

    def reduce(self):
        """After sync: per-iter per-family summed ms. Returns dict family ->
        {'fwd': [per-iter ms...], 'bwd': [per-iter ms...]}."""
        torch.cuda.synchronize()
        out = defaultdict(lambda: {"fwd": [], "bwd": []})
        for it in sorted(self.events):
            fam_ms = defaultdict(lambda: {"fwd": 0.0, "bwd": 0.0})
            for fam, phases in self.events[it].items():
                for phase in ("fwd", "bwd"):
                    total = 0.0
                    for (s, e) in phases[phase]:
                        total += s.elapsed_time(e)  # ms
                    fam_ms[fam][phase] = total
            # push this iter's totals
            fams = set(fam_ms) | set(out)
            for fam in fams:
                out[fam]["fwd"].append(fam_ms[fam]["fwd"])
                out[fam]["bwd"].append(fam_ms[fam]["bwd"])
        return out


def _attr(e, *names, default=0):
    """torch renamed profiler attrs (self_cuda_time_total -> self_device_time_total
    in 2.8+). Try each name; return first present & non-None."""
    for n in names:
        v = getattr(e, n, None)
        if v is not None:
            return v
    return default


def ev_self_cuda_us(e):
    return float(_attr(e, "self_device_time_total", "self_cuda_time_total"))


def ev_cuda_us(e):
    return float(_attr(e, "device_time_total", "cuda_time_total"))


def ev_self_cuda_mem(e):
    return float(_attr(e, "self_device_memory_usage", "self_cuda_memory_usage"))


def ev_cuda_mem(e):
    return float(_attr(e, "device_memory_usage", "cuda_memory_usage"))


def cv(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = statistics.mean(xs)
    if m == 0:
        return 0.0
    return statistics.pstdev(xs) / m


# --------------------------------------------------------------------------- #
# One profiling run for a single (config, batch) shape.                        #
# --------------------------------------------------------------------------- #
class _SkipProfile(Exception):
    """Internal: signals the profiler passes were deliberately skipped."""


def run_one(config_path, batch, warmup, iters, outdir, device, kernels="off",
            compile_mode="off", path=None, freeze_rewire=True, prebuilt=None,
            do_profile=True):
    """Measure one (config, path, batch) cell.

    ``prebuilt`` lets the caller construct the core ONCE and reuse it across the
    batch sweep. Construction is dominated by index sampling and scales with
    contact count -- 77.6 s for the 116M-contact stress config -- so rebuilding
    per cell would spend ~20 min per sweep doing nothing but resampling an
    identical topology. The core is stateless between cells here (grads are
    zeroed, a fresh input is drawn, and the gather path / rewire pin are
    re-applied), so reuse does not change what is measured. Not used when
    ``compile_mode == "on"``, since that permanently rebinds ``core.forward``.

    ``do_profile`` gates the expensive torch.profiler + split-fwd/bwd passes.
    Only the per-op figure needs them, and only at one batch; the other figures
    need just wall-clock and memory.
    """
    result = {"config": config_path, "batch": batch}
    if prebuilt is not None:
        core, input_dim, meta = prebuilt
    else:
        try:
            core, input_dim, meta = build_core_from_config(config_path)
        except Exception as e:
            result["status"] = "BUILD_FAILED"
            result["error"] = f"{type(e).__name__}: {e}"
            result["traceback"] = traceback.format_exc()
            return result

    result.update(meta)
    core = core.to(device).train()
    # `--path` is the precise selector; `--kernels on/off` is kept as a legacy
    # alias (on -> auto, off -> loop) so old invocations keep working.
    resolved_path = path if path is not None else (
        "auto" if kernels == "on" else "loop"
    )
    info = apply_path(core, resolved_path)
    result["path"] = resolved_path
    result["path_layers"] = info["layers"]
    result["kernels"] = kernels  # legacy field, kept for older readers
    result["kernels_layers_on"] = info["layers"] if resolved_path != "loop" else 0
    if freeze_rewire:
        result["rewire_layers_frozen"] = freeze_rewiring(core)
    result["compile"] = compile_mode

    # Synthetic input: standard-normal, realistic magnitude. Non-negative input
    # is enforced by some EI networks; population cores accept signed input, but
    # to be safe we use abs() so the non-neg guard (if any) does not trip.
    x = torch.randn(batch, input_dim, device=device).abs().requires_grad_(True)

    def step():
        out = core(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = out.float().pow(2).mean()
        return out, loss

    # ---- optional torch.compile + inline correctness check ----
    # Capture an eager reference BEFORE compiling, then compare the first
    # compiled output so a compile that silently changes numerics is caught
    # (not just a speedup measurement). The compile-triggering call happens
    # here, outside the timed loop.
    if compile_mode == "on":
        try:
            with torch.no_grad():
                ref_out = step()[0].detach().float().clone()
            apply_compile(core, compile_mode)
            with torch.no_grad():
                cmp_out = step()[0].detach().float()
            result["compile_equiv_max_abs"] = (ref_out - cmp_out).abs().max().item()
            del ref_out, cmp_out
        except Exception as e:
            result["status"] = "COMPILE_FAILED"
            result["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            result["traceback"] = traceback.format_exc()
            del core, x
            gc.collect()
            torch.cuda.empty_cache()
            return result

    # ---- warmup (also catches shape/OOM before timing) ----
    try:
        for _ in range(warmup):
            core.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            out, loss = step()
            loss.backward()
        torch.cuda.synchronize()
    except RuntimeError as e:
        msg = str(e)
        result["status"] = "OOM" if "out of memory" in msg.lower() else "RUNTIME_ERROR"
        result["error"] = f"{type(e).__name__}: {msg.splitlines()[0]}"
        result["traceback"] = traceback.format_exc()
        del core, x
        gc.collect()
        torch.cuda.empty_cache()
        return result
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        return result

    out_shape = tuple(step()[0].shape)
    result["output_shape"] = out_shape

    # ---- timed loop: whole-step fwd/bwd cuda.Event + per-module timer ----
    timer = ModuleTimer(core)
    fwd_events, bwd_events = [], []  # (start,end) per iter
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    timer.recording = True
    for it in range(iters):
        timer.cur_iter = it
        core.zero_grad(set_to_none=True)
        x.grad = None

        f0, f1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        b0, b1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        f0.record()
        out, loss = step()
        f1.record()

        b0.record()
        loss.backward()
        b1.record()

        fwd_events.append((f0, f1))
        bwd_events.append((b0, b1))
    torch.cuda.synchronize()
    timer.recording = False

    peak_mem = torch.cuda.max_memory_allocated(device)
    fwd_ms = [s.elapsed_time(e) for s, e in fwd_events]
    bwd_ms = [s.elapsed_time(e) for s, e in bwd_events]
    per_family = timer.reduce()
    timer.remove()

    result["status"] = "OK"
    result["whole_step"] = {
        "fwd_ms_mean": statistics.mean(fwd_ms),
        "fwd_ms_cv": cv(fwd_ms),
        "bwd_ms_mean": statistics.mean(bwd_ms),
        "bwd_ms_cv": cv(bwd_ms),
        "step_ms_mean": statistics.mean(fwd_ms) + statistics.mean(bwd_ms),
    }
    result["peak_mem_mb"] = peak_mem / 1e6
    result["activation_mem_mb"] = (peak_mem - mem_before) / 1e6
    result["param_mem_mb"] = mem_before / 1e6
    # Assumption, not hope: confirm structural plasticity stayed off, so every
    # path in the A/B measured the identical topology.
    result["rewiring"] = verify_rewiring_frozen(core)

    # per-family mean ms + CV + % of step
    step_total = result["whole_step"]["step_ms_mean"]
    fam_report = {}
    for fam, d in per_family.items():
        fmean = statistics.mean(d["fwd"]) if d["fwd"] else 0.0
        bmean = statistics.mean(d["bwd"]) if d["bwd"] else 0.0
        fam_report[fam] = {
            "label": FAMILY_LABELS.get(fam, fam),
            "fwd_ms_mean": fmean,
            "bwd_ms_mean": bmean,
            "fwd_cv": cv(d["fwd"]),
            "bwd_cv": cv(d["bwd"]),
            "pct_fwd": 100 * fmean / step_total if step_total else 0.0,
            "pct_bwd": 100 * bmean / step_total if step_total else 0.0,
            "pct_step": 100 * (fmean + bmean) / step_total if step_total else 0.0,
        }
    result["families"] = fam_report

    # ---- torch.profiler pass (few iters) for self-CUDA / mem / FLOPs ----
    # Gated: only the per-op breakdown figure needs these, and only at one
    # batch. They are also the slowest part of a cell (profiler overhead).
    result["profiled"] = bool(do_profile)
    try:
        if not do_profile:
            raise _SkipProfile()
        result["profiler"] = profiler_pass(
            core, step, outdir, config_path, batch, path=resolved_path
        )
    except _SkipProfile:
        pass
    except Exception as e:
        result["profiler_error"] = f"{type(e).__name__}: {e}"

    # ---- clean forward-only / backward-only per-op split ----
    try:
        if not do_profile:
            raise _SkipProfile()
        result["split_fb"] = split_fb_profile(
            core, step, outdir, config_path, batch, path=resolved_path
        )
    except _SkipProfile:
        pass
    except Exception as e:
        result["split_fb_error"] = f"{type(e).__name__}: {e}"

    del x
    if prebuilt is None:
        del core
    gc.collect()
    torch.cuda.empty_cache()
    return result


def split_fb_profile(core, step, outdir, config_path, batch, n=8, path="loop"):
    """Two separate profiler passes so per-op self-CUDA is cleanly attributable
    to forward vs backward (no scope-nesting ambiguity). Returns dict with
    'forward' and 'backward' top-op lists."""
    from torch.profiler import ProfilerActivity, profile

    tag = f"{os.path.basename(config_path).replace('.yaml','')}_{path}_b{batch}"
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    def topops(ka, limit=30):
        tot = sum(max(ev_self_cuda_us(e), 0) for e in ka) or 1
        rows = []
        for e in sorted(ka, key=ev_self_cuda_us, reverse=True):
            sc = ev_self_cuda_us(e)
            if sc <= 0:
                continue
            rows.append({
                "op": e.key,
                "count": int(e.count),
                "self_cuda_us": sc,
                "pct_self_cuda": 100.0 * sc / tot,
                "flops": float(getattr(e, "flops", 0) or 0),
                "self_cuda_mem_b": ev_self_cuda_mem(e),
            })
            if len(rows) >= limit:
                break
        return {"total_self_cuda_us": tot, "ops": rows}

    # ---- forward-only ----
    with profile(activities=acts, record_shapes=False, profile_memory=True,
                 with_flops=True) as pf:
        for _ in range(n):
            core.zero_grad(set_to_none=True)
            out, loss = step()
            del out, loss
        torch.cuda.synchronize()
    fwd = topops(pf.key_averages())

    # ---- backward-only: forward ONCE outside the profiler, then profile only
    # repeated backward on the retained graph, so no forward op is recorded. ----
    core.zero_grad(set_to_none=True)
    out, loss = step()
    torch.cuda.synchronize()
    with profile(activities=acts, record_shapes=False, profile_memory=True,
                 with_flops=True) as pb:
        for _ in range(n):
            loss.backward(retain_graph=True)
        torch.cuda.synchronize()
    bwd = topops(pb.key_averages())
    del out, loss

    with open(os.path.join(outdir, f"splitfb_{tag}.json"), "w") as f:
        json.dump({"forward": fwd, "backward": bwd}, f, indent=2)
    return {"forward": fwd, "backward": bwd}


def profiler_pass(core, step, outdir, config_path, batch, path="loop"):
    from torch.profiler import ProfilerActivity, profile, record_function

    tag = f"{os.path.basename(config_path).replace('.yaml','')}_{path}_b{batch}"
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    n = 6
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=False,
    ) as prof:
        for _ in range(n):
            core.zero_grad(set_to_none=True)
            with record_function("## FORWARD ##"):
                out, loss = step()
            with record_function("## BACKWARD ##"):
                loss.backward()
        torch.cuda.synchronize()

    # Export raw trace (opt-in; large + slow for index-heavy backward) + table.
    trace_path = None
    if os.environ.get("DENDRINET_TRACE"):
        trace_path = os.path.join(outdir, f"trace_{tag}.json")
        try:
            prof.export_chrome_trace(trace_path)
        except Exception:
            trace_path = None

    ka = prof.key_averages()
    try:
        table = ka.table(sort_by="self_device_time_total", row_limit=40)
    except Exception:
        table = ka.table(sort_by="self_cuda_time_total", row_limit=40)
    table_path = os.path.join(outdir, f"ops_{tag}.txt")
    with open(table_path, "w") as f:
        f.write(table)

    # Programmatic top-op list (name, self_cuda_us, cuda_us, cpu_us, flops, mem).
    tot_self_cuda = sum(max(ev_self_cuda_us(e), 0) for e in ka) or 1
    top = []
    for e in sorted(ka, key=ev_self_cuda_us, reverse=True)[:25]:
        top.append(
            {
                "op": e.key,
                "count": int(e.count),
                "self_cuda_us": ev_self_cuda_us(e),
                "cuda_us": ev_cuda_us(e),
                "cpu_us": float(_attr(e, "self_cpu_time_total", "cpu_time_total")),
                "pct_self_cuda": 100.0 * ev_self_cuda_us(e) / tot_self_cuda,
                "flops": float(getattr(e, "flops", 0) or 0),
                "cuda_mem_b": ev_cuda_mem(e),
                "self_cuda_mem_b": ev_self_cuda_mem(e),
            }
        )
    return {"trace": trace_path, "table_file": table_path, "top_ops": top, "n_iters": n}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/examples/population_network_local_learning_mnist.yaml",
            "configs/examples/population_network_two_layer.yaml",
            "configs/examples/population_network_cifar10.yaml",
        ],
    )
    ap.add_argument("--batches", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--path", choices=list(_PATH_FLAGS), default=None,
                    help="Which gather implementation to measure, exactly one: "
                         "loop (shipped advanced-index), fused (custom autograd), "
                         "triton, cuda, or auto (all flags on -> cuda>triton>fused "
                         "precedence picks the fastest AVAILABLE, so it cannot be "
                         "attributed to one implementation). Fails loudly if the "
                         "requested path is unavailable rather than silently "
                         "falling back. Overrides --kernels.")
    ap.add_argument("--kernels", choices=["off", "on"], default="off",
                    help="LEGACY alias for --path (off->loop, on->auto). Prefer "
                         "--path; this cannot isolate triton or fused.")
    ap.add_argument("--no-freeze-rewire", dest="freeze_rewire",
                    action="store_false",
                    help="Do NOT pin freeze_connectivity. Default is to pin it so "
                         "every path measures the identical topology; the harness "
                         "verifies rewire_step stayed 0 either way.")
    ap.add_argument("--compile", dest="compile_mode", choices=["off", "on"],
                    default="off",
                    help="off (default) = eager; on = wrap core.forward in "
                         "torch.compile AFTER kernels are set, to fuse the "
                         "elementwise/reduction tail. Reports output equivalence "
                         "vs eager.")
    ap.add_argument("--profile-all-batches", action="store_true",
                    help="Run the torch.profiler + split-fwd/bwd passes at EVERY "
                         "batch. Default runs them only at the largest batch, "
                         "since only the per-op figure needs them and they are "
                         "the slowest part of a cell.")
    ap.add_argument("--outdir", default="profilings/dendrinet_harness")
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="Build + 1 fwd/bwd on CPU with tiny batch; no timing. "
                         "Validates the API without a GPU.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.cpu_smoke:
        device = torch.device("cpu")
        print("[smoke] CPU build+forward+backward validation (no timing)\n")
        ok = True
        for cfg in args.configs:
            try:
                core, in_dim, meta = build_core_from_config(cfg)
                core = core.to(device).train()
                x = torch.randn(4, in_dim).abs().requires_grad_(True)
                out = core(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                loss = out.float().pow(2).mean()
                loss.backward()
                fams = defaultdict(int)
                for m in core.modules():
                    fams[classify(m)] += 1
                print(f"[smoke] OK  {os.path.basename(cfg)}")
                print(f"        input_dim={in_dim} out={tuple(out.shape)} "
                      f"recurrent={meta['is_recurrent']} params={meta['n_params']:,}")
                print(f"        module families: "
                      + ", ".join(f"{k}={v}" for k, v in sorted(fams.items())))
            except Exception as e:
                ok = False
                print(f"[smoke] FAILED {os.path.basename(cfg)}: {type(e).__name__}: {e}")
                traceback.print_exc()
            print()
        sys.exit(0 if ok else 1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in this process. Run on a GPU node "
              "(sbatch jobs/profile_dendrinet.sbatch). Refusing to fake GPU numbers.")
        sys.exit(2)

    device = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    resolved_path = args.path if args.path is not None else (
        "auto" if args.kernels == "on" else "loop"
    )
    # Check availability ONCE, before any timing, so an unbuildable path fails in
    # seconds instead of producing a full run of mislabelled numbers.
    assert_path_available(resolved_path)
    print(f"# GPU: {p.name} | {p.total_memory/1e9:.1f} GB | sm_{p.major}{p.minor}")
    print(f"# torch {torch.__version__} cuda {torch.version.cuda}")
    print(f"# path={resolved_path.upper()} (verified available) "
          f"freeze_rewire={args.freeze_rewire}")
    print(f"# warmup={args.warmup} iters={args.iters} batches={args.batches} "
          f"compile={args.compile_mode.upper()}\n")

    # Which batches get the expensive torch.profiler + split-fwd/bwd passes.
    # Default: the largest only -- just the per-op figure needs them.
    prof_batches = (
        set(args.batches) if args.profile_all_batches
        else ({max(args.batches)} if args.batches else set())
    )
    print(f"# per-op profiling at batch(es): {sorted(prof_batches)} "
          f"(others: timing + memory only)\n")

    all_results = []
    for cfg in args.configs:
        # Build ONCE per config and reuse across the batch sweep; construction is
        # index sampling and costs ~78 s on the 116M-contact stress config, so
        # per-cell rebuilds would burn ~20 min per sweep resampling an identical
        # topology. torch.compile permanently rebinds core.forward, so that mode
        # keeps the old rebuild-per-cell behaviour.
        prebuilt = None
        if args.compile_mode != "on":
            t0 = time.perf_counter()
            try:
                core_b, in_dim_b, meta_b = build_core_from_config(cfg)
                prebuilt = (core_b.to(device).train(), in_dim_b, meta_b)
                print(f"# built {os.path.basename(cfg)} once in "
                      f"{time.perf_counter() - t0:.1f}s, reused for "
                      f"{len(args.batches)} batches")
            except Exception as e:
                print(f"# BUILD FAILED {os.path.basename(cfg)}: "
                      f"{type(e).__name__}: {e}")
        for b in args.batches:
            print(f"=== {os.path.basename(cfg)}  batch={b}  "
                  f"[path {resolved_path}, compile {args.compile_mode}] ===")
            r = run_one(cfg, b, args.warmup, args.iters, args.outdir, device,
                        kernels=args.kernels, compile_mode=args.compile_mode,
                        path=resolved_path, freeze_rewire=args.freeze_rewire,
                        prebuilt=prebuilt, do_profile=(b in prof_batches))
            all_results.append(r)
            print_result(r)
            print()
        if prebuilt is not None:
            del prebuilt, core_b
            gc.collect()
            torch.cuda.empty_cache()

    out_json = os.path.join(args.outdir, f"results_{resolved_path}.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n# wrote {out_json}")


def print_result(r):
    st = r.get("status")
    if st != "OK":
        print(f"  STATUS: {st}")
        if "error" in r:
            print(f"  {r['error']}")
        return
    ws = r["whole_step"]
    kn = r.get("kernels", "off")
    cm = r.get("compile", "off")
    line = f"  kernels={kn.upper()} ({r.get('kernels_layers_on', 0)} gather layers on)  compile={cm.upper()}"
    if "compile_equiv_max_abs" in r:
        line += f"  [compile vs eager max|Δ|={r['compile_equiv_max_abs']:.2e}]"
    print(line)
    print(f"  params={r['n_params']:,} out={r['output_shape']} recurrent={r['is_recurrent']}")
    print(f"  whole-step: fwd {ws['fwd_ms_mean']:.3f} ms (CV {ws['fwd_ms_cv']*100:.1f}%)  "
          f"bwd {ws['bwd_ms_mean']:.3f} ms (CV {ws['bwd_ms_cv']*100:.1f}%)  "
          f"step {ws['step_ms_mean']:.3f} ms")
    print(f"  peak_mem={r['peak_mem_mb']:.1f} MB  activation_mem={r['activation_mem_mb']:.1f} MB")
    print(f"  {'family':<48}{'%fwd':>7}{'%bwd':>7}{'%step':>7}  CV(f/b)")
    for fam in sorted(r["families"], key=lambda k: -r["families"][k]["pct_step"]):
        d = r["families"][fam]
        flag = "  <== HIGH CV" if (d["fwd_cv"] > 0.05 or d["bwd_cv"] > 0.05) else ""
        print(f"  {d['label']:<48}{d['pct_fwd']:>6.1f}%{d['pct_bwd']:>6.1f}%"
              f"{d['pct_step']:>6.1f}%  {d['fwd_cv']*100:.0f}/{d['bwd_cv']*100:.0f}%{flag}")


if __name__ == "__main__":
    main()
