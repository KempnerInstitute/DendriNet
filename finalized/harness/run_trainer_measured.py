"""Run the real trainer and report torch's OWN peak-memory counter.

`nvidia-smi` polling measures the whole process: CUDA context, cuBLAS
workspaces, and -- dominant here -- the caching allocator's RESERVED pool, which
grows to its high-water mark and is not returned to the driver. On the first real
training A/B it reported 85,107 vs 77,823 MB (1.09x) for a config the profiling
harness measures at 14.4 GB. That gap is allocator caching, not tensors, so
nvidia-smi cannot answer whether the 2.19x survives a real training step.

`torch.cuda.max_memory_allocated()` counts live tensor bytes, which is the same
quantity every other figure in this work reports. It is only reachable from
inside the process, hence this wrapper: it calls the trainer's own `main()` --
no reimplementation of the training loop -- and prints the counter afterwards.

    python run_trainer_measured.py <config> --output_dir ... --run_name ...
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from dendritic_modeling.scripts.training import train_experiments as T  # noqa: E402


def _instrument():
    """Report peak allocated at phase boundaries.

    eager and fused differ by only 6.5 MB of peak under the real trainer while
    the profiling harness shows 2x, which means the peak is NOT set inside the
    reactivation-bearing forward/backward. This localises which phase does set
    it, by resetting and reading torch's counter around each one.
    """
    from dendritic_modeling.training.strategies import standard as S

    def wrap(cls, name, label):
        fn = getattr(cls, name, None)
        if fn is None:
            return False

        def inner(self, *a, **k):
            torch.cuda.synchronize()
            before = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
            try:
                return fn(self, *a, **k)
            finally:
                torch.cuda.synchronize()
                print(f"PHASE_PEAK {label}: during={torch.cuda.max_memory_allocated()/2**20:,.1f} MB"
                      f"  (prior_max={before:,.1f})", flush=True)
        setattr(cls, name, inner)
        return True

    for cls_name in dir(S):
        cls = getattr(S, cls_name)
        if not isinstance(cls, type):
            continue
        for meth, label in (("_epoch_train", "train_epoch"),
                            ("_epoch_valid", "validation"),
                            ("_epoch_valid_stats", "valid_stats")):
            if wrap(cls, meth, f"{cls_name}.{label}"):
                print(f"  instrumented {cls_name}.{meth}", flush=True)


def _report_peak_from_trace() -> None:
    """Replay the allocator event trace; attribute the live set AT the peak.

    The phase wrappers localised the training step (9.6 GB) and validation
    (7.5 GB), yet the process peaks at 59.6 GB -- so the spike is outside every
    wrapped phase, and wrapping more functions is guessing. The trace replay
    (same method that cracked the transient gap) answers with a file:line.

    The history buffer is a ring: if the spike is late in the run (it is --
    the counter measured from the LAST phase's entry still reaches 59.6 GB),
    its events are retained even though early-training events are evicted.
    """
    import collections
    import os as _os

    snap = torch.cuda.memory._snapshot()
    events = (snap.get("device_traces") or [[]])[0]
    if not events:
        print("PEAK_TRACE: no events captured", flush=True)
        return
    live, total, best_total, best_live, best_i = {}, 0, -1, None, 0
    for i, e in enumerate(events):
        act = e.get("action")
        if act == "alloc":
            live[e["addr"]] = (e["size"], e.get("frames"))
            total += e["size"]
            if total > best_total:
                best_total, best_live, best_i = total, dict(live), i
        elif act in ("free_completed", "free_requested"):
            got = live.pop(e["addr"], None)
            if got is not None:
                total -= got[0]
    if best_live is None:
        print("PEAK_TRACE: no alloc events", flush=True)
        return
    by_site = collections.Counter()
    for size, frames in best_live.values():
        f = next((q for q in (frames or [])
                  if "dendritic_modeling" in (q.get("filename") or "")),
                 (frames or [{}])[0])
        by_site[f"{_os.path.basename(f.get('filename', '?'))}:"
                f"{f.get('line', '?')}  {f.get('name', '?')}"] += size
    print(f"PEAK_TRACE: {len(events):,} events, replay-peak "
          f"{best_total / 2**20:,.1f} MB at event {best_i:,}", flush=True)
    for k, sz in by_site.most_common(12):
        print(f"PEAK_TRACE:{sz / 2**20:>11,.1f} MB  {k[:70]}", flush=True)


def main() -> int:
    args = T._parse_cli_args()
    _instrument()
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=6_000_000
        )
    except TypeError:
        torch.cuda.memory._record_memory_history(max_entries=6_000_000)
    rc = 0
    try:
        T.main(config_path=args.config, args=args)
    except SystemExit as e:                      # the trainer may exit cleanly
        rc = int(e.code or 0)
    finally:
        torch.cuda.synchronize()
        _report_peak_from_trace()
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"TORCH_PEAK_ALLOCATED_MB={torch.cuda.max_memory_allocated() / 2**20:.1f}",
              flush=True)
        print(f"TORCH_PEAK_RESERVED_MB={torch.cuda.max_memory_reserved() / 2**20:.1f}",
              flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
