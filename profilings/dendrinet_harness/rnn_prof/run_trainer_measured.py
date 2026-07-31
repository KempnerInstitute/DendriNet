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


def main() -> int:
    args = T._parse_cli_args()
    _instrument()
    torch.cuda.reset_peak_memory_stats()
    rc = 0
    try:
        T.main(config_path=args.config, args=args)
    except SystemExit as e:                      # the trainer may exit cleanly
        rc = int(e.code or 0)
    finally:
        torch.cuda.synchronize()
        print(f"TORCH_PEAK_ALLOCATED_MB={torch.cuda.max_memory_allocated() / 2**20:.1f}",
              flush=True)
        print(f"TORCH_PEAK_RESERVED_MB={torch.cuda.max_memory_reserved() / 2**20:.1f}",
              flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
