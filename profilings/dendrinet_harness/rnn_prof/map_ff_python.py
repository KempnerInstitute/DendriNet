"""Map the Python that actually runs on a feedforward training step.

The repo has ~488 Python files; almost all of them are analysis/ and never run
during training. Guessing which ones matter from directory names is how you end
up optimizing something that is never called. This measures it instead, on one
warmed forward+backward of a real config.

Three views, because they answer different questions:

  [1] FILES   -- which dendritic_modeling files execute per step at all. This is
      the inventory: the search space for any Python-level rewrite.

  [2] CALLS   -- Python functions ranked by call count. In a launch-bound regime
      (batch-1 RNN: 31-37% GPU busy across 13k kernels) call count, not CPU
      time, is the thing that hurts.

  [3] OPS     -- every aten op counted and attributed to the dendritic_modeling
      SOURCE LINE that issued it. This is the launch map: one aten op is
      approximately one kernel launch, so this ranks lines by how much work they
      hand the GPU. It is the view that found the sigmoid identity.

CPU time here is not GPU time -- CUDA is async, so a line that launches a huge
kernel looks cheap. Read [1] and [2] for structure and [3] for op count; use
scale_diff.py / ncu for actual kernel cost.

    python map_ff_python.py --config <cfg.yaml> --path cuda --batch 128
"""

from __future__ import annotations

import argparse
import cProfile
import collections
import importlib.util
import pstats
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

PKG = "dendritic_modeling"


def _load_profiler():
    spec = importlib.util.spec_from_file_location("pd", "profile_dendrinet.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["pd"] = pd
    spec.loader.exec_module(pd)
    return pd


def _short(path: str) -> str:
    i = path.find(PKG)
    return path[i + len(PKG) + 1:] if i >= 0 else path


# --------------------------------------------------------------------------- #
# [3] aten op attribution                                                      #
# --------------------------------------------------------------------------- #
class OpMap(TorchDispatchMode):
    """Count aten ops and blame the innermost dendritic_modeling frame.

    During backward the Python stack is the autograd engine, not the module that
    built the graph, so those ops are bucketed separately rather than
    misattributed to whatever happened to be on the stack.
    """

    def __init__(self):
        self.by_line = collections.Counter()
        self.by_op = collections.Counter()
        self.unattributed = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        frame, site = sys._getframe(1), None
        for _ in range(40):
            if frame is None:
                break
            fn = frame.f_code.co_filename
            if PKG in fn and "rnn_prof" not in fn:
                site = f"{_short(fn)}:{frame.f_lineno}  {frame.f_code.co_name}()"
                break
            frame = frame.f_back
        if site is None:
            self.unattributed += 1
        else:
            self.by_line[site] += 1
        self.by_op[str(func).split(".")[-2] if "." in str(func) else str(func)] += 1
        return func(*args, **(kwargs or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1

    pd = _load_profiler()
    core, dim, _ = pd.build_core_from_config(a.config)
    pd.assert_path_available(a.path)
    pd.apply_path(core, a.path)
    pd.freeze_rewiring(core)
    core = core.to("cuda").train()
    x = torch.randn(a.batch, dim, device="cuda").abs().requires_grad_(True)

    def step():
        core.zero_grad(set_to_none=True)
        out = core(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out.float().pow(2).mean().backward()

    for _ in range(3):          # warm up: build caches, autotune, allocate
        step()
    torch.cuda.synchronize()

    print(f"GPU {torch.cuda.get_device_name(0)}")
    print(f"config {a.config}\npath={a.path}  batch={a.batch}  "
          f"(op counts are batch-independent)\n")

    # ---------------- [1] + [2] ------------------------------------------- #
    prof = cProfile.Profile()
    prof.enable()
    step()
    torch.cuda.synchronize()
    prof.disable()

    st = pstats.Stats(prof)
    per_file = collections.defaultdict(lambda: [0, 0, 0.0])  # funcs, calls, cum
    per_func = []
    for (fn, lineno, name), (_cc, nc, tt, ct, _cal) in st.stats.items():
        if PKG not in fn or "rnn_prof" in fn:
            continue
        rec = per_file[_short(fn)]
        rec[0] += 1
        rec[1] += nc
        rec[2] += tt
        per_func.append((nc, tt, f"{_short(fn)}:{lineno}  {name}()"))

    print("=" * 86)
    print("[1] FILES THAT EXECUTE ON ONE TRAINING STEP  "
          f"({len(per_file)} of 488 in the repo)")
    print("=" * 86)
    print(f"{'calls':>10}{'funcs':>7}{'self s':>9}  file")
    for f, (nf, nc, tt) in sorted(per_file.items(), key=lambda kv: -kv[1][1]):
        print(f"{nc:>10,}{nf:>7}{tt:>9.3f}  {f}")

    print("\n" + "=" * 86)
    print(f"[2] PYTHON FUNCTIONS BY CALL COUNT (top {a.top})")
    print("=" * 86)
    print(f"{'calls':>10}{'self s':>9}  site")
    for nc, tt, site in sorted(per_func, reverse=True)[:a.top]:
        print(f"{nc:>10,}{tt:>9.3f}  {site}")

    # ---------------- [3] -------------------------------------------------- #
    with OpMap() as om:
        step()
        torch.cuda.synchronize()

    total = sum(om.by_line.values()) + om.unattributed
    print("\n" + "=" * 86)
    print(f"[3] ATEN OPS PER STEP: {total:,} total  "
          f"({om.unattributed:,} issued from the autograd engine)")
    print("=" * 86)
    print(f"{'ops':>10}{'share':>8}  issuing source line")
    for site, n in om.by_line.most_common(a.top):
        print(f"{n:>10,}{100 * n / total:>7.1f}%  {site}")

    print(f"\n{'ops':>10}{'share':>8}  most-issued aten ops")
    for op, n in om.by_op.most_common(15):
        print(f"{n:>10,}{100 * n / total:>7.1f}%  {op}")

    print("\n  [3] is the launch map: ~1 aten op = 1 kernel launch. A line high")
    print("  here is cheap to fix and pays most where launches dominate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
