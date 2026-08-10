"""Figures for the LEARNING-level gather overlay (learn_overlay.sh output).

  L1  loss trajectories, all five arms overlaid (train solid, valid dashed),
      with the loop-vs-loop_ctrl noise floor stated on the figure -- the
      receipt that the three optimized levels learn identically.
  L2  wall-clock per run, with speedup labels vs the loop baseline.
  L3  peak GPU memory per run, with reduction labels vs loop.

    python make_learn_figures.py <overlay_outdir> [--gpu "name"]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"loop": "#1f77b4", "loop_ctrl": "#9ecae1", "fused": "#e6550d",
          "triton": "#31a354", "cuda": "#e6a817"}
LABELS = {"loop": "loop (baseline)", "loop_ctrl": "loop re-run (control)",
          "fused": "fused autograd", "triton": "Triton", "cuda": "CUDA"}
ARMS = ("loop", "loop_ctrl", "fused", "triton", "cuda")
BAR_ARMS = ("loop", "fused", "triton", "cuda")


def style(ax, title, sub):
    n_sub = sub.count("\n") + 1
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left",
                 pad=18 + 14 * n_sub)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10, color="#555",
            va="bottom", linespacing=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(outdir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--gpu", default=None)
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.outdir, "overlay.json")))
    gpu = a.gpu or open(os.path.join(a.outdir, "gpu.txt")).read().strip()
    figdir = os.path.join(a.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    traj, meta, verd = d["traj"], d["meta"], d.get("verdicts", {})

    # L1 -- loss overlay. The five trajectories are numerically identical, so
    # they'd hide each other at equal width: draw with DECREASING linewidths so
    # every arm stays visible as a concentric border under the next -- the
    # perfect stacking is itself the result.
    fig, ax = plt.subplots(figsize=(9, 6.4))
    widths = {"loop": 8, "loop_ctrl": 6.2, "fused": 4.4, "triton": 2.6, "cuda": 1.2}
    for tag in ARMS:
        if tag not in traj:
            continue
        ep = range(1, len(traj[tag]["train"]) + 1)
        lw = widths.get(tag, 2)
        ax.plot(ep, traj[tag]["train"], color=COLORS[tag], lw=lw,
                label=LABELS[tag], solid_capstyle="round")
        ax.plot(ep, traj[tag]["valid"], color=COLORS[tag], lw=max(lw * 0.7, 1),
                ls="--")
    nf = verd.get("noise_floor")
    devs = [v["max_dev"] for k, v in verd.items() if isinstance(v, dict)]
    note = ""
    if nf is not None and devs:
        note = (f"run-to-run noise floor (loop vs loop rerun): {nf:.1e} · "
                f"max deviation of any optimized arm: {max(devs):.1e}")
    style(ax, "Learning is unchanged across all gather levels",
          f"{gpu} · real training runs, same config/seed · train solid, valid dashed\n{note}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(frameon=False, fontsize=10)
    save(fig, figdir, "L1_loss_overlay.png")

    # L2 -- wall-clock bars
    have = [t for t in BAR_ARMS if t in meta and meta[t].get("wall_s")]
    if have:
        wall = {t: float(meta[t]["wall_s"]) for t in have}
        fig, ax = plt.subplots(figsize=(8, 5.5))
        xs = range(len(have))
        ax.bar(xs, [wall[t] for t in have],
               color=[COLORS[t] for t in have])
        for i, t in enumerate(have):
            lab = f"{wall['loop']/wall[t]:.1f}×" if t != "loop" and wall.get("loop") else ""
            ax.text(i, wall[t], f"{wall[t]:.0f}s\n{lab}", ha="center",
                    va="bottom", fontweight="bold")
        ax.set_xticks(list(xs), [LABELS[t] for t in have])
        style(ax, "Training wall-clock, same run",
              f"{gpu} · full training run · labels: speedup vs loop")
        ax.set_ylabel("seconds")
        save(fig, figdir, "L2_wallclock.png")

    # L3 -- peak memory bars
    have = [t for t in BAR_ARMS if t in meta and meta[t].get("peak_mb")]
    if have:
        peak = {t: float(meta[t]["peak_mb"]) for t in have}
        fig, ax = plt.subplots(figsize=(8, 5.5))
        xs = range(len(have))
        ax.bar(xs, [peak[t] for t in have],
               color=[COLORS[t] for t in have])
        for i, t in enumerate(have):
            lab = f"{peak['loop']/peak[t]:.1f}× less" if t != "loop" and peak.get("loop") else ""
            ax.text(i, peak[t], f"{peak[t]:,.0f} MB\n{lab}", ha="center",
                    va="bottom", fontweight="bold")
        ax.set_xticks(list(xs), [LABELS[t] for t in have])
        style(ax, "Peak GPU memory, same run",
              f"{gpu} · nvidia-smi device peak during training · labels vs loop")
        ax.set_ylabel("MB")
        save(fig, figdir, "L3_peak_memory.png")


if __name__ == "__main__":
    main()
