#!/usr/bin/env python
"""Six figures: what the reactivation/shunt work bought across batch 128-4096.

Three panels per GPU, drawn by the same code so the two sets are
panel-for-panel comparable:

  B1 / A1  peak memory vs batch, with the card's capacity drawn in -- this is the
           figure that shows the OOM frontier, i.e. which modes can run at all
  B2 / A2  step time vs batch -- the check that none of the memory was bought
           with speed
  B3 / A3  memory conserved vs eager -- the headline quantity, in GB

Design constraints (dataviz skill):
  * categorical hues in FIXED order (eager, identity, fused-act, fused-shunt),
    never cycled, so a mode keeps its colour across all six panels and both GPUs;
  * legend present AND all four series direct-labelled, so identity is never
    carried by colour alone;
  * single y-axis everywhere; no dual axes;
  * recessive grid/axes, thin marks, text in ink tokens rather than series colour;
  * OOM is drawn, not omitted -- a hollow marker on the capacity line at the batch
    where a mode died, because "this configuration cannot run" is the result, and
    a series that simply stops is ambiguous between "ended" and "failed".

Palette note: the four slots are inherited unchanged from make_ff_figures.py,
where they were validated for four categorical series on this same light surface.
The skill's validator is a Node script and this cluster has no node runtime, so it
was NOT re-run here -- the values carry over rather than being re-derived.

    python make_batch_figures.py --outdir figures/batchsweep
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"

MODES = ["eager", "identity", "fused-act", "fused-shunt"]
LABEL = {"eager": "eager", "identity": "sigmoid identity",
         "fused-act": "fused activation", "fused-shunt": "fused act + shunt"}
COLOR = {"eager": "#2a78d6", "identity": "#eb6834",
         "fused-act": "#1baf7a", "fused-shunt": "#eda100"}
MARKER = {"eager": "o", "identity": "s", "fused-act": "^", "fused-shunt": "D"}


def style(ax, title, xlabel, ylabel, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5, labelpad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5, labelpad=8)
    y = 1.055 if subtitle else 1.02
    ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=16, y=y,
                 fontweight="medium")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_MUTED,
                fontsize=9, va="bottom", ha="left")


def place_endlabels(ax, items, min_gap=0.055):
    """Direct labels at the right edge, nudged apart in axes coords."""
    items = sorted(items, key=lambda t: t[1])
    lo, hi = ax.get_ylim()
    span = hi - lo or 1.0
    frac = [(y - lo) / span for _, y, _, _ in items]
    for i in range(1, len(frac)):
        if frac[i] - frac[i - 1] < min_gap:
            frac[i] = frac[i - 1] + min_gap
    for (x, _, text, color), f in zip(items, frac):
        ax.annotate(text, xy=(x, lo + f * span), xytext=(9, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", ha="left", fontweight="medium",
                    path_effects=None, clip_on=False)


def _legend_below(ax):
    """Legend under the axes. Required for >= 2 series, but the data rises left
    to right and the capacity rule spans the top, so no in-axes corner is
    reliably free -- an in-axes legend collided with the capacity annotation."""
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, handlelength=1.6,
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              columnspacing=1.6)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{name}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def cells(data):
    out = {}
    for r in data["rows"]:
        out.setdefault(r["mode"], {})[r["batch"]] = r
    return out


def _xaxis(ax, batches):
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.minorticks_off()


def fig_memory(data, tag, outdir):
    c = cells(data)
    cap = data["capacity_mb"] / 1024
    batches = sorted({r["batch"] for r in data["rows"]})
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    labels = []
    for i, m in enumerate(MODES):
        xs = [b for b in batches if c[m].get(b, {}).get("status") == "ok"]
        ys = [c[m][b]["peak_mb"] / 1024 for b in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=COLOR[m], lw=2.0, marker=MARKER[m], ms=6.5,
                mec=SURFACE, mew=1.4, label=LABEL[m], zorder=3)
        labels.append((xs[-1], ys[-1], LABEL[m], COLOR[m]))
        # Where it died: hollow marker on the capacity line. Several modes can
        # die at the SAME batch, so nudge along the log x-axis by mode index --
        # otherwise two of them occupy one point and the reader sees one.
        died = [b for b in batches
                if c[m].get(b, {}).get("status") == "oom" and b > xs[-1]]
        if died:
            # No leader line back to the last good point: it ran through the
            # series' own end label, and the hollow marker plus the subtitle
            # already say what happened. Wider stagger so co-dying modes are
            # visibly two marks rather than one.
            xo = died[0] * (1.0 + 0.075 * (i - 1.5))
            ax.plot([xo], [cap], marker=MARKER[m], ms=7.5, mfc=SURFACE,
                    mec=COLOR[m], mew=1.8, zorder=4)
    ax.axhline(cap, color=INK_MUTED, lw=1.1, ls="--", zorder=1)
    # Upper-left is the only reliably empty region once the legend moved below.
    ax.annotate(f"card capacity  {cap:,.0f} GB", xy=(batches[0], cap),
                xytext=(0, 7), textcoords="offset points", color=INK_MUTED,
                fontsize=8.5, va="bottom", ha="left")
    style(ax, "Peak memory vs batch size",
          "batch size", "peak GPU memory (GB)",
          f"{data['gpu']} · hollow marker = first batch that would not fit")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _xaxis(ax, batches)
    ax.set_ylim(0, max(cap * 1.14, 1))
    place_endlabels(ax, labels)
    _legend_below(ax)
    return save(fig, outdir, f"{tag}1_memory_vs_batch")


def fig_speed(data, tag, outdir):
    """Step time RELATIVE to eager, not absolute.

    Absolute step time spans 0.2-7 s and is dominated by batch, so all four modes
    lie on top of each other and only the last one drawn is visible -- a chart
    where three of four series are hidden communicates "identical" by accident
    rather than by design. The ratio is the quantity the panel is actually about:
    did any memory get bought with speed?

    Restricted to batches where eager itself runs, because a ratio needs a
    denominator. That eager stops at 1024 is the subject of the memory panel.
    """
    c = cells(data)
    batches = [b for b in sorted({r["batch"] for r in data["rows"]})
               if c["eager"].get(b, {}).get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    labels = []
    # Repeat-measurement spread on this harness is ~1%; drawing it stops the
    # reader over-reading sub-percent wiggles as real differences.
    ax.axhspan(0.98, 1.02, color=GRID, alpha=0.55, zorder=0)
    ax.axhline(1.0, color=INK_MUTED, lw=1.1, ls="--", zorder=1)
    ax.annotate("eager baseline  ±2% run-to-run", xy=(batches[0], 1.02),
                xytext=(0, 5), textcoords="offset points", color=INK_MUTED,
                fontsize=8.5, va="bottom", ha="left")
    for m in MODES[1:]:
        xs = [b for b in batches if c[m].get(b, {}).get("status") == "ok"]
        ys = [c[m][b]["step_ms"] / c["eager"][b]["step_ms"] for b in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=COLOR[m], lw=2.0, marker=MARKER[m], ms=6.5,
                mec=SURFACE, mew=1.4, label=LABEL[m], zorder=3)
        labels.append((xs[-1], ys[-1], LABEL[m], COLOR[m]))
    style(ax, "Step time relative to the eager baseline",
          "batch size", "step time ÷ eager step time",
          f"{data['gpu']} · below 1.0 is faster than eager · "
          f"batches where eager itself fits")
    _xaxis(ax, batches)
    ax.set_ylim(0.94, 1.06)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}×"))
    place_endlabels(ax, labels)
    _legend_below(ax)
    return save(fig, outdir, f"{tag}2_speed_vs_batch")


def fig_conserved(data, tag, outdir):
    """GB conserved against eager -- only where eager itself ran."""
    c = cells(data)
    batches = [b for b in sorted({r["batch"] for r in data["rows"]})
               if c["eager"].get(b, {}).get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    labels = []
    for m in MODES[1:]:
        xs = [b for b in batches if c[m].get(b, {}).get("status") == "ok"]
        ys = [(c["eager"][b]["peak_mb"] - c[m][b]["peak_mb"]) / 1024 for b in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=COLOR[m], lw=2.0, marker=MARKER[m], ms=6.5,
                mec=SURFACE, mew=1.4, label=LABEL[m], zorder=3)
        labels.append((xs[-1], ys[-1], LABEL[m], COLOR[m]))
    style(ax, "Memory conserved vs the eager baseline",
          "batch size", "memory saved (GB)",
          f"{data['gpu']} · only where eager itself fits, so a like-for-like "
          f"comparison exists")
    _xaxis(ax, batches)
    ax.set_ylim(bottom=0)
    place_endlabels(ax, labels)
    _legend_below(ax)
    return save(fig, outdir, f"{tag}3_memory_conserved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="profilings/dendrinet_harness/ff_stress/batchsweep")
    ap.add_argument("--outdir", default="figures/batchsweep")
    a = ap.parse_args()
    made = []
    for fname, tag in (("blackwell.json", "B"), ("a100_80gb.json", "A")):
        p = os.path.join(a.root, fname)
        if not os.path.exists(p):
            print(f"  SKIP {p} (missing)")
            continue
        data = json.load(open(p))
        print(f"{data['gpu']}:")
        made += [fig_memory(data, tag, a.outdir),
                 fig_speed(data, tag, a.outdir),
                 fig_conserved(data, tag, a.outdir)]
    print(f"\n{len(made)} figures written to {a.outdir}")
    return 0 if len(made) == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
