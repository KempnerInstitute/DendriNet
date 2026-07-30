#!/usr/bin/env python
"""Generate the feedforward kernel-benchmark figure set from profile_dendrinet output.

Reads every ``results_<path>.json`` under a GPU's result tree and emits the ten
figures for that GPU (S1-S5 speed, M1-M5 memory). Run it once per GPU; the two
sets are panel-for-panel comparable because the same code draws both.

Design constraints followed (dataviz skill):
  * categorical hues assigned in FIXED order (loop, fused, triton, cuda) and never
    cycled, so a path keeps its colour across every figure and across GPUs;
  * every series direct-labelled -- mandatory at 4 series, and two of the four
    slots sit below 3:1 on the light surface, which triggers the relief rule;
  * single y-axis everywhere (no dual axes);
  * recessive grid/axes, thin marks, text in ink tokens rather than series colour;
  * the speed/memory trade-off figure is drawn as connected per-path trajectories
    (an "adjacent" form) rather than a free scatter, because a 4-series all-pairs
    scatter cannot clear the colour-separation floors.

Usage:
    python make_ff_figures.py --root profilings/dendrinet_harness/ff_stress/blackwell \
                              --gpu "RTX PRO 6000 Blackwell" --outdir figures/blackwell
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- palette (light mode, validated categorical slots 1-4, fixed order) -------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"
PATHS = ["loop", "fused", "triton", "cuda"]
COLOR = {"loop": "#2a78d6", "fused": "#eb6834", "triton": "#1baf7a", "cuda": "#eda100"}
MARKER = {"loop": "o", "fused": "s", "triton": "^", "cuda": "D"}
LABEL = {
    "loop": "loop (baseline)",
    "fused": "fused autograd",
    "triton": "Triton",
    "cuda": "CUDA",
}


def style(ax, title, xlabel, ylabel, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    # Title sits above the subtitle, which sits just above the axes. Keep the pad
    # larger than the subtitle's offset or the two collide.
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="600", loc="left",
                 pad=30 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=INK_2,
                fontsize=9.5, va="bottom")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def endlabel(ax, x, y, text, color, dx=6):
    """Direct label at the end of a series (identity never by colour alone)."""
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                color=INK, fontsize=9, fontweight="600", va="center")


def place_endlabels(ax, items, dx=6, min_gap_px=13):
    """Direct-label series ends, nudging apart any that would overlap.

    Series whose final values nearly coincide (Triton and CUDA sit within a few
    MB of each other) would otherwise print on top of one another, which defeats
    the point of direct labelling.
    """
    if not items:
        return
    ax.figure.canvas.draw()
    rows = []
    for x, y, text, color in items:
        rows.append([x, y, ax.transData.transform((x, y))[1], text, color])
    rows.sort(key=lambda r: r[2])
    for i in range(1, len(rows)):
        if rows[i][2] - rows[i - 1][2] < min_gap_px:
            rows[i][2] = rows[i - 1][2] + min_gap_px
    for x, y, ypix, text, color in rows:
        dy = ypix - ax.transData.transform((x, y))[1]
        leader = (dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8,
                       shrinkA=0, shrinkB=1) if abs(dy) > 1 else None)
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    color=INK, fontsize=9, fontweight="600", va="center",
                    arrowprops=leader)


def thousands(ax, axis="y"):
    """Comma-format the VALUE axis. On horizontal bars that is x -- applying it
    to y replaces the category names with tick numbers."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {p}")


# --- data ---------------------------------------------------------------------
def load(root):
    """Return {(path, batch): record} across every results_*.json under root."""
    cells, oom = {}, {}
    for f in sorted(glob.glob(os.path.join(root, "**", "results_*.json"),
                              recursive=True)):
        for r in json.load(open(f)):
            p = r.get("path")
            b = r.get("batch")
            if p not in COLOR or b is None:
                continue
            if r.get("status") == "OK":
                cells[(p, b)] = r
            else:
                oom.setdefault((p, b), r.get("status"))
    return cells, oom


def series(cells, p, key):
    """Sorted (batches, values) for one path; key(record) -> number."""
    pts = sorted((b, key(r)) for (pp, b), r in cells.items() if pp == p)
    return [b for b, _ in pts], [v for _, v in pts]


STEP = lambda r: r["whole_step"]["step_ms_mean"]
FWD = lambda r: r["whole_step"]["fwd_ms_mean"]
BWD = lambda r: r["whole_step"]["bwd_ms_mean"]
PEAK = lambda r: r["peak_mem_mb"]


def short_kernel(name):
    """A readable stem for a mangled CUDA kernel symbol."""
    # Strip the wrappers BEFORE splitting on "(" -- "(anonymous namespace)::" has
    # its own parenthesis, so splitting first reduced the loop's dominant kernel
    # to the empty string and the bar was labelled with a bare percentage.
    s = name
    for junk in ("void ", "at::native::", "(anonymous namespace)::", "at_cuda_detail::"):
        s = s.replace(junk, "")
    s = s.split("(")[0].split("<")[0].split("::")[-1]
    return (s[:24] + "…") if len(s) > 25 else s


def kernel_ops(rec, phase):
    """Deduplicated device-kernel self-CUDA ms for one phase.

    The profiler lists an aten:: dispatch row AND its kernel row, and the custom
    paths additionally list an autograd.Function wrapper whose time is the sum of
    its own kernels -- summing all three would triple-count. Keep kernels only.
    """
    sf = rec.get("split_fb") or {}
    d = sf.get(phase) or {}
    out = {}
    for o in d.get("ops", []):
        op = o["op"]
        if op.startswith("aten::") or "Command Buffer" in op:
            continue
        if op.startswith(("_TritonSparseGather", "_CudaSparseGather")):
            continue
        out[op] = out.get(op, 0.0) + o["self_cuda_us"] / 1000.0
    return out


# --- figures ------------------------------------------------------------------
def fig_speedup_bars(cells, gpu, outdir):
    """S1: speedup vs loop, grouped by batch."""
    batches = sorted({b for (p, b) in cells if p == "loop"})
    kernels = [p for p in PATHS if p != "loop"]
    if not batches:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    w = 0.26
    for i, p in enumerate(kernels):
        xs, ys = [], []
        for j, b in enumerate(batches):
            if (p, b) in cells and ("loop", b) in cells:
                xs.append(j + (i - 1) * w)
                ys.append(STEP(cells[("loop", b)]) / STEP(cells[(p, b)]))
        bars = ax.bar(xs, ys, w * 0.92, color=COLOR[p], label=LABEL[p],
                      zorder=3, edgecolor=SURFACE, linewidth=2)
        for rect, v in zip(bars, ys):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}×",
                    ha="center", va="bottom", fontsize=8.5, color=INK,
                    fontweight="600")
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.4, zorder=2)
    ax.text(len(batches) - 0.5, 1.0, " loop = 1.0×", va="center",
            fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches)
    style(ax, "Speedup over the eager baseline", "batch size", "× faster than loop",
          subtitle=f"{gpu} · whole training step (fwd+bwd) · higher is better")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=3, loc="upper left")
    save(fig, outdir, "S1_speedup_by_batch.png")


def fig_step_time(cells, gpu, outdir, oom):
    """S2: absolute step time vs batch."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = []
    for p in PATHS:
        xs, ys = series(cells, p, STEP)
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKER[p], color=COLOR[p], linewidth=2,
                markersize=8, label=LABEL[p], zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6)
        labels.append((xs[-1], ys[-1], LABEL[p], COLOR[p]))
    for (p, b) in sorted(oom):
        if p == "loop":
            ax.annotate("OOM", xy=(b, ax.get_ylim()[1] * 0.92), color=COLOR[p],
                        fontsize=9, fontweight="600", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted({b for _, b in cells}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "Step time scales with batch", "batch size", "ms per step (log)",
          subtitle=f"{gpu} · lower is better · log-log")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.55)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    place_endlabels(ax, labels)
    save(fig, outdir, "S2_step_time_vs_batch.png")


def fig_fwd_bwd(cells, gpu, outdir, batch):
    """S3: forward vs backward split at one batch."""
    have = [p for p in PATHS if (p, batch) in cells]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    widest = max(FWD(cells[(p, batch)]) + BWD(cells[(p, batch)]) for p in have)
    for i, p in enumerate(have):
        r = cells[(p, batch)]
        f, b = FWD(r), BWD(r)
        ax.barh(i, f, color=COLOR[p], zorder=3, height=0.58,
                edgecolor=SURFACE, linewidth=2)
        ax.barh(i, b, left=f, color=COLOR[p], alpha=0.42, zorder=3, height=0.58,
                edgecolor=SURFACE, linewidth=2)
        ax.text(f + b, i, f"  {f + b:,.0f} ms", va="center", fontsize=9,
                color=INK, fontweight="600")
        if f > widest * 0.10:
            ax.text(f / 2, i, f"fwd {f:,.0f}", va="center", ha="center",
                    fontsize=8, color=SURFACE, fontweight="600")
        if b > widest * 0.10:
            ax.text(f + b / 2, i, f"bwd {b:,.0f}", va="center", ha="center",
                    fontsize=8, color=INK, fontweight="600")
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([LABEL[p] for p in have])
    ax.invert_yaxis()
    style(ax, "Backward dominates, and it is where the kernels win",
          "ms per step", "",
          subtitle=f"{gpu} · batch {batch} · solid = forward, translucent = backward")
    save(fig, outdir, "S3_fwd_bwd_split.png")


def fig_per_op(cells, gpu, outdir, batch, topn=6):
    """S4: per-op device-kernel breakdown at one batch."""
    have = [p for p in PATHS if (p, batch) in cells and cells[(p, batch)].get("split_fb")]
    if not have:
        print(f"  SKIP S4: no per-op data at batch {batch}")
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, p in enumerate(have):
        ops = kernel_ops(cells[(p, batch)], "forward")
        for k, v in kernel_ops(cells[(p, batch)], "backward").items():
            ops[k] = ops.get(k, 0) + v
        top = sorted(ops.items(), key=lambda kv: -kv[1])[:topn]
        left = 0.0
        # Label ONLY the dominant kernel: the tail segments are far narrower than
        # their mangled symbol names, so labelling each one produced a pile-up.
        # The rest is legible in the per-op JSON / ops_*.txt tables.
        for j, (name, ms) in enumerate(top):
            ax.barh(i, ms, left=left, height=0.58, zorder=3,
                    color=COLOR[p], alpha=1.0 - j * 0.13,
                    edgecolor=SURFACE, linewidth=2)
            left += ms
        big_name, big_ms = top[0]
        ax.annotate(f"{short_kernel(big_name)}  {100 * big_ms / left:.0f}%",
                    xy=(big_ms / 2, i), xytext=(0, 15), textcoords="offset points",
                    ha="center", fontsize=8, color=INK, fontweight="600")
        ax.text(left, i, f"  {left:,.0f} ms", va="center", fontsize=9,
                color=INK, fontweight="600")
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([LABEL[p] for p in have])
    ax.invert_yaxis()
    style(ax, "Where the GPU time actually goes", "device-kernel self-CUDA ms", "",
          subtitle=f"{gpu} · batch {batch} · top {topn} kernels, aten/wrapper rows "
                   f"deduplicated")
    save(fig, outdir, "S4_per_op_breakdown.png")


def fig_efficiency(cells, gpu, outdir, batch):
    """S5: kernel launches per step (dispatch pressure)."""
    have = [p for p in PATHS if (p, batch) in cells and cells[(p, batch)].get("split_fb")]
    if not have:
        print(f"  SKIP S5: no per-op data at batch {batch}")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    counts = []
    for p in have:
        sf = cells[(p, batch)]["split_fb"]
        n = sum(o["count"] for ph in ("forward", "backward")
                for o in sf[ph]["ops"]
                if not o["op"].startswith("aten::")
                and "Command Buffer" not in o["op"]
                and not o["op"].startswith(("_TritonSparseGather", "_CudaSparseGather")))
        counts.append(n)
    bars = ax.bar(range(len(have)), counts, 0.55,
                  color=[COLOR[p] for p in have], zorder=3,
                  edgecolor=SURFACE, linewidth=2)
    for rect, v in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:,}", ha="center",
                va="bottom", fontsize=9, color=INK, fontweight="600")
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([LABEL[p] for p in have])
    thousands(ax)
    style(ax, "Fewer, larger kernel launches", "", "device kernels launched",
          subtitle=f"{gpu} · batch {batch} · fused kernels replace many small ops")
    save(fig, outdir, "S5_launch_count.png")


def fig_mem_bars(cells, gpu, outdir):
    """M1: peak memory by path, grouped by batch."""
    batches = sorted({b for _, b in cells})
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    w = 0.2
    for i, p in enumerate(PATHS):
        xs, ys = [], []
        for j, b in enumerate(batches):
            if (p, b) in cells:
                xs.append(j + (i - 1.5) * w)
                ys.append(PEAK(cells[(p, b)]))
        ax.bar(xs, ys, w * 0.9, color=COLOR[p], label=LABEL[p], zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches)
    thousands(ax)
    style(ax, "Peak GPU memory", "batch size", "peak MB",
          subtitle=f"{gpu} · lower is better")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=4, loc="upper left")
    save(fig, outdir, "M1_peak_memory_by_batch.png")


def fig_mem_scaling(cells, gpu, outdir, oom, capacity_mb=None):
    """M2: peak memory vs batch, with the per-sample slope."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = []
    for p in PATHS:
        xs, ys = series(cells, p, PEAK)
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKER[p], color=COLOR[p], linewidth=2,
                markersize=8, label=LABEL[p], zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6)
        slope = ((ys[-1] - ys[0]) / (xs[-1] - xs[0])) if len(xs) > 1 else 0
        labels.append((xs[-1], ys[-1], f"{LABEL[p]}  ({slope:,.0f} MB/sample)",
                       COLOR[p]))
    ax.set_xscale("log", base=2)
    if capacity_mb:
        ax.axhline(capacity_mb, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 4)),
                   zorder=2)
        # Blended transform: x in axes fraction, y in data. Anchoring this label
        # in data-x captured the pre-log-scale xlim and threw the tight bbox out
        # to -881 inches, producing a 160-megapixel sliver.
        ax.text(0.99, capacity_mb, "GPU capacity ", va="bottom", ha="right",
                fontsize=8.5, color=INK_2, transform=ax.get_yaxis_transform())
    ceiling = capacity_mb or max(PEAK(r) for r in cells.values())
    for (p, b) in sorted(oom):
        ax.plot([b], [ceiling], marker="x", color=COLOR.get(p, INK), markersize=11,
                markeredgewidth=2.4, zorder=4)
        ax.annotate("OOM", xy=(b, ceiling), xytext=(0, 11),
                    textcoords="offset points", color=COLOR.get(p, INK),
                    fontsize=9, fontweight="600", ha="center")
    ax.set_xticks(sorted({b for _, b in cells} | {b for _, b in oom}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    thousands(ax)
    style(ax, "Memory growth per sample is what caps the baseline",
          "batch size", "peak MB",
          subtitle=f"{gpu} · slope = activation cost per sample")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.6)
    place_endlabels(ax, labels)
    save(fig, outdir, "M2_memory_vs_batch.png")


def fig_mem_reduction(cells, gpu, outdir):
    """M3: memory saved vs loop."""
    batches = sorted({b for (p, b) in cells if p == "loop"})
    kernels = [p for p in PATHS if p != "loop"]
    if not batches:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    w = 0.26
    for i, p in enumerate(kernels):
        xs, ys = [], []
        for j, b in enumerate(batches):
            if (p, b) in cells and ("loop", b) in cells:
                base = PEAK(cells[("loop", b)])
                xs.append(j + (i - 1) * w)
                ys.append(100 * (base - PEAK(cells[(p, b)])) / base)
        bars = ax.bar(xs, ys, w * 0.92, color=COLOR[p], label=LABEL[p], zorder=3,
                      edgecolor=SURFACE, linewidth=2)
        for rect, v in zip(bars, ys):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"−{v:.0f}%",
                    ha="center", va="bottom", fontsize=8.5, color=INK,
                    fontweight="600")
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches)
    style(ax, "Memory saved versus the eager baseline", "batch size",
          "% less peak memory", subtitle=f"{gpu} · higher is better")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.30))
    save(fig, outdir, "M3_memory_reduction.png")


def fig_mem_split(cells, gpu, outdir, batch):
    """M4: activation vs resident (parameter/state) memory."""
    have = [p for p in PATHS if (p, batch) in cells]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    widest = max(PEAK(cells[(p, batch)]) for p in have)
    for i, p in enumerate(have):
        r = cells[(p, batch)]
        base = r.get("param_mem_mb", 0.0)
        act = r.get("activation_mem_mb", PEAK(r) - base)
        ax.barh(i, base, color=COLOR[p], zorder=3, height=0.58,
                edgecolor=SURFACE, linewidth=2)
        ax.barh(i, act, left=base, color=COLOR[p], alpha=0.42, zorder=3,
                height=0.58, edgecolor=SURFACE, linewidth=2)
        ax.text(base + act, i, f"  {base + act:,.0f} MB", va="center", fontsize=9,
                color=INK, fontweight="600")
        if act > widest * 0.22:
            ax.text(base + act / 2, i, f"activations {act:,.0f}", va="center",
                    ha="center", fontsize=8, color=INK, fontweight="600")
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([LABEL[p] for p in have])
    ax.invert_yaxis()
    thousands(ax, axis="x")
    style(ax, "Activations, not parameters, are the memory", "MB", "",
          subtitle=f"{gpu} · batch {batch} · solid = resident, translucent = activations")
    save(fig, outdir, "M4_activation_vs_resident.png")


def fig_pareto(cells, gpu, outdir):
    """M5: speed/memory trade-off as per-path trajectories over batch."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    labels = []
    for p in PATHS:
        pts = sorted((b, PEAK(cells[(p, b)]), STEP(cells[(p, b)]))
                     for (pp, b) in cells if pp == p)
        if not pts:
            continue
        ax.plot([m for _, m, _ in pts], [t for _, _, t in pts], marker=MARKER[p],
                color=COLOR[p], linewidth=2, markersize=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=LABEL[p])
        labels.append((pts[-1][1], pts[-1][2], LABEL[p], COLOR[p]))
        for b, m, t in pts:
            ax.annotate(f"{b}", xy=(m, t), xytext=(0, -12),
                        textcoords="offset points", fontsize=7.5,
                        color=INK_MUTED, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    thousands(ax)
    style(ax, "Speed and memory move together, not against each other",
          "peak MB (log)", "ms per step (log)",
          subtitle=f"{gpu} · each line is one path swept over batch (labelled) · "
                   f"toward lower-left is better")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    place_endlabels(ax, labels)
    save(fig, outdir, "M5_speed_memory_tradeoff.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing results_*.json trees")
    ap.add_argument("--gpu", required=True, help="GPU name for subtitles")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--capacity-mb", type=float, default=None,
                    help="GPU memory capacity, drawn as the OOM ceiling in M2")
    ap.add_argument("--headline-batch", type=int, default=None,
                    help="batch for the single-batch figures (default: largest "
                         "batch where every path has data)")
    args = ap.parse_args()

    cells, oom = load(args.root)
    if not cells:
        raise SystemExit(f"no OK results under {args.root}")
    batches = sorted({b for _, b in cells})
    complete = [b for b in batches if all((p, b) in cells for p in PATHS)]
    hb = args.headline_batch or (complete[-1] if complete else batches[-1])
    print(f"loaded {len(cells)} cells | batches {batches} | "
          f"all-path batches {complete} | headline batch {hb}")
    if oom:
        print(f"OOM/failed cells: {sorted(oom)}")

    fig_speedup_bars(cells, args.gpu, args.outdir)
    fig_step_time(cells, args.gpu, args.outdir, oom)
    fig_fwd_bwd(cells, args.gpu, args.outdir, hb)
    prof_ok = [b for b in batches
               if all((p, b) in cells and cells[(p, b)].get("split_fb") for p in PATHS)]
    pb = prof_ok[-1] if prof_ok else hb
    if pb != hb:
        print(f"  per-op figures use batch {pb} (only rung with split_fb for all paths)")
    fig_per_op(cells, args.gpu, args.outdir, pb)
    fig_efficiency(cells, args.gpu, args.outdir, pb)
    fig_mem_bars(cells, args.gpu, args.outdir)
    fig_mem_scaling(cells, args.gpu, args.outdir, oom, args.capacity_mb)
    fig_mem_reduction(cells, args.gpu, args.outdir)
    fig_mem_split(cells, args.gpu, args.outdir, hb)
    fig_pareto(cells, args.gpu, args.outdir)


if __name__ == "__main__":
    main()
