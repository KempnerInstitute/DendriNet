"""Draw Supplementary Figure S29 from the two circulated models and replayed tables."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[1]
INK, BLUE, ORANGE, GRAY = "#202933", "#005c82", "#a44013", "#52606c"
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def sections(ax, letter, title):
    ax.text(
        -0.02,
        1.08,
        letter,
        weight="bold",
        size=11,
        transform=ax.transAxes,
        va="baseline",
    )
    ax.text(
        0.075,
        1.08,
        title,
        weight="bold",
        size=8.6,
        transform=ax.transAxes,
        va="baseline",
    )


def drawing(fig, bounds):
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax


def box(ax, x, y, width, height, text, color=INK, fontsize=8):
    rect = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.007,rounding_size=0.018",
        ec=color,
        fc="white",
        lw=0.85,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=color,
        size=fontsize,
    )


def connection(ax, start, end, inhibition=False):
    style = "-[,widthB=0.48,lengthB=0" if inhibition else "-|>"
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            lw=1.1,
            color=ORANGE if inhibition else BLUE,
            shrinkA=0,
            shrinkB=0,
        )
    )


def population(config, name):
    net = yaml.safe_load(config.read_text())["model"]["core"]["population_network"]
    layer = net["layers"][0]
    cell = next(p for p in layer["populations"] if p["name"] == name)
    params = {**layer["population_defaults"], **cell.get("population", {})}
    return net, layer, cell, params


def routing(ax, config, tutorial):
    name = "e" if tutorial else "e0"
    net, layer, cell, settings = population(config, name)
    origins = {
        c.get("source", "input")
        for c in layer["connections"]
        if c["target"] == name and c.get("enabled", True)
    }
    if tutorial:
        if origins != {"input", "i"} or not any(
            c["source"] == "input" and c["target"] == "i" for c in layer["connections"]
        ):
            raise ValueError("The tutorial wiring has changed")
        box(ax, 0.01, 0.59, 0.24, 0.25, "External\nfeatures")
        box(ax, 0.36, 0.28, 0.24, 0.22, "Local I cells", ORANGE)
        box(ax, 0.74, 0.59, 0.24, 0.25, "E cells")
        connection(ax, (0.26, 0.72), (0.73, 0.72))
        connection(ax, (0.22, 0.58), (0.36, 0.44))
        connection(ax, (0.61, 0.40), (0.78, 0.57), True)
    else:
        if (
            origins != {"input_e", "input_i"}
            or net["transfer_params"]["input_source_identity"] != "paired_afferents"
        ):
            raise ValueError("The study afferents have changed")
        box(ax, 0.02, 0.70, 0.41, 0.20, "External E afferents", BLUE)
        box(ax, 0.02, 0.37, 0.41, 0.20, "External I afferents", ORANGE)
        box(ax, 0.74, 0.53, 0.24, 0.25, "E cells")
        connection(ax, (0.44, 0.80), (0.73, 0.72))
        connection(ax, (0.44, 0.47), (0.713, 0.58), True)
    branch_count = sum(np.cumprod(cell["branch_factors"]))
    targets = f"{branch_count} dendritic compartments" + (
        " + soma" if settings["somatic_synapses"] else " only"
    )
    ax.text(
        0.02,
        0.085,
        f'{settings["ff_excitatory_synapses"]} E + {settings["ff_inhibitory_synapses"]} I contacts per compartment',
        size=8,
        weight="bold",
    )
    ax.text(0.02, -0.09, targets, color=GRAY, size=7.6)


def plot(args):
    fig = plt.figure(figsize=(7.05, 5.8), facecolor="white")
    ax = drawing(fig, [0.055, 0.815, 0.89, 0.108])
    sections(ax, "A", "From a configured network to a measured effect")
    for pos, label in zip(
        [0.01, 0.27, 0.53, 0.79],
        [
            "YAML\npopulations, routes",
            "DendriNet\nbuild and train",
            "Checkpoint\nbranch responses",
            "Study analysis\nseed-level effects",
        ],
    ):
        box(ax, pos, 0.08, 0.20, 0.68, label, fontsize=7.6)
    for start in [0.21, 0.47, 0.73]:
        connection(ax, (start + 0.01, 0.42), (start + 0.054, 0.42))
    left = drawing(fig, [0.055, 0.535, 0.41, 0.20])
    right = drawing(fig, [0.535, 0.535, 0.41, 0.20])
    sections(left, "B", "Tutorial · local inhibition")
    sections(right, "C", "CIFAR-10 · matched external afferents")
    routing(left, args.tutorial, True)
    routing(right, PROJECT / "configs/cifar_primary_shunting.yaml", False)
    panels = [
        fig.add_axes([0.115, 0.12, 0.33, 0.275]),
        fig.add_axes([0.60, 0.12, 0.34, 0.275]),
    ]
    tx, cx = panels
    sections(tx, "D", "Branch-to-soma transmission")
    sections(cx, "E", "Inhibitory cue transplant")
    transmission = pd.read_csv(args.summaries / "summary.csv")
    context = pd.read_csv(args.summaries / "context_summary.csv")
    for idx, (label, color, marker, line) in enumerate(
        [("Shunting", BLUE, "o", "-"), ("Additive", ORANGE, "s", "--")]
    ):
        group = transmission[
            (transmission.rule == label.lower()) & (transmission.metric == "mean")
        ].sort_values("depth")
        tx.errorbar(
            group.depth,
            group["mean"],
            yerr=[group["mean"] - group.lower, group.upper - group["mean"]],
            fmt=marker,
            ls=line,
            color=color,
            ms=4,
            lw=1.15,
            capsize=2,
            label=label,
        )
        arm = "main" if idx == 0 else "additive"
        choice = context[
            (context.arm == arm) & (context.metric == "donor_label")
        ].set_index("intervention")
        selected = choice.loc[["intact", "L2 I +1"]]
        cx.errorbar(
            np.array([0, 1]) + (idx - 0.5) * 0.055,
            selected["mean"],
            yerr=[selected["mean"] - selected.lower, selected.upper - selected["mean"]],
            fmt=marker,
            ls=line,
            color=color,
            ms=4,
            lw=1.15,
            capsize=2,
        )
    tx.set(
        yscale="log",
        xticks=[1, 2, 3, 4],
        xlabel="Dendritic depth (proximal to distal)",
        ylabel="Mean transmission",
        xlim=(0.83, 4.17),
    )
    tx.legend(fontsize=7, frameon=False, loc="lower left")
    cx.set(
        xticks=[0, 1],
        xticklabels=["Intact", "Donor cue, layer 2"],
        xlim=(-0.18, 1.18),
        ylabel="Donor-target responses (%)",
    )
    cx.set(ylim=(0, 20), yticks=[0, 5, 10, 15, 20])
    for ax in panels:
        ax.tick_params(labelsize=7, length=3)
    fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tutorial",
        type=Path,
        required=True,
        help="Exported synthetic_population YAML",
    )
    parser.add_argument(
        "--summaries",
        type=Path,
        required=True,
        help="Directory written by reproduce.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args())
