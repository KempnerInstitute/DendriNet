#!/usr/bin/env python3
"""Generate the dendritic neuron schematic used in the README.

Synapse types differ in shape as well as colour (excitatory: triangles,
inhibitory: squares), and every structure is labelled, so the figure does not
rely on colour alone.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon

COLORS = {
    "excitatory": "#D55E00",  # vermillion
    "inhibitory": "#0072B2",  # blue
    "dendrite": "#7A7A7A",
    "axon": "#2C3E50",
    "soma": "#000000",
    "shadow": "#444444",
}
SYNAPSE_RADIUS = 0.11


def add_shadow(artist, dx: float = 2.0, dy: float = -2.0, alpha: float = 0.2) -> None:
    """Apply a light drop-shadow for visual separation."""
    artist.set_path_effects(
        [
            pe.SimplePatchShadow(
                offset=(dx, dy), alpha=alpha, shadow_rgbFace=COLORS["shadow"]
            ),
            pe.Normal(),
        ]
    )


def add_synapse(ax, x: float, y: float, excitatory: bool) -> None:
    """Draw an excitatory synapse as a triangle and an inhibitory one as a square."""
    if excitatory:
        patch = RegularPolygon(
            (x, y),
            numVertices=3,
            radius=SYNAPSE_RADIUS * 1.25,
            facecolor=COLORS["excitatory"],
            edgecolor="black",
            lw=0.6,
            zorder=5,
        )
    else:
        patch = RegularPolygon(
            (x, y),
            numVertices=4,
            radius=SYNAPSE_RADIUS * 1.1,
            orientation=np.pi / 4,
            facecolor=COLORS["inhibitory"],
            edgecolor="black",
            lw=0.6,
            zorder=5,
        )
    ax.add_patch(patch)


def draw_dendrite(ax, angle_deg: float, trunk: float, width: float, rng) -> None:
    """Draw one dendrite: a trunk, two levels of branches and distal synapses."""
    angle = np.radians(angle_deg)
    start = 0.8
    ax.plot(
        [start * np.cos(angle), (start + trunk) * np.cos(angle)],
        [start * np.sin(angle), (start + trunk) * np.sin(angle)],
        color=COLORS["dendrite"],
        lw=width,
        solid_capstyle="round",
    )
    for i in range(2):
        distance = start + trunk * (0.45 + 0.4 * i)
        bx1, by1 = distance * np.cos(angle), distance * np.sin(angle)
        for offset in (-22, 22):
            branch_angle = angle + np.radians(offset)
            bx2 = bx1 + 1.1 * np.cos(branch_angle)
            by2 = by1 + 1.1 * np.sin(branch_angle)
            ax.plot(
                [bx1, bx2],
                [by1, by2],
                color=COLORS["dendrite"],
                lw=width * 0.55,
                solid_capstyle="round",
            )
            for j in range(2):
                tbx1 = bx1 + (j + 1) * 0.42 * np.cos(branch_angle)
                tby1 = by1 + (j + 1) * 0.42 * np.sin(branch_angle)
                for toffset in (-18, 18):
                    tip_angle = branch_angle + np.radians(toffset)
                    tbx2 = tbx1 + 0.6 * np.cos(tip_angle)
                    tby2 = tby1 + 0.6 * np.sin(tip_angle)
                    ax.plot(
                        [tbx1, tbx2],
                        [tby1, tby2],
                        color=COLORS["dendrite"],
                        lw=width * 0.3,
                        solid_capstyle="round",
                    )
                    add_synapse(ax, tbx2, tby2, excitatory=rng.random() > 0.3)


def create_dendritic_diagram(
    output_dir: Path,
    basename: str = "dendritic_diagram",
    include_text: bool = True,
    write_svg: bool = False,
) -> None:
    """Create the dendritic neuron schematic."""
    fig, ax = plt.subplots(figsize=(9, 9))

    soma = Circle((0.0, 0.0), 0.8, color=COLORS["soma"], ec="black", lw=2.5, zorder=10)
    ax.add_patch(soma)
    add_shadow(soma)

    # Axon with terminals.
    axon = [(0, -0.8), (0, -2.5), (-0.5, -3.5), (-1, -4)]
    for (x0, y0), (x1, y1) in pairwise(axon):
        ax.plot([x0, x1], [y0, y1], color=COLORS["axon"], lw=4, solid_capstyle="round")
    for tx, ty in [(-1.5, -4.2), (-0.8, -4.3), (-1.2, -3.8)]:
        ax.plot(
            [axon[-1][0], tx],
            [axon[-1][1], ty],
            color=COLORS["axon"],
            lw=2.5,
            solid_capstyle="round",
        )
        ax.add_patch(Circle((tx, ty), 0.15, color="#34495E", ec="black", lw=1))

    rng = np.random.default_rng(42)
    draw_dendrite(ax, 75, trunk=2.0, width=6.0, rng=rng)  # apical
    for angle in (150, 210, 330, 15):  # basal
        draw_dendrite(ax, angle, trunk=1.2, width=4.5, rng=rng)

    if include_text:
        label = {"ha": "center", "va": "center", "fontsize": 15}
        box = {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9}
        ax.text(0, 0, "Soma", color="white", weight="bold", zorder=11, **label)
        ax.text(0.75, -3.0, "Axon", color=COLORS["axon"], weight="bold", **label)
        ax.text(2.35, 3.75, "Apical\ndendrite", bbox=box, **label)
        ax.text(-4.1, 0.0, "Basal\ndendrites", bbox=box, **label)
        legend = [
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                markerfacecolor=COLORS["excitatory"],
                markeredgecolor="black",
                markersize=13,
                label="Excitatory synapse (triangle)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor=COLORS["inhibitory"],
                markeredgecolor="black",
                markersize=11,
                label="Inhibitory synapse (square)",
            ),
        ]
        ax.legend(handles=legend, loc="upper left", frameon=True, fontsize=13)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5.2, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{basename}.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    if write_svg:
        fig.savefig(
            output_dir / f"{basename}.svg", bbox_inches="tight", facecolor="white"
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the README neuron schematic."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "_static",
        help="Directory for output assets.",
    )
    parser.add_argument(
        "--basename",
        default="dendritic_diagram",
        help="Base output filename without extension.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Omit the labels and the legend.",
    )
    parser.add_argument("--svg", action="store_true", help="Also write an SVG file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dendritic_diagram(
        output_dir=args.output_dir,
        basename=args.basename,
        include_text=not args.no_text,
        write_svg=args.svg,
    )


if __name__ == "__main__":
    main()
