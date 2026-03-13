#!/usr/bin/env python3
"""Generate the dendritic schematic used in the README.

Restored from the historical docs figure pipeline and simplified to emit a
text-free schematic by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

COLORS = {
    "excitatory": "#D55E00",
    "inhibitory": "#56B4E9",
    "soma": "#000000",
    "shadow": "#444444",
}


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


def save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    """Save PNG and SVG outputs with stable settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")


def create_dendritic_diagram(
    output_dir: Path,
    basename: str = "dendritic_diagram",
    include_text: bool = False,
) -> None:
    """Create the dendritic neuron schematic."""
    fig, ax = plt.subplots(figsize=(10, 10))

    soma_x, soma_y = 0.0, 0.0
    soma = Circle(
        (soma_x, soma_y),
        0.8,
        color=COLORS["soma"],
        ec="black",
        lw=2.5,
        zorder=10,
    )
    ax.add_patch(soma)
    add_shadow(soma)

    if include_text:
        ax.text(
            soma_x,
            soma_y,
            "Soma",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
            weight="bold",
        )

    # Axon with terminals
    axon_points = [(0, -0.8), (0, -2.5), (-0.5, -3.5), (-1, -4)]
    for i in range(len(axon_points) - 1):
        ax.plot(
            [axon_points[i][0], axon_points[i + 1][0]],
            [axon_points[i][1], axon_points[i + 1][1]],
            color="#2C3E50",
            lw=4,
            solid_capstyle="round",
        )

    terminal_points = [(-1.5, -4.2), (-0.8, -4.3), (-1.2, -3.8)]
    for tx, ty in terminal_points:
        ax.plot(
            [axon_points[-1][0], tx],
            [axon_points[-1][1], ty],
            color="#2C3E50",
            lw=2.5,
            solid_capstyle="round",
        )
        terminal = Circle((tx, ty), 0.15, color="#34495E", ec="black", lw=1)
        ax.add_patch(terminal)

    if include_text:
        ax.text(
            -1,
            -4.8,
            "Axon",
            ha="center",
            fontsize=12,
            weight="bold",
            color="#2C3E50",
        )

    rng = np.random.default_rng(42)
    dendrite_data = [
        {"angle": 45, "color": "#9B59B6", "label": "Apical"},
        {"angle": 135, "color": "#3498DB", "label": "Basal"},
        {"angle": 225, "color": "#3498DB", "label": "Basal"},
        {"angle": 315, "color": "#3498DB", "label": "Basal"},
    ]

    for dend in dendrite_data:
        angle_rad = np.radians(dend["angle"])

        start_x = soma_x + 0.8 * np.cos(angle_rad)
        start_y = soma_y + 0.8 * np.sin(angle_rad)
        end_x = soma_x + 2.5 * np.cos(angle_rad)
        end_y = soma_y + 2.5 * np.sin(angle_rad)

        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            color=dend["color"],
            lw=5,
            solid_capstyle="round",
            alpha=0.9,
        )

        for i in range(2):
            branch_start = 0.6 + i * 0.7
            bx1 = soma_x + (branch_start + 0.8) * np.cos(angle_rad)
            by1 = soma_y + (branch_start + 0.8) * np.sin(angle_rad)

            for offset in (-20, 20):
                branch_angle = angle_rad + np.radians(offset)
                bx2 = bx1 + 1.2 * np.cos(branch_angle)
                by2 = by1 + 1.2 * np.sin(branch_angle)

                ax.plot(
                    [bx1, bx2],
                    [by1, by2],
                    color=dend["color"],
                    lw=3,
                    solid_capstyle="round",
                    alpha=0.8,
                )

                for j in range(2):
                    tbx1 = bx1 + (j + 1) * 0.4 * np.cos(branch_angle)
                    tby1 = by1 + (j + 1) * 0.4 * np.sin(branch_angle)

                    for toffset in (-15, 15):
                        tbranch_angle = branch_angle + np.radians(toffset)
                        tbx2 = tbx1 + 0.6 * np.cos(tbranch_angle)
                        tby2 = tby1 + 0.6 * np.sin(tbranch_angle)

                        ax.plot(
                            [tbx1, tbx2],
                            [tby1, tby2],
                            color=dend["color"],
                            lw=1.5,
                            solid_capstyle="round",
                            alpha=0.7,
                        )

                        syn_color = (
                            COLORS["excitatory"]
                            if rng.random() > 0.3
                            else COLORS["inhibitory"]
                        )
                        synapse = Circle(
                            (tbx2, tby2), 0.08, color=syn_color, ec="black", lw=0.5
                        )
                        ax.add_patch(synapse)

    if include_text:
        ax.text(
            2,
            2,
            "Apical\nDendrite",
            ha="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
        ax.text(
            -2.5,
            1.5,
            "Basal\nDendrites",
            ha="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["excitatory"],
                markersize=8,
                label="Excitatory synapse",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["inhibitory"],
                markersize=8,
                label="Inhibitory synapse",
            ),
            plt.Line2D([0], [0], color="#9B59B6", lw=3, label="Apical dendrite"),
            plt.Line2D([0], [0], color="#3498DB", lw=3, label="Basal dendrite"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", frameon=True, fancybox=True
        )
        ax.set_title("Dendritic Neuron Structure", fontsize=18, weight="bold", pad=20)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    save_figure(fig, output_dir, basename)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the dendritic README schematic."
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
        "--with-text",
        action="store_true",
        help="Include labels, legend, and title from the historical figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dendritic_diagram(
        output_dir=args.output_dir,
        basename=args.basename,
        include_text=args.with_text,
    )


if __name__ == "__main__":
    main()
