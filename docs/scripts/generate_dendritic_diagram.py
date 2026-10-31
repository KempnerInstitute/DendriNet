#!/usr/bin/env python3
"""Generate the schematic at the top of the README.

Three panels without text, left to right: a pyramidal neuron, with excitatory
synapses as dots on spines and inhibitory synapses as bars across the
dendrite; the same neuron as a DendriNet compartment tree fed by sparse
excitatory and inhibitory input banks; and a small network of such neurons in
excitatory (triangular soma) and inhibitory (round soma) populations, with
excitatory projections ending on dendrites and inhibitory ones in bars. Shape,
not colour alone, separates excitation from inhibition. The morphology is
drawn procedurally from a fixed seed, so the figure is reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath

COLORS = {
    "neurite": "#26313d",
    "axon": "#7f8a95",
    "excitatory": "#0072B2",  # Okabe-Ito blue
    "inhibitory": "#D55E00",  # Okabe-Ito vermillion
    "guide": "#b9c1c9",
}
SEED = 20261030


@dataclass
class Section:
    """One unbranched stretch of neurite, drawn with a linear taper."""

    points: np.ndarray
    width_start: float
    width_end: float
    order: int
    region: str


def grow(rng, start, heading, length, step=0.045, wander=0.03, bend=0.0):
    """A gently tortuous path leaving ``start`` along ``heading`` (radians)."""
    count = max(3, round(float(length) / step))
    deviation = 0.0
    points = [np.asarray(start, dtype=float)]
    for i in range(count):
        deviation = 0.9 * deviation + rng.normal(0.0, wander)
        angle = heading + bend * (i / count) + deviation
        points.append(points[-1] + step * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(points)


def end_heading(points):
    delta = points[-1] - points[-4]
    return float(np.arctan2(delta[1], delta[0]))


def branch(
    rng,
    sections,
    start,
    heading,
    length,
    widths,
    order,
    region,
    depth,
    spread,
    shrink,
    bend=0.0,
):
    """A section that bifurcates ``depth`` more times, tapering as it goes."""
    points = grow(rng, start, heading, length, bend=bend)
    sections.append(Section(points, widths[0], widths[1], order, region))
    if depth == 0:
        return
    heading = end_heading(points)
    child_widths = (0.82 * widths[1], 0.58 * widths[1])
    for side in (-1, 1):
        branch(
            rng,
            sections,
            points[-1],
            heading + side * spread * rng.uniform(0.8, 1.2),
            length * shrink * rng.uniform(0.85, 1.15),
            child_widths,
            order + 1,
            region,
            depth - 1,
            0.9 * spread,
            shrink,
            bend,
        )


def pyramidal_neuron(rng):
    """Apical trunk with obliques and a tuft, seven basal dendrites and an axon."""
    sections = []
    trunk = grow(rng, (0.0, 0.32), np.pi / 2, 4.5, wander=0.02)
    sections.append(Section(trunk, 3.0, 1.5, 0, "apical"))
    for k, fraction in enumerate((0.1, 0.19, 0.29, 0.39, 0.5, 0.61, 0.73)):
        side = 1 if k % 2 == 0 else -1
        heading = np.pi / 2 + side * rng.uniform(0.85, 1.05)
        length = rng.uniform(0.9, 1.35) * (1.0 - 0.25 * fraction)
        branch(
            rng,
            sections,
            trunk[int(fraction * (len(trunk) - 1))],
            heading,
            length,
            (1.2, 0.8),
            1,
            "oblique",
            depth=2,
            spread=0.36,
            shrink=0.66,
            bend=-side * 0.45,
        )
    for heading in (np.pi / 2 + 0.42, np.pi / 2 - 0.4):
        branch(
            rng,
            sections,
            trunk[-1],
            heading,
            rng.uniform(0.75, 0.9),
            (1.45, 1.15),
            1,
            "tuft",
            depth=3,
            spread=0.38,
            shrink=0.8,
        )
    for heading in np.radians(
        (-170.0, -146.0, -121.0, -95.0, -66.0, -38.0, -11.0)
    ) + rng.normal(0, 0.05, 7):
        start = (0.26 * np.cos(heading), -0.18 + 0.1 * np.sin(heading))
        branch(
            rng,
            sections,
            start,
            heading,
            rng.uniform(0.65, 0.9),
            (1.9, 1.3),
            1,
            "basal",
            depth=3,
            spread=0.36,
            shrink=0.8,
        )
    axon = grow(rng, (0.0, -0.26), -np.pi / 2, 2.5, wander=0.02)
    sections.append(Section(axon, 0.9, 0.6, 0, "axon"))
    for fraction, side in ((0.5, 1), (0.78, -1)):
        start = axon[int(fraction * (len(axon) - 1))]
        sections.append(
            Section(
                grow(rng, start, -np.pi / 2 + side * 1.0, rng.uniform(0.6, 0.85)),
                0.6,
                0.45,
                1,
                "axon",
            )
        )
    return sections


def draw_tapered(ax, section, color, zorder=2):
    points = section.points
    segments = np.stack([points[:-1], points[1:]], axis=1)
    widths = np.linspace(section.width_start, section.width_end, len(segments))
    ax.add_collection(
        LineCollection(
            segments,
            linewidths=widths,
            colors=color,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )


def local_frame(points, index):
    tangent = points[min(index + 1, len(points) - 1)] - points[max(index - 1, 0)]
    tangent = tangent / np.linalg.norm(tangent)
    return tangent, np.array([-tangent[1], tangent[0]])


def spine(ax, point, normal, side, scale):
    """An excitatory synapse: a spine head just off the dendrite."""
    ax.add_patch(
        Circle(
            point + side * normal * 0.075 * scale,
            0.036 * scale,
            facecolor=COLORS["excitatory"],
            edgecolor="white",
            lw=0.3,
            zorder=4,
        )
    )


def bar(ax, point, direction, half_length, width=1.9, zorder=5):
    """An inhibitory synapse or terminal: a short bar across ``direction``'s normal."""
    ends = np.stack([point - half_length * direction, point + half_length * direction])
    ax.plot(
        *ends.T,
        color=COLORS["inhibitory"],
        lw=width,
        solid_capstyle="butt",
        zorder=zorder,
    )


def neuron_panel(ax, rng):
    sections = pyramidal_neuron(rng)
    for section in sections:
        draw_tapered(
            ax, section, COLORS["axon" if section.region == "axon" else "neurite"]
        )
    ax.add_patch(
        Polygon(
            [(0.0, 0.42), (-0.34, -0.24), (0.34, -0.24)],
            closed=True,
            facecolor=COLORS["neurite"],
            edgecolor=COLORS["neurite"],
            lw=1.2,
            joinstyle="round",
            zorder=6,
        )
    )
    for section in sections:
        if section.region == "axon" or section.order == 0:
            continue
        points = section.points
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        count = rng.poisson(length / (0.3 if section.region == "basal" else 0.22))
        for index in rng.integers(int(0.25 * len(points)), len(points) - 1, size=count):
            _, normal = local_frame(points, index)
            spine(ax, points[index], normal, rng.choice((-1, 1)), 1.0)
    # Inhibition on proximal shafts, near branch points and in the tuft.
    for section in sections:
        if section.region == "axon":
            continue
        points = section.points
        if section.region == "apical":
            fractions = (0.07, 0.17, 0.3, 0.93)
        elif section.order == 1 and section.region in ("basal", "oblique"):
            fractions = (0.6,) if rng.random() < 0.5 else ()
        elif section.region == "tuft" and section.order in (1, 2):
            fractions = (0.5,) if rng.random() < 0.5 else ()
        else:
            fractions = ()
        for fraction in fractions:
            index = int(fraction * (len(points) - 1))
            _, normal = local_frame(points, index)
            half = 0.06 + 0.025 * section.width_start
            bar(ax, points[index], normal, half, width=1.8)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-2.9, 7.7)


def curve(ax, start, end, color, lw=0.6, alpha=0.75, zorder=1):
    """A smooth cubic from ``start`` down to ``end``."""
    (x0, y0), (x1, y1) = start, end
    mid = 0.5 * (y0 + y1)
    path = MplPath(
        [(x0, y0), (x0, mid), (x1, mid), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=lw,
            alpha=alpha,
            capstyle="round",
            zorder=zorder,
        )
    )
    return np.array([x1 - x1, y1 - mid])  # final direction (vertical)


def compartment(ax, center, radius, zorder=4):
    ax.add_patch(
        Circle(
            center,
            radius,
            facecolor="white",
            edgecolor=COLORS["neurite"],
            lw=1.1,
            zorder=zorder,
        )
    )


def triangle_soma(ax, center, size, zorder=5):
    x, y = center
    ax.add_patch(
        Polygon(
            [
                (x, y + 0.62 * size),
                (x - 0.5 * size, y - 0.36 * size),
                (x + 0.5 * size, y - 0.36 * size),
            ],
            closed=True,
            facecolor=COLORS["neurite"],
            edgecolor=COLORS["neurite"],
            lw=1.0,
            joinstyle="round",
            zorder=zorder,
        )
    )


def model_panel(ax, rng):
    """A binary compartment tree (8, 4, 2 compartments) on a soma, with sparse inputs."""
    rows = {0: (8, 3.15), 1: (4, 2.05), 2: (2, 1.0)}
    centers = {
        level: [(-2.4 + (4.8 / count) * (k + 0.5), y) for k in range(count)]
        for level, (count, y) in rows.items()
    }
    soma = (0.0, 0.0)
    radius = 0.17
    links = [(centers[0][k], centers[1][k // 2]) for k in range(8)]
    links += [(centers[1][k], centers[2][k // 2]) for k in range(4)]
    links += [(centers[2][k], (soma[0], soma[1] + 0.12)) for k in range(2)]
    for (x0, y0), (x1, y1) in links:
        ax.plot(
            [x0, x1],
            [y0 - radius, y1 + radius],
            color=COLORS["neurite"],
            lw=1.2,
            solid_capstyle="round",
            zorder=2,
        )
    for level in rows:
        for center in centers[level]:
            compartment(ax, center, radius)
    triangle_soma(ax, soma, 0.46)
    ax.add_patch(
        FancyArrowPatch(
            (0.0, -0.22),
            (0.0, -0.95),
            arrowstyle="-|>",
            mutation_scale=9,
            color=COLORS["neurite"],
            lw=1.1,
            zorder=3,
        )
    )
    bank = [(-2.565 + 0.27 * k, 4.75) for k in range(20)]
    excitatory = [point for k, point in enumerate(bank) if k % 4 != 2]
    inhibitory = [point for k, point in enumerate(bank) if k % 4 == 2]
    for point in excitatory:
        ax.add_patch(
            Circle(
                point, 0.06, facecolor=COLORS["excitatory"], edgecolor="none", zorder=5
            )
        )
    for point in inhibitory:
        bar(ax, np.asarray(point), np.array([1.0, 0.0]), 0.075, width=2.2)
    for x, y in centers[0]:
        weights = np.exp(-np.abs(np.array([p[0] for p in excitatory]) - x) / 0.7)
        for index in rng.choice(
            len(excitatory), size=3, replace=False, p=weights / weights.sum()
        ):
            curve(
                ax,
                (excitatory[index][0], excitatory[index][1] - 0.07),
                (x, y + radius),
                COLORS["excitatory"],
            )
        weights = np.exp(-np.abs(np.array([p[0] for p in inhibitory]) - x) / 0.5)
        index = rng.choice(len(inhibitory), p=weights / weights.sum())
        curve(
            ax,
            (inhibitory[index][0], inhibitory[index][1] - 0.04),
            (x, y + radius + 0.07),
            COLORS["inhibitory"],
        )
        bar(ax, np.array([x, y + radius + 0.07]), np.array([1.0, 0.0]), 0.07, width=1.6)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.3, 5.3)


def glyph(ax, x, y, excitatory, scale=1.0):
    """A small model neuron; returns its axon start, distal and intermediate compartments."""
    lw = 0.9
    leaves, mids = [], []
    if excitatory:
        mids = [
            (x - 0.17 * scale, y + 0.34 * scale),
            (x + 0.17 * scale, y + 0.34 * scale),
        ]
        for mx, my in mids:
            ax.plot(
                [x, mx], [y + 0.1 * scale, my], color=COLORS["neurite"], lw=lw, zorder=2
            )
            for side in (-1, 1):
                lx, ly = mx + side * 0.09 * scale, my + 0.27 * scale
                ax.plot([mx, lx], [my, ly], color=COLORS["neurite"], lw=lw, zorder=2)
                leaves.append((lx, ly))
        for point in mids:
            compartment(ax, point, 0.045 * scale, zorder=4)
        triangle_soma(ax, (x, y), 0.24 * scale)
        out = (x, y - 0.12 * scale)
    else:
        # One dendritic compartment, as in the inhibitory populations of the examples.
        leaves = [(x, y + 0.42 * scale)]
        ax.plot([x, x], [y, leaves[0][1]], color=COLORS["neurite"], lw=lw, zorder=2)
        ax.add_patch(
            Circle(
                (x, y),
                0.075 * scale,
                facecolor=COLORS["neurite"],
                edgecolor=COLORS["neurite"],
                zorder=5,
            )
        )
        out = (x, y)
    for point in leaves:
        compartment(ax, point, 0.045 * scale, zorder=4)
    return {"out": out, "leaves": leaves, "mids": mids}


def arc(ax, start, end, color, bulge, lw=0.6, alpha=0.75, zorder=1):
    """A quadratic arc bowed sideways by ``bulge``; returns its final unit direction."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    chord = end - start
    control = 0.5 * (start + end) + bulge * np.array([-chord[1], chord[0]])
    path = MplPath(
        [start, control, end], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=lw,
            alpha=alpha,
            capstyle="round",
            zorder=zorder,
        )
    )
    direction = end - control
    return direction / np.linalg.norm(direction)


def nearby(rng, sources, x, count, length):
    """``count`` distinct sources drawn with weights that fall off with distance from ``x``."""
    weights = np.exp(-np.abs(np.asarray([s[0] for s in sources]) - x) / length)
    return rng.choice(
        len(sources), size=count, replace=False, p=weights / weights.sum()
    )


def network_panel(ax, rng):
    """Two E/I layers: the input drives both populations, inhibitory cells inhibit the
    dendrites of their excitatory neighbours, and only excitatory cells project onward.
    """
    scale = 1.45
    inputs = [(-2.45 + 0.7 * k, 4.75) for k in range(8)]
    for point in inputs:
        ax.add_patch(
            Circle(point, 0.075, facecolor=COLORS["guide"], edgecolor="none", zorder=5)
        )
    columns = (
        ("e", -2.45),
        ("i", -1.6),
        ("e", -0.75),
        ("e", 0.75),
        ("i", 1.6),
        ("e", 2.45),
    )
    layers = []
    for y in (2.95, 0.5):
        layers.append(
            [(kind, x, glyph(ax, x, y, kind == "e", scale)) for kind, x in columns]
        )
    readout = (0.0, -1.0)
    ax.add_patch(
        Circle(readout, 0.075, facecolor=COLORS["guide"], edgecolor="none", zorder=5)
    )
    radius = 0.045 * scale
    sources = [(x, y - 0.09) for x, y in inputs]
    for kind, _, cell in layers[0]:
        for leaf in cell["leaves"]:
            for index in nearby(rng, sources, leaf[0], 2 if kind == "i" else 1, 0.6):
                curve(
                    ax,
                    sources[index],
                    (leaf[0], leaf[1] + radius),
                    COLORS["excitatory"],
                )
    for previous, layer in pairwise(layers):
        # Each cell reads the excitatory cells of the previous layer within two columns,
        # each of them at least once, in left-to-right order across its leaves.
        for k, (kind, _, cell) in enumerate(layer):
            candidates = [
                j
                for j, (source_kind, _, _) in enumerate(previous)
                if source_kind == "e" and abs(j - k) <= 2
            ]
            leaves = cell["leaves"]
            if kind == "i":
                pairs = [(leaves[0], j) for j in candidates]
            else:
                chosen = candidates + list(
                    rng.choice(candidates, size=len(leaves) - len(candidates))
                )
                pairs = list(zip(leaves, sorted(chosen)))
            for leaf, j in pairs:
                curve(
                    ax,
                    previous[j][2]["out"],
                    (leaf[0], leaf[1] + radius),
                    COLORS["excitatory"],
                )
    for layer in layers:
        for k, (kind, _, cell) in enumerate(layer):
            if kind != "i":
                continue
            for neighbour, side in ((layer[k - 1], 1), (layer[k + 1], 0)):
                target = np.asarray(neighbour[2]["mids"][side])
                start = np.asarray(cell["out"]) + np.array(
                    [0.08 if side == 0 else -0.08, -0.05]
                )
                end = target + np.array([0.11 if side == 1 else -0.11, -0.03])
                bulge = -0.35 if side == 1 else 0.35
                direction = arc(
                    ax, start, end, COLORS["inhibitory"], bulge, lw=0.7, alpha=0.9
                )
                bar(ax, end, np.array([-direction[1], direction[0]]), 0.055, width=1.6)
    sources = [cell["out"] for kind, _, cell in layers[-1] if kind == "e"]
    for source in sources:
        curve(ax, source, (readout[0], readout[1] + 0.09), COLORS["excitatory"])
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.3, 5.3)


def create_dendritic_diagram(
    output_dir: Path, basename: str = "dendritic_diagram", write_svg: bool = False
) -> None:
    """Create the README schematic."""
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(15.0, 6.0))
    panels = {
        "neuron": [0.0, 0.0, 0.34, 1.0],
        "model": [0.37, 0.06, 0.29, 0.86],
        "network": [0.70, 0.06, 0.29, 0.86],
    }
    axes = {name: fig.add_axes(box) for name, box in panels.items()}
    neuron_panel(axes["neuron"], rng)
    model_panel(axes["model"], rng)
    network_panel(axes["network"], rng)
    for ax in axes.values():
        ax.set_aspect("equal")
        ax.axis("off")
    for x in (0.345, 0.675):
        fig.add_artist(
            FancyArrowPatch(
                (x, 0.5),
                (x + 0.022, 0.5),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=11,
                color=COLORS["guide"],
                lw=1.2,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{basename}.png",
        dpi=170,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    if write_svg:
        fig.savefig(
            output_dir / f"{basename}.svg",
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the README schematic.")
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
    parser.add_argument("--svg", action="store_true", help="Also write an SVG file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dendritic_diagram(
        output_dir=args.output_dir, basename=args.basename, write_svg=args.svg
    )


if __name__ == "__main__":
    main()
