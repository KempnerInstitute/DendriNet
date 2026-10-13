"""Series styles that stay distinguishable without colour.

Every series a library plot draws gets a colour from the Okabe-Ito palette,
which stays distinct under the common colour-vision deficiencies, and a marker
and line style that identify the series on their own. Bar series get a hatch
pattern instead, and series without markers (histogram outlines, dense
traces) a line style.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Okabe-Ito colours, ordered for adjacent series to contrast.
PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h")
LINESTYLES = ("-", "--", "-.", ":")
#: Hatch patterns of bar series; the first series is solid.
HATCHES = ("", "//", "..", "xx", "\\\\", "oo", "++", "**")
NEUTRAL = "#7F7F7F"


@dataclass(frozen=True)
class SeriesStyle:
    """Colour, marker and line style of one plotted series."""

    color: str
    marker: str
    linestyle: str

    def kwargs(self) -> dict[str, str]:
        """Keyword arguments for ``Axes.plot`` and ``Axes.errorbar``."""
        return {
            "color": self.color,
            "marker": self.marker,
            "linestyle": self.linestyle,
        }


def indexed_style(index: int) -> SeriesStyle:
    """Style of the ``index``-th series of a plot.

    The (marker, line style) pair is unique for the first
    ``len(MARKERS) * len(LINESTYLES)`` (48) indices.
    """
    index = int(index)
    if index < 0:
        raise ValueError("series index must be non-negative")
    return SeriesStyle(
        color=PALETTE[index % len(PALETTE)],
        marker=MARKERS[index % len(MARKERS)],
        linestyle=LINESTYLES[(index // len(MARKERS)) % len(LINESTYLES)],
    )


def depth_style(depth: int) -> SeriesStyle:
    """Style of the series of one soma-relative dendritic depth."""
    return indexed_style(depth)


def indexed_linestyle(index: int) -> str:
    """Line style of the ``index``-th series drawn without markers.

    Unique for the first ``len(LINESTYLES)`` (4) indices.
    """
    index = int(index)
    if index < 0:
        raise ValueError("series index must be non-negative")
    return LINESTYLES[index % len(LINESTYLES)]


def indexed_hatch(index: int) -> str:
    """Hatch pattern of the ``index``-th bar series of a plot.

    Unique for the first ``len(HATCHES)`` (8) indices; the first is solid.
    """
    index = int(index)
    if index < 0:
        raise ValueError("series index must be non-negative")
    return HATCHES[index % len(HATCHES)]


def hatch_color(color, alpha: float = 1.0) -> str:
    """Black or white, whichever stands out against ``color`` on white paper."""
    from matplotlib.colors import to_rgb

    alpha = float(alpha)
    red, green, blue = (alpha * c + (1.0 - alpha) for c in to_rgb(color))
    return "black" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "white"


def bar_kwargs(color, hatch: str, *, alpha: float | None = None) -> dict:
    """Keyword arguments for ``Axes.bar`` that fill with ``color`` and ``hatch``.

    The hatch is drawn in black or white, whichever stands out against the
    fill, and the bars keep no outline.
    """
    kwargs: dict = {"color": color}
    if alpha is not None:
        kwargs["alpha"] = alpha
    if hatch:
        kwargs.update(
            hatch=hatch,
            edgecolor=hatch_color(color, 1.0 if alpha is None else alpha),
            linewidth=0,
        )
    return kwargs


_SPLITS = {
    "train": SeriesStyle(PALETTE[0], "o", "-"),
    "valid": SeriesStyle(PALETTE[1], "s", "--"),
    "test": SeriesStyle(PALETTE[2], "^", ":"),
}
_SPLIT_ALIASES = {"validation": "valid", "val": "valid", "training": "train"}


def split_style(split: str) -> SeriesStyle:
    """Style of a data split's series (train, valid/validation, test)."""
    key = str(split).lower()
    key = _SPLIT_ALIASES.get(key, key)
    if key not in _SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return _SPLITS[key]


#: Network categories: colour by mechanism, solid lines for dendritic and
#: dashed for flat variants, and a marker unique to each category.
CATEGORY_STYLES: dict[str, SeriesStyle] = {
    "dendritic_shunting": SeriesStyle(PALETTE[0], "o", "-"),
    "flat_shunting": SeriesStyle(PALETTE[0], "^", "--"),
    "dendritic_additive": SeriesStyle(PALETTE[1], "s", "-"),
    "flat_additive": SeriesStyle(PALETTE[1], "v", "--"),
    "dendritic_normalized_additive": SeriesStyle(PALETTE[4], "X", "-"),
    "flat_normalized_additive": SeriesStyle(PALETTE[4], "P", "--"),
    "dendritic_signed": SeriesStyle(PALETTE[2], "D", "-"),
    "flat_signed": SeriesStyle(PALETTE[2], "d", "--"),
    "dendritic_mlp": SeriesStyle(PALETTE[6], "*", "-"),
    "flat_mlp": SeriesStyle(PALETTE[6], "p", "--"),
    "point_mlp": SeriesStyle(PALETTE[3], "h", ":"),
    "ss_mlp": SeriesStyle(PALETTE[5], "<", "-."),
    "ss_mlp_flat": SeriesStyle(PALETTE[5], ">", ":"),
    "total_param_mlp": SeriesStyle(PALETTE[3], "8", "-."),
    "active_param_mlp": SeriesStyle(PALETTE[3], "+", "--"),
    "unknown": SeriesStyle(NEUTRAL, "x", ":"),
    "default": SeriesStyle(NEUTRAL, ".", "-"),
}
CATEGORY_ALIASES = {
    "shunting": "dendritic_shunting",
    "no_shunting": "dendritic_additive",
    "mlp": "point_mlp",
}


def category_style(category: str) -> SeriesStyle:
    """Style of a network category (aliases resolve to canonical ones)."""
    key = CATEGORY_ALIASES.get(category, category)
    return CATEGORY_STYLES.get(key, CATEGORY_STYLES["unknown"])


__all__ = [
    "CATEGORY_ALIASES",
    "CATEGORY_STYLES",
    "HATCHES",
    "LINESTYLES",
    "MARKERS",
    "NEUTRAL",
    "PALETTE",
    "SeriesStyle",
    "bar_kwargs",
    "category_style",
    "depth_style",
    "hatch_color",
    "indexed_hatch",
    "indexed_linestyle",
    "indexed_style",
    "split_style",
]
