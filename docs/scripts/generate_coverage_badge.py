#!/usr/bin/env python3
"""Generate a simple repo-hosted SVG coverage badge.

This avoids external badge/token issues for private repositories while still
keeping the README coverage number current.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape


def _color_for_coverage(coverage: float) -> str:
    if coverage >= 90:
        return "#2e8b57"
    if coverage >= 75:
        return "#4c9f38"
    if coverage >= 60:
        return "#a38f2d"
    if coverage >= 40:
        return "#c17d11"
    return "#c0392b"


def _coverage_from_xml(path: Path) -> float:
    root = ET.parse(path).getroot()
    rate = float(root.attrib.get("line-rate", "0"))
    return rate * 100.0


def _text_width(text: str) -> int:
    # Good enough for badge sizing without pulling in font/render deps.
    return max(28, 7 * len(text) + 10)


def render_badge(label: str, value: str, color: str) -> str:
    left = _text_width(label)
    right = _text_width(value)
    total = left + right
    right_center = left + right / 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" role="img" aria-label="{escape(label)}: {escape(value)}">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="{total}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h{left}v20H0z"/>
    <path fill="{color}" d="M{left} 0h{right}v20H{left}z"/>
    <path fill="url(#b)" d="M0 0h{total}v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{left/2:.1f}" y="15" fill="#010101" fill-opacity=".3">{escape(label)}</text>
    <text x="{left/2:.1f}" y="14">{escape(label)}</text>
    <text x="{right_center:.1f}" y="15" fill="#010101" fill-opacity=".3">{escape(value)}</text>
    <text x="{right_center:.1f}" y="14">{escape(value)}</text>
  </g>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-xml", type=Path, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output SVG badge.",
    )
    parser.add_argument("--label", default="coverage")
    args = parser.parse_args()

    if args.coverage is None and args.coverage_xml is None:
        raise SystemExit("Provide either --coverage or --coverage-xml.")

    coverage = (
        float(args.coverage)
        if args.coverage is not None
        else _coverage_from_xml(args.coverage_xml)
    )
    coverage = max(0.0, min(100.0, coverage))
    coverage = math.floor(coverage * 10 + 0.5) / 10.0
    value = f"{coverage:.1f}%"
    svg = render_badge(args.label, value, _color_for_coverage(coverage))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
