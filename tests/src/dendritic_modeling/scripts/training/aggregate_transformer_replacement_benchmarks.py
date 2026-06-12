#!/usr/bin/env python
"""Aggregate transformer replacement benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _flatten_metrics(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    initial = payload.get("initial", {})
    trained = payload.get("trained", {})
    row = {
        "run_dir": str(path.parent),
        "teacher_source": payload.get("teacher_source"),
        "layers": ",".join(str(v) for v in payload.get("layer_indices", [])),
        "teacher_dense_params": payload.get("teacher_dense_params"),
        "dendritic_stored_params": payload.get("dendritic_stored_params"),
        "dendritic_active_params": payload.get("dendritic_active_params"),
        "stored_reduction_factor": payload.get("stored_reduction_factor"),
        "active_reduction_factor": payload.get("active_reduction_factor"),
        "initial_label": payload.get("initial_label"),
        "trained_label": payload.get("trained_label"),
    }
    for prefix, metrics in (("initial", initial), ("trained", trained)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    if "mse" in initial and "mse" in trained:
        row["mse_improvement_factor"] = initial["mse"] / max(trained["mse"], 1e-12)
        row["mse_reduction"] = initial["mse"] - trained["mse"]
    if "lm_loss" in initial and "lm_loss" in trained:
        row["lm_loss_delta"] = trained["lm_loss"] - initial["lm_loss"]
        row["perplexity_delta"] = trained["perplexity"] - initial["perplexity"]
    train_metrics_path = path.parent / "metrics.json"
    if train_metrics_path.exists():
        with train_metrics_path.open(encoding="utf-8") as handle:
            train_metrics = json.load(handle)
        initial_loss = train_metrics.get("initial_valid_loss")
        final_loss = train_metrics.get("final_valid_loss")
        if initial_loss is not None and final_loss is not None:
            row["distill_initial_loss"] = initial_loss
            row["distill_final_loss"] = final_loss
            row["distill_improvement_factor"] = initial_loss / max(final_loss, 1e-12)
    return row


def _write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    preferred = [
        "run_dir",
        "teacher_source",
        "trained_mse",
        "trained_relative_mse",
        "trained_cosine",
        "trained_r2",
        "mse_improvement_factor",
        "distill_initial_loss",
        "distill_final_loss",
        "distill_improvement_factor",
        "initial_lm_loss",
        "trained_lm_loss",
        "lm_loss_delta",
        "initial_perplexity",
        "trained_perplexity",
        "perplexity_delta",
        "stored_reduction_factor",
    ]
    available = {key for row in rows for key in row}
    columns = [column for column in preferred if column in available]
    lines = [
        "# Transformer Replacement Benchmarks",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        help="Directory containing benchmark_metrics.json files.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Output CSV path. Defaults to <results_root>/summary.csv.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Output Markdown path. Defaults to <results_root>/summary.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    metric_paths = sorted(root.glob("**/benchmark_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No benchmark_metrics.json files under {root}")

    rows = []
    for path in metric_paths:
        with path.open(encoding="utf-8") as handle:
            rows.append(_flatten_metrics(path, json.load(handle)))

    csv_path = Path(args.csv) if args.csv else root / "summary.csv"
    md_path = Path(args.markdown) if args.markdown else root / "summary.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
