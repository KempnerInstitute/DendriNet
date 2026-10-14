#!/usr/bin/env python
"""Fail-closed paired comparison for replicated sweep arms.

This command uses the standard sweep :class:`DataCollector`, pairs conditions
at the trained-model seed level, and writes both the auditable paired units and
a deterministic bootstrap interval.  It is intended for confirmatory sweep
questions whose experimental unit is one trained seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval, stable_seed
from dendritic_modeling.scripts.sweeps.collectors import DataCollector
from dendritic_modeling.utils.logging_config import setup_logging


def collect_labeled_sweeps(
    sweep_roots: dict[str, Path], *, label_column: str
) -> pd.DataFrame:
    """Collect several sweep roots and attach a declared experimental arm.

    The label is supplied by the caller rather than inferred from directory
    names. Empty roots and collisions with a config-derived column fail closed.
    """
    if not sweep_roots:
        raise ValueError("At least one labeled sweep root is required")
    if not str(label_column).strip():
        raise ValueError("label_column must be non-empty")

    frames = []
    for label, root in sweep_roots.items():
        if not str(label).strip():
            raise ValueError("Sweep labels must be non-empty")
        data = DataCollector(Path(root)).collect_all()
        if data.empty:
            raise ValueError(f"No completed sweep results found under {root}")
        if label_column in data.columns:
            raise ValueError(
                f"Declared label column {label_column!r} collides with collected data"
            )
        data = data.copy()
        data[label_column] = str(label)
        data["sweep_root"] = str(Path(root).resolve())
        frames.append(data)
    return pd.concat(frames, ignore_index=True, sort=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(sweep_root: Path) -> tuple[list[dict[str, str]], str]:
    """Hash configs and completed held-out metrics used by the comparison."""
    paths = sorted((sweep_root / "configs").glob("*config_*.yaml"))
    paths.extend(
        sorted((sweep_root / "results").glob("config_*/**/performance/final.json"))
    )
    records = [
        {"path": str(path.relative_to(sweep_root)), "sha256": _sha256(path)}
        for path in paths
    ]
    payload = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    ).encode()
    return records, hashlib.sha256(payload).hexdigest()


def _analysis_identity() -> dict:
    """Hash the analyzer's modules where they were imported from.

    Paths are recorded as ``src/dendritic_modeling/...`` for a checkout and an
    installed package alike.
    """
    import dendritic_modeling
    from dendritic_modeling.analysis import statistics
    from dendritic_modeling.scripts.sweeps.collectors import (
        config_extractor,
        data_collector,
    )

    package_parent = Path(dendritic_modeling.__file__).resolve().parents[1]
    files = [
        Path(__file__).resolve(),
        *(
            Path(module.__file__).resolve()
            for module in (data_collector, config_extractor, statistics)
        ),
    ]
    return {
        "source_files": [
            {
                "path": f"src/{path.relative_to(package_parent).as_posix()}",
                "sha256": _sha256(path),
            }
            for path in files
        ]
    }


def paired_arm_comparison(
    data: pd.DataFrame,
    *,
    arm_column: str,
    reference: str,
    comparison: str,
    metric: str,
    pair_columns: list[str],
    stratum_columns: list[str] | None = None,
    require_pairs: int | None = None,
    draws: int = 10_000,
    bootstrap_seed: int = 0,
    contrast_operation: str = "difference",
    protocol_version: Literal[
        "current", "legacy_difference", "legacy_grouped"
    ] = "current",
) -> tuple[pd.DataFrame, list[dict]]:
    """Pair two sweep arms and summarize a declared within-unit contrast.

    ``difference`` computes ``comparison - reference``. The scale-free
    ``symmetric_difference`` computes ``(comparison - reference) /
    (comparison + reference)`` and fails when its denominator is zero.

    ``current`` preserves the current seed identity and output schema.
    ``legacy_difference`` reproduces ordinary contrasts before the operation
    entered the seed identity. ``legacy_grouped`` additionally supports the
    historical normalized-contrast column, retaining ordinary-contrast seeds.
    NumPy/pandas versions remain part of the reproducibility contract because
    group-key representations can change across their versions.
    """
    if protocol_version not in {"current", "legacy_difference", "legacy_grouped"}:
        raise ValueError(f"Unsupported paired protocol version: {protocol_version!r}")
    if protocol_version == "legacy_difference" and contrast_operation != "difference":
        raise ValueError("legacy_difference supports only the difference operation")
    strata = list(stratum_columns or [])
    required = [arm_column, metric, *pair_columns, *strata]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if reference == comparison:
        raise ValueError("reference and comparison arms must differ")
    if contrast_operation not in {"difference", "symmetric_difference"}:
        raise ValueError(
            f"Unsupported paired contrast operation: {contrast_operation!r}"
        )

    selected = data.loc[
        data[arm_column].astype(str).isin([str(reference), str(comparison)]),
        required,
    ].copy()
    selected[arm_column] = selected[arm_column].astype(str)
    selected[metric] = pd.to_numeric(selected[metric], errors="raise")

    identity_columns = [*strata, *pair_columns, arm_column]
    duplicate = selected.duplicated(identity_columns, keep=False)
    if duplicate.any():
        records = selected.loc[duplicate, identity_columns].to_dict("records")
        raise ValueError(f"Duplicate paired-arm observations: {records[:5]}")

    index_columns = [*strata, *pair_columns]
    wide = selected.pivot(index=index_columns, columns=arm_column, values=metric)
    missing_arms = [arm for arm in (reference, comparison) if arm not in wide.columns]
    if missing_arms:
        raise ValueError(f"Missing requested arms: {missing_arms}")
    incomplete = wide[[reference, comparison]].isna().any(axis=1)
    if incomplete.any():
        units = wide.loc[incomplete, [reference, comparison]].reset_index()
        raise ValueError(
            f"Unpaired experimental units detected: {units.head().to_dict('records')}"
        )

    paired = wide[[reference, comparison]].reset_index()
    paired = paired.rename(
        columns={
            reference: f"{metric}__{reference}",
            comparison: f"{metric}__{comparison}",
        }
    )
    prefix = (
        "symmetric_delta"
        if protocol_version == "legacy_grouped"
        and contrast_operation == "symmetric_difference"
        else "delta"
    )
    delta_column = f"{prefix}__{comparison}_minus_{reference}"
    difference = paired[f"{metric}__{comparison}"] - paired[f"{metric}__{reference}"]
    if contrast_operation == "difference":
        paired[delta_column] = difference
        delta_definition = f"{comparison} - {reference}"
    else:
        denominator = (
            paired[f"{metric}__{comparison}"] + paired[f"{metric}__{reference}"]
        )
        if denominator.eq(0).any():
            raise ValueError("symmetric_difference has a zero denominator")
        paired[delta_column] = difference / denominator
        delta_definition = (
            f"({comparison} - {reference}) / ({comparison} + {reference})"
        )

    grouped = [((), paired)]
    if strata:
        grouped = list(paired.groupby(strata, dropna=False, sort=True))
    summaries = []
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        if require_pairs is not None and len(group) != int(require_pairs):
            identity = dict(zip(strata, key_tuple))
            raise ValueError(
                f"Expected {int(require_pairs)} pairs for {identity}, observed {len(group)}"
            )
        seed_identity = [arm_column, reference, comparison, metric]
        if protocol_version == "current" or contrast_operation != "difference":
            seed_identity.append(contrast_operation)
        seed_identity.extend(key_tuple)
        interval = bootstrap_mean_interval(
            group[delta_column].to_numpy(),
            draws=draws,
            seed=stable_seed(bootstrap_seed, *seed_identity),
            require_n=require_pairs,
        )
        interval_fields = interval.to_dict()
        interval_fields["interval_method"] = interval_fields.pop("method")
        stratum_fields = dict(zip(strata, key_tuple))
        collisions = sorted(set(stratum_fields) & set(interval_fields))
        if collisions:
            raise ValueError(
                f"stratum columns collide with bootstrap summary fields: {collisions}"
            )
        record = {
            **stratum_fields,
            "arm_column": arm_column,
            "reference": reference,
            "comparison": comparison,
            "metric": metric,
            "contrast_operation": contrast_operation,
            "delta_definition": delta_definition,
            **interval_fields,
        }
        if protocol_version == "legacy_difference":
            del record["contrast_operation"]
        summaries.append(record)

    return paired.sort_values(index_columns).reset_index(drop=True), summaries


def analyze_paired_sweep(
    sweep_root: Path,
    output_dir: Path,
    **comparison_kwargs,
) -> tuple[Path, Path]:
    """Collect one sweep root and write paired units plus JSON summary."""
    data = DataCollector(sweep_root).collect_all()
    if data.empty:
        raise ValueError(f"No completed sweep results found under {sweep_root}")
    paired, summaries = paired_arm_comparison(data, **comparison_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_units.csv"
    summary_path = output_dir / "paired_summary.json"
    paired.to_csv(paired_path, index=False)
    artifact_records, artifact_set_sha256 = _artifact_inventory(sweep_root)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "paired_sweep_comparison_v1",
                "sweep_root": str(sweep_root.resolve()),
                "n_collected_configs": len(data),
                "n_paired_units": len(paired),
                "metric_units": (
                    "proportion"
                    if comparison_kwargs["metric"].endswith("accuracy")
                    else "as_recorded"
                ),
                "source_artifact_set_sha256": artifact_set_sha256,
                "source_artifacts": artifact_records,
                "analysis_identity": _analysis_identity(),
                "comparison_configuration": {
                    "protocol_version": "current",
                    **comparison_kwargs,
                },
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return paired_path, summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair two sweep arms at the trained-seed level"
    )
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-column", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--pair-column", action="append")
    parser.add_argument("--stratum-column", action="append", default=[])
    parser.add_argument("--require-pairs", type=int)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--contrast-operation",
        choices=["difference", "symmetric_difference"],
        default="difference",
    )
    parser.add_argument(
        "--protocol-version",
        choices=["current", "legacy_difference", "legacy_grouped"],
        default="current",
    )
    args = parser.parse_args(argv)

    paired_path, summary_path = analyze_paired_sweep(
        args.sweep_root,
        args.output,
        arm_column=args.arm_column,
        reference=args.reference,
        comparison=args.comparison,
        metric=args.metric,
        pair_columns=args.pair_column or ["seed"],
        stratum_columns=args.stratum_column,
        require_pairs=args.require_pairs,
        draws=args.draws,
        bootstrap_seed=args.bootstrap_seed,
        contrast_operation=args.contrast_operation,
        protocol_version=args.protocol_version,
    )
    print(f"Paired units: {paired_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
