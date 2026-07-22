"""Create seed-level summaries from collected checkpoint metric tables.

This command is the inferential boundary between package collectors and
render-only publication code. It requires explicit grouping and experimental
unit columns, rejects duplicate units, and computes deterministic bootstrap
intervals across independently trained checkpoints. Optional paired contrasts
reuse the package's standard paired-comparison implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval, stable_seed
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.scripts.sweeps.paired_comparison import paired_arm_comparison

SCHEMA_VERSION = "checkpoint_metric_summary_v1"
RELEASE_SCHEMA_VERSION = "checkpoint_metric_summary_release_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint metric summary config: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"input table is missing required columns: {missing}")


def _apply_derived_columns(
    frame: pd.DataFrame,
    declarations: Mapping[str, Any] | None,
) -> pd.DataFrame:
    selected = frame.copy()
    for name, declaration in (declarations or {}).items():
        column_name = str(name)
        if not column_name or column_name in selected.columns:
            raise ValueError(f"invalid or colliding derived column: {column_name!r}")
        if not isinstance(declaration, Mapping):
            raise TypeError(f"derived column {column_name!r} must be a mapping")
        operation = str(declaration.get("operation", ""))
        columns = [str(value) for value in declaration.get("columns", [])]
        _require_columns(selected, columns)
        if operation == "sum":
            if not columns:
                raise ValueError("sum-derived columns require at least one input")
            values = selected[columns].apply(pd.to_numeric, errors="raise")
            selected[column_name] = values.sum(axis=1)
        elif operation in {"difference", "ratio", "equal"}:
            if len(columns) != 2:
                raise ValueError(f"{operation} requires exactly two columns")
            if operation == "equal":
                selected[column_name] = selected[columns[0]].eq(selected[columns[1]])
                continue
            left = pd.to_numeric(selected[columns[0]], errors="raise")
            right = pd.to_numeric(selected[columns[1]], errors="raise")
            if operation == "difference":
                selected[column_name] = left - right
            else:
                if right.eq(0).any():
                    raise ValueError(
                        f"ratio-derived column {column_name!r} divides by zero"
                    )
                selected[column_name] = left / right
        else:
            raise ValueError(f"unsupported derived-column operation: {operation!r}")
    return selected


def _apply_filters(
    frame: pd.DataFrame,
    *,
    filters: Mapping[str, Any] | None = None,
    regex_filters: Mapping[str, str] | None = None,
    any_of_filters: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    selected = frame.copy()
    for column, expected in (filters or {}).items():
        _require_columns(selected, [str(column)])
        values = expected if isinstance(expected, list) else [expected]
        selected = selected.loc[selected[str(column)].isin(values)]
    for column, pattern in (regex_filters or {}).items():
        _require_columns(selected, [str(column)])
        selected = selected.loc[
            selected[str(column)].astype(str).str.fullmatch(str(pattern), na=False)
        ]
    if any_of_filters:
        union_mask = pd.Series(False, index=selected.index)
        for index, clause in enumerate(any_of_filters):
            if not isinstance(clause, Mapping) or not clause:
                raise ValueError(f"any_of_filters[{index}] must be a non-empty mapping")
            clause_mask = pd.Series(True, index=selected.index)
            for column, expected in clause.items():
                _require_columns(selected, [str(column)])
                values = expected if isinstance(expected, list) else [expected]
                clause_mask &= selected[str(column)].isin(values)
            union_mask |= clause_mask
        selected = selected.loc[union_mask]
    if selected.empty:
        raise ValueError("metric selection produced no rows")
    return selected.reset_index(drop=True)


def _summary_rows(
    frame: pd.DataFrame,
    *,
    group_by: list[str],
    unit_columns: list[str],
    value_column: str,
    expected_units_per_group: int | None,
    draws: int,
    bootstrap_seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    _require_columns(frame, [*group_by, *unit_columns, value_column])
    identity = [*group_by, *unit_columns]
    duplicate = frame.duplicated(identity, keep=False)
    if duplicate.any():
        records = frame.loc[duplicate, identity].head().to_dict("records")
        raise ValueError(f"duplicate experimental units within group: {records}")

    rows: list[dict[str, Any]] = []
    grouped = (
        [((), frame)]
        if not group_by
        else frame.groupby(group_by, dropna=False, sort=True)
    )
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = pd.to_numeric(group[value_column], errors="raise").to_numpy(float)
        interval = bootstrap_mean_interval(
            values,
            draws=draws,
            seed=stable_seed(bootstrap_seed, "checkpoint-metric-summary", *key_tuple),
            confidence=confidence,
            require_n=expected_units_per_group,
        )
        interval_fields = interval.to_dict()
        interval_fields["interval_method"] = interval_fields.pop("method")
        group_fields = dict(zip(group_by, key_tuple))
        collisions = sorted(set(group_fields) & set(interval_fields))
        if collisions:
            raise ValueError(
                f"group columns collide with bootstrap summary fields: {collisions}"
            )
        rows.append(
            {
                **group_fields,
                "value_column": value_column,
                "experimental_unit_columns": json.dumps(unit_columns),
                **interval_fields,
            }
        )
    return rows


def _paired_outputs(
    frame: pd.DataFrame,
    declarations: Sequence[Mapping[str, Any]],
    *,
    default_value_column: str,
    default_draws: int,
    default_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for index, declaration in enumerate(declarations):
        name = str(declaration.get("name", f"contrast_{index}"))
        selected = _apply_filters(
            frame,
            filters=declaration.get("filters"),
            regex_filters=declaration.get("regex_filters"),
            any_of_filters=declaration.get("any_of_filters"),
        )
        paired, summaries = paired_arm_comparison(
            selected,
            arm_column=str(declaration["arm_column"]),
            reference=str(declaration["reference"]),
            comparison=str(declaration["comparison"]),
            metric=str(declaration.get("value_column", default_value_column)),
            pair_columns=[str(value) for value in declaration["pair_columns"]],
            stratum_columns=[
                str(value) for value in declaration.get("stratum_columns", [])
            ],
            require_pairs=declaration.get("expected_pairs_per_stratum"),
            draws=int(declaration.get("draws", default_draws)),
            bootstrap_seed=int(declaration.get("bootstrap_seed", default_seed)),
        )
        paired.insert(0, "contrast", name)
        summary = pd.DataFrame(summaries)
        summary.insert(0, "contrast", name)
        unit_frames.append(paired)
        summary_frames.append(summary)
    return (
        (
            pd.concat(unit_frames, ignore_index=True, sort=False)
            if unit_frames
            else pd.DataFrame()
        ),
        (
            pd.concat(summary_frames, ignore_index=True, sort=False)
            if summary_frames
            else pd.DataFrame()
        ),
    )


def summarize_checkpoint_metrics(
    *, config_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Validate one declared table and write seed-level summaries."""

    config_path = config_path.resolve()
    config = _load_config(config_path)
    input_config = config.get("input")
    if not isinstance(input_config, dict) or "path" not in input_config:
        raise ValueError("input.path is required")
    input_path = _resolve_path(str(input_config["path"]), base=config_path.parent)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_hash = _sha256(input_path)
    declared_hash = input_config.get("sha256")
    if declared_hash is not None and str(declared_hash) != input_hash:
        raise ValueError("input table hash differs from summary config")

    frame = pd.read_csv(input_path)
    derived = config.get("derived_columns", {})
    if not isinstance(derived, dict):
        raise TypeError("derived_columns must be a mapping")
    with_derived = _apply_derived_columns(frame, derived)
    selected = _apply_filters(
        with_derived,
        filters=config.get("filters"),
        regex_filters=config.get("regex_filters"),
        any_of_filters=config.get("any_of_filters"),
    )
    group_by = [str(value) for value in config.get("group_by", [])]
    unit_columns = [str(value) for value in config.get("unit_columns", [])]
    if not unit_columns:
        raise ValueError("unit_columns must explicitly identify trained-model units")
    value_column = str(config.get("value_column", "value"))
    expected = config.get("expected_units_per_group")
    expected_units = None if expected is None else int(expected)
    bootstrap = config.get("bootstrap", {})
    if not isinstance(bootstrap, dict):
        raise TypeError("bootstrap must be a mapping")
    draws = int(bootstrap.get("draws", 10_000))
    seed = int(bootstrap.get("seed", 0))
    confidence = float(bootstrap.get("confidence", 0.95))

    summary_rows = _summary_rows(
        selected,
        group_by=group_by,
        unit_columns=unit_columns,
        value_column=value_column,
        expected_units_per_group=expected_units,
        draws=draws,
        bootstrap_seed=seed,
        confidence=confidence,
    )
    paired_units, paired_summary = _paired_outputs(
        selected,
        config.get("paired_comparisons", []),
        default_value_column=value_column,
        default_draws=draws,
        default_seed=seed,
    )

    destination = (
        output_dir.resolve()
        if output_dir is not None
        else _resolve_path(str(config["output_dir"]), base=config_path.parent)
    )
    destination.mkdir(parents=True, exist_ok=True)
    selected_path = destination / "checkpoint_units.csv"
    summary_path = destination / "group_summary.csv"
    paired_units_path = destination / "paired_units.csv"
    paired_summary_path = destination / "paired_summary.csv"
    selected.to_csv(selected_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    paired_units.to_csv(paired_units_path, index=False)
    paired_summary.to_csv(paired_summary_path, index=False)

    release = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "input_path": str(input_path),
        "input_sha256": input_hash,
        "input_row_count": len(frame),
        "selected_row_count": len(selected),
        "inferential_unit": "independently_trained_checkpoint",
        "unit_columns": unit_columns,
        "group_by": group_by,
        "derived_columns": derived,
        "expected_units_per_group": expected_units,
        "bootstrap": {"draws": draws, "seed": seed, "confidence": confidence},
        "source_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
        "outputs": {
            path.name: _sha256(path)
            for path in (
                selected_path,
                summary_path,
                paired_units_path,
                paired_summary_path,
            )
        },
    }
    (destination / "summary_provenance.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return summarize_checkpoint_metrics(
        config_path=args.config, output_dir=args.output_dir
    )


if __name__ == "__main__":  # pragma: no cover
    main()
