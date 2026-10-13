"""Select training epochs with validation metrics and report held-out metrics.

The command consumes the long-form trajectory emitted by
``dendritic-collect-sweep-metrics``.  Epochs are selected independently within
each trained checkpoint using validation metrics only.  Metrics from the
selected epoch, including held-out test metrics when declared, are then
exported for seed-level summarization.  Test metrics are never eligible as
selection criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dendritic_modeling.config.keys import check_section_keys
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.utils.logging_config import setup_logging

SCHEMA_VERSION = "validation_epoch_selection_v1"
RELEASE_SCHEMA_VERSION = "validation_epoch_selection_release_v1"


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


_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "metadata",
        "history",
        "units",
        "join_columns",
        "metadata_columns",
        "expected_unit_count",
        "expected_epochs_per_unit",
        "epoch_column",
        "metric_column",
        "value_column",
        "selectors",
        "report_metrics",
        "output_dir",
    }
)
_TABLE_KEYS = frozenset({"path", "sha256"})
_SELECTOR_KEYS = frozenset({"name", "metric", "objective", "tie_break"})


def _load_config(path: Path) -> dict[str, Any]:
    """Load a selection config; unknown keys raise instead of being ignored."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported validation epoch selection config: {path}")
    check_section_keys(payload, _CONFIG_KEYS, section="validation_epoch_selection")
    for name in ("history", "units"):
        if isinstance(payload.get(name), Mapping):
            check_section_keys(payload[name], _TABLE_KEYS, section=name)
    for index, selector in enumerate(payload.get("selectors") or []):
        if isinstance(selector, Mapping):
            check_section_keys(selector, _SELECTOR_KEYS, section=f"selectors[{index}]")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_hashed_table(
    declaration: Mapping[str, Any], *, base: Path, name: str
) -> tuple[Path, pd.DataFrame, str]:
    if "path" not in declaration:
        raise ValueError(f"{name}.path is required")
    path = _resolve_path(str(declaration["path"]), base=base)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    declared_hash = declaration.get("sha256")
    if declared_hash is not None and str(declared_hash) != digest:
        raise ValueError(f"{name} table hash differs from selection config")
    return path, pd.read_csv(path), digest


def _normalize_selectors(declarations: Any) -> list[dict[str, str]]:
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("selectors must be a non-empty list")
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, Mapping):
            raise TypeError(f"selectors[{index}] must be a mapping")
        name = str(declaration.get("name", ""))
        metric = str(declaration.get("metric", ""))
        objective = str(declaration.get("objective", "")).lower()
        tie_break = str(declaration.get("tie_break", "earliest")).lower()
        if not name or name in names:
            raise ValueError(f"duplicate or blank selector name: {name!r}")
        if not metric.startswith("valid_"):
            raise ValueError(
                f"selector {name!r} must use a validation metric named valid_*; "
                f"received {metric!r}"
            )
        if objective not in {"min", "max"}:
            raise ValueError(f"selector {name!r} objective must be min or max")
        if tie_break not in {"earliest", "latest"}:
            raise ValueError(f"selector {name!r} tie_break must be earliest or latest")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "metric": metric,
                "objective": objective,
                "tie_break": tie_break,
            }
        )
    return normalized


def _select_epoch(
    values: pd.Series, *, objective: str, tie_break: str
) -> tuple[int, float]:
    numeric = pd.to_numeric(values, errors="raise")
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("selection metric contains a non-finite value")
    optimum = float(numeric.min() if objective == "min" else numeric.max())
    tied_epochs = numeric.index[numeric.eq(optimum)]
    epoch = int(tied_epochs.min() if tie_break == "earliest" else tied_epochs.max())
    return epoch, optimum


def select_validation_epochs(
    *, config_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Select one epoch per declared validation criterion and checkpoint."""

    config_path = config_path.resolve()
    config = _load_config(config_path)
    base = config_path.parent
    history_decl = config.get("history")
    units_decl = config.get("units")
    if not isinstance(history_decl, Mapping) or not isinstance(units_decl, Mapping):
        raise TypeError("history and units must be mappings")
    history_path, history, history_hash = _load_hashed_table(
        history_decl, base=base, name="history"
    )
    units_path, units, units_hash = _load_hashed_table(
        units_decl, base=base, name="units"
    )

    join_columns = [str(value) for value in config.get("join_columns", [])]
    metadata_columns = [str(value) for value in config.get("metadata_columns", [])]
    if not join_columns:
        raise ValueError("join_columns must explicitly identify trained checkpoints")
    declared_columns = [*join_columns, *metadata_columns]
    if len(set(declared_columns)) != len(declared_columns):
        raise ValueError("join_columns and metadata_columns must be unique")
    epoch_column = str(config.get("epoch_column", "epoch_number"))
    metric_column = str(config.get("metric_column", "metric"))
    value_column = str(config.get("value_column", "value"))
    _require_columns(
        history,
        [*join_columns, epoch_column, metric_column, value_column],
        name="history table",
    )
    _require_columns(
        units, [*join_columns, *metadata_columns], name="checkpoint-unit table"
    )

    unit_metadata = units[[*join_columns, *metadata_columns]].copy()
    duplicate_units = unit_metadata.duplicated(join_columns, keep=False)
    if duplicate_units.any():
        records = (
            unit_metadata.loc[duplicate_units, join_columns].head().to_dict("records")
        )
        raise ValueError(f"checkpoint-unit table contains duplicate units: {records}")

    selectors = _normalize_selectors(config.get("selectors"))
    report_metrics = [str(value) for value in config.get("report_metrics", [])]
    if not report_metrics or len(report_metrics) != len(set(report_metrics)):
        raise ValueError("report_metrics must be a non-empty unique list")
    required_metrics = sorted(
        {*report_metrics, *(selector["metric"] for selector in selectors)}
    )
    selected_history = history.loc[
        history[metric_column].astype(str).isin(required_metrics)
    ].copy()
    selected_history[metric_column] = selected_history[metric_column].astype(str)
    if selected_history.empty:
        raise ValueError("history table contains none of the declared metrics")
    selected_history[epoch_column] = pd.to_numeric(
        selected_history[epoch_column], errors="raise"
    ).astype(int)
    if selected_history[epoch_column].le(0).any():
        raise ValueError("epoch numbers must be positive")
    selected_history[value_column] = pd.to_numeric(
        selected_history[value_column], errors="raise"
    )

    observed_metrics = set(selected_history[metric_column].unique())
    absent_metrics = sorted(set(required_metrics) - observed_metrics)
    if absent_metrics:
        raise ValueError(f"history table lacks declared metrics: {absent_metrics}")
    duplicate_history = selected_history.duplicated(
        [*join_columns, epoch_column, metric_column], keep=False
    )
    if duplicate_history.any():
        records = (
            selected_history.loc[
                duplicate_history, [*join_columns, epoch_column, metric_column]
            ]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"history table contains duplicate metric rows: {records}")

    history_units = selected_history[join_columns].drop_duplicates()
    merged_units = unit_metadata.merge(
        history_units,
        on=join_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not merged_units["_merge"].eq("both").all():
        records = (
            merged_units.loc[
                ~merged_units["_merge"].eq("both"), [*join_columns, "_merge"]
            ]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"history and checkpoint-unit tables do not align: {records}")
    expected_units = config.get("expected_unit_count")
    if expected_units is not None and len(unit_metadata) != int(expected_units):
        raise ValueError(
            f"expected {int(expected_units)} trained checkpoints, found "
            f"{len(unit_metadata)}"
        )

    merged = selected_history.merge(
        unit_metadata,
        on=join_columns,
        how="left",
        validate="many_to_one",
    )
    expected_epochs = config.get("expected_epochs_per_unit")
    expected_epoch_count = None if expected_epochs is None else int(expected_epochs)
    if expected_epoch_count is not None and expected_epoch_count < 1:
        raise ValueError("expected_epochs_per_unit must be positive")

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    grouped = merged.groupby(join_columns, dropna=False, sort=True)
    for unit_key, group in grouped:
        key_tuple = unit_key if isinstance(unit_key, tuple) else (unit_key,)
        unit_fields = dict(zip(join_columns, key_tuple))
        metadata = {column: group[column].iloc[0] for column in metadata_columns}
        pivot = group.pivot(
            index=epoch_column, columns=metric_column, values=value_column
        ).sort_index()
        if expected_epoch_count is not None:
            expected_index = list(range(1, expected_epoch_count + 1))
            if pivot.index.tolist() != expected_index:
                raise ValueError(
                    f"epoch grid differs for checkpoint {unit_fields}: expected "
                    f"1..{expected_epoch_count}"
                )
        if pivot[required_metrics].isna().any().any():
            raise ValueError(
                f"incomplete metric-by-epoch grid for checkpoint {unit_fields}"
            )
        for selector in selectors:
            epoch, optimum = _select_epoch(
                pivot[selector["metric"]],
                objective=selector["objective"],
                tie_break=selector["tie_break"],
            )
            common = {
                **unit_fields,
                **metadata,
                "selection": selector["name"],
                "selection_metric": selector["metric"],
                "selection_objective": selector["objective"],
                "selection_tie_break": selector["tie_break"],
                "selected_epoch_number": epoch,
                "selection_value": optimum,
            }
            wide = dict(common)
            for metric in report_metrics:
                value = float(pivot.at[epoch, metric])
                if not math.isfinite(value):
                    raise ValueError(
                        f"reported metric {metric!r} is non-finite for "
                        f"checkpoint {unit_fields} at epoch {epoch}"
                    )
                wide[metric] = value
                long_rows.append({**common, "metric": metric, "value": value})
            wide_rows.append(wide)

    destination = (
        output_dir.resolve()
        if output_dir is not None
        else _resolve_path(str(config["output_dir"]), base=base)
    )
    destination.mkdir(parents=True, exist_ok=True)
    wide_path = destination / "selected_epochs.csv"
    long_path = destination / "selected_epoch_metrics.csv"
    pd.DataFrame(wide_rows).to_csv(wide_path, index=False)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)

    release = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "history_path": str(history_path),
        "history_sha256": history_hash,
        "units_path": str(units_path),
        "units_sha256": units_hash,
        "input_history_row_count": len(history),
        "trained_checkpoint_count": len(unit_metadata),
        "selected_epoch_count": len(wide_rows),
        "selected_metric_row_count": len(long_rows),
        "join_columns": join_columns,
        "metadata_columns": metadata_columns,
        "selectors": selectors,
        "report_metrics": report_metrics,
        "expected_epochs_per_unit": expected_epoch_count,
        "selection_data_role": "validation_only",
        "source_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
        "outputs": {
            wide_path.name: _sha256(wide_path),
            long_path.name: _sha256(long_path),
        },
    }
    (destination / "selection_provenance.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Selection configuration file (validation_epoch_selection_v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write into (default: the configuration's output_dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return select_validation_epochs(config_path=args.config, output_dir=args.output_dir)


def cli_main(argv: list[str] | None = None) -> int:
    """Console entry point: run :func:`main` and exit 0 on success."""
    setup_logging()
    main(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
