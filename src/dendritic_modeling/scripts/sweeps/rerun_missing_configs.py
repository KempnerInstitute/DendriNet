#!/usr/bin/env python
"""Find the configs of a sweep that did not finish and write a job that reruns them.

A config has finished when its result directory holds ``performance/final.json``
and ``training_summary.json``. The rerun job is the sweep's own array script
restricted to the unfinished configs: the same resources, environment and
launcher, bound to the source identity frozen in ``frozen_sweep_manifest.json``.
It can be written only from that source state.

Usage:
    python -m dendritic_modeling.scripts.sweeps.rerun_missing_configs SWEEP_DIR
    python -m dendritic_modeling.scripts.sweeps.rerun_missing_configs \\
        --timestamp 20261008120000 --base-dir results --write-script
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dendritic_modeling.scripts.sweeps.control_plane import (
    discover_source_identity,
    validate_scheduler_profile,
)
from dendritic_modeling.scripts.sweeps.slurm_profile import CHECKOUT_ROOT
from dendritic_modeling.scripts.sweeps.unified.base_sweep import (
    REPO_ROOT,
    SweepJobManager,
)
from dendritic_modeling.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

MANIFEST_NAME = "frozen_sweep_manifest.json"


def resolve_sweep_dir(
    sweep_dir: str | Path | None = None,
    *,
    timestamp: str | None = None,
    base_dir: str | Path = ".",
) -> Path:
    """Return the sweep directory, given directly or as ``<base_dir>/*_<timestamp>``."""
    if sweep_dir is None:
        if not timestamp:
            raise ValueError("give the sweep directory or --timestamp")
        matches = [
            path
            for pattern in (f"*_{timestamp}", f"*_{timestamp}_[0-9]*")
            for path in Path(base_dir).glob(pattern)
            if path.is_dir()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one sweep directory *_{timestamp} under {base_dir}, "
                f"found {len(matches)}: {sorted(str(path) for path in matches)}"
            )
        sweep_dir = matches[0]
    path = Path(sweep_dir).resolve()
    if not (path / MANIFEST_NAME).is_file():
        raise FileNotFoundError(f"{path} has no {MANIFEST_NAME}; it is not a sweep")
    return path


def _manifest_file(entry: str, *, sweep_dir: Path) -> Path:
    """Resolve a manifest path (``repo:``-prefixed, sweep-relative or absolute)."""
    if entry.startswith("repo:"):
        return REPO_ROOT / entry[len("repo:") :]
    path = Path(entry)
    return path if path.is_absolute() else sweep_dir / path


def load_sweep(sweep_dir: Path) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    """Return the sweep's manifest, its configs in index order and its slurm_config."""
    manifest = json.loads((sweep_dir / MANIFEST_NAME).read_text())
    records = sorted(manifest["generated_configs"], key=lambda item: item["index"])
    if [item["index"] for item in records] != list(range(len(records))):
        raise ValueError(f"{MANIFEST_NAME} does not list configs 0..N-1")
    config_paths = [
        _manifest_file(item["path"], sweep_dir=sweep_dir) for item in records
    ]
    original = OmegaConf.to_container(
        OmegaConf.load(sweep_dir / "original_config.yaml"), resolve=True
    )
    return manifest, config_paths, dict(original.get("slurm_config") or {})


def result_dir(sweep_dir: Path, slurm_config: dict[str, Any], index: int) -> Path:
    """Return the result directory the sweep's job writes for config ``index``."""
    root = slurm_config.get("execution_results_dir")
    return (Path(root) if root else sweep_dir / "results") / f"config_{index}"


def is_finished(directory: Path) -> bool:
    """Whether a run wrote its final performance and training summary."""
    return (directory / "performance" / "final.json").is_file() and (
        directory / "training_summary.json"
    ).is_file()


def find_unfinished(
    sweep_dir: Path, n_configs: int, slurm_config: dict[str, Any]
) -> list[int]:
    """Return the indices of the configs without finished results."""
    return [
        index
        for index in range(n_configs)
        if not is_finished(result_dir(sweep_dir, slurm_config, index))
    ]


def write_rerun_script(
    sweep_dir: Path,
    manifest: dict[str, Any],
    config_paths: list[Path],
    slurm_config: dict[str, Any],
    indices: Iterable[int],
    *,
    max_concurrent: int | None = None,
) -> Path:
    """Write ``jobs/rerun_<time>.sh``, the sweep's array script for ``indices``.

    Raises when this checkout is not the source state the sweep was
    generated from, so a rerun never executes different code.
    """
    validate_scheduler_profile(
        slurm_config.get("account"), slurm_config.get("partition")
    )
    frozen = manifest.get("source_identity") or {}
    identity = discover_source_identity(
        REPO_ROOT, include_package=CHECKOUT_ROOT is None
    )
    current = identity.get("git", {})
    if CHECKOUT_ROOT is None and identity["package"]["tree_sha256"] != frozen.get(
        "package", {}
    ).get("tree_sha256"):
        raise RuntimeError(
            "The installed package differs from the one recorded by the sweep"
        )
    keys = ["commit", "tracked_diff_sha256"]
    if "untracked_src_sha256" in frozen.get("git", {}):
        keys.append("untracked_src_sha256")
    for key in keys:
        if frozen.get("git", {}).get(key) != current.get(key):
            raise RuntimeError(
                f"{REPO_ROOT} is not the source the sweep was generated from "
                f"({key} {current.get(key)} != {frozen.get('git', {}).get(key)}); "
                "rerun from that source state"
            )
    if max_concurrent is not None:
        slurm_config = {**slurm_config, "max_concurrent_jobs": int(max_concurrent)}
    script = sweep_dir / "jobs" / f"rerun_{datetime.now():%Y%m%d_%H%M%S}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    # Earlier installed sweeps ran from their output directory.
    working_dir = manifest.get("job_working_directory", sweep_dir)
    SweepJobManager(slurm_config, working_dir=working_dir)._create_array_job_script(
        str(script),
        [str(path) for path in config_paths],
        str(sweep_dir),
        indices=sorted(indices),
        source_identity=frozen,
    )
    return script


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweep_dir", nargs="?", help="Sweep directory")
    parser.add_argument("--timestamp", help="Find <base-dir>/*_<timestamp>")
    parser.add_argument("--base-dir", default=".", help="Where sweeps are written")
    parser.add_argument(
        "--config-ids",
        help="Comma-separated configs to rerun instead of every unfinished one",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --config-ids to name finished configs (their results are rewritten)",
    )
    parser.add_argument(
        "--write-script",
        action="store_true",
        help="Write jobs/rerun_<time>.sh; without it, only list the configs",
    )
    parser.add_argument("--max-concurrent", type=int, help="Array job concurrency")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sweep_dir = resolve_sweep_dir(
        args.sweep_dir, timestamp=args.timestamp, base_dir=args.base_dir
    )
    manifest, config_paths, slurm_config = load_sweep(sweep_dir)
    unfinished = find_unfinished(sweep_dir, len(config_paths), slurm_config)
    selected = unfinished
    if args.config_ids:
        selected = sorted({int(item) for item in args.config_ids.split(",")})
        outside = [index for index in selected if not 0 <= index < len(config_paths)]
        if outside:
            raise ValueError(f"config ids {outside} are not in the sweep")
        finished = sorted(set(selected) - set(unfinished))
        if finished and not args.force:
            raise ValueError(
                f"configs {finished} have finished results; pass --force to rerun them"
            )
    logger.info(
        "%s: %d configs, %d unfinished %s",
        sweep_dir,
        len(config_paths),
        len(unfinished),
        unfinished,
    )
    if not selected:
        logger.info("Nothing to rerun.")
        return 0
    if args.write_script:
        script = write_rerun_script(
            sweep_dir,
            manifest,
            config_paths,
            slurm_config,
            selected,
            max_concurrent=args.max_concurrent,
        )
        logger.info("Wrote %s; submit it with: sbatch %s", script, script)
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
