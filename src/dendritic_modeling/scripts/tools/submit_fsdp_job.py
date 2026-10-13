#!/usr/bin/env python
"""Estimate, write and submit a single-node FSDP training job.

The same tool as ``generate_slurm_job`` for configs with
``distributed.mode: fsdp``, submitting by default.

Usage:
    python -m dendritic_modeling.scripts.tools.submit_fsdp_job \\
        configs/config_exp_fsdp.yaml --account ACCOUNT --partition PARTITION
    # write the script without submitting
    python -m dendritic_modeling.scripts.tools.submit_fsdp_job CONFIG ... --generate-only
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.tools.generate_slurm_job import (
    _job_input_shape,
    build_parser,
    run,
)
from dendritic_modeling.utils.logging_config import setup_logging


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser(__doc__.split("\n\n")[0])
    parser.add_argument(
        "--generate-only", action="store_true", help="Write the script only"
    )
    args = parser.parse_args(argv)
    mode = str(load_config(args.config).distributed.mode).lower()
    if mode != "fsdp":
        parser.error(
            f"{args.config} sets distributed.mode {mode!r}; use generate_slurm_job"
        )
    args.submit = not (args.generate_only or args.dry_run)
    run(args)
    return 0


__all__ = ["_job_input_shape", "main"]


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
