#!/usr/bin/env python
"""Benchmark a trained dendritic transformer replacement."""

from __future__ import annotations

import argparse
import json
import logging

from dendritic_modeling.config import load_config
from dendritic_modeling.training import benchmark_saved_transformer_replacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original teacher MLP output with a saved dendritic "
            "replacement on the configured validation/benchmark hidden states."
        )
    )
    parser.add_argument("config_path", help="Path to a YAML experiment config.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing layer_<idx>_replacement.pt files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    load_config.cache_clear()
    config = load_config(args.config_path)
    result = benchmark_saved_transformer_replacement(
        config,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps(result.asdict(), indent=2), flush=True)


if __name__ == "__main__":
    main()
