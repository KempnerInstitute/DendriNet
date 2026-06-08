#!/usr/bin/env python
"""Train dendritic replacements for transformer MLP/FFN modules."""

from __future__ import annotations

import argparse
import json
import logging

from dendritic_modeling.config import load_config
from dendritic_modeling.training import run_transformer_replacement_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill transformer MLP/FFN modules into dendritic replacements."
    )
    parser.add_argument("config_path", help="Path to a YAML experiment config.")
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
    result = run_transformer_replacement_training(config)
    print(json.dumps(result.asdict(), indent=2), flush=True)


if __name__ == "__main__":
    main()
