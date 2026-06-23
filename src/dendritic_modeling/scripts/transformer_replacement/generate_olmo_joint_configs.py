"""Generate OLMo joint all-layer fine-tuning configs."""

from __future__ import annotations

import argparse
import json

from dendritic_modeling.scripts.transformer_replacement._olmo_generators import (
    generate_joint_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="1b")
    parser.add_argument(
        "--config-root",
        default="generated/transformer_replacement_benchmarks/configs",
    )
    parser.add_argument(
        "--results-root",
        default="results/transformer_replacement_benchmarks",
    )
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--print-configs", action="store_true")
    args = parser.parse_args()

    manifest = generate_joint_configs(
        variant=args.variant,
        config_root=args.config_root,
        results_root=args.results_root,
        max_steps=args.max_steps,
    )
    if args.print_configs:
        for item in manifest["configs"]:
            print(item["config"])
    else:
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
