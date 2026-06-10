"""Generate OLMo shared EI-stack replacement configs."""

from __future__ import annotations

import argparse
import json

from dendritic_modeling.scripts.transformer_replacement._olmo_generators import (
    generate_ei_stack_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="1b")
    parser.add_argument(
        "--config-root",
        default="jobs/transformer_replacement_benchmarks/configs/generated",
    )
    parser.add_argument(
        "--results-root",
        default="results/transformer_replacement_benchmarks",
    )
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    manifest = generate_ei_stack_configs(
        variant=args.variant,
        config_root=args.config_root,
        results_root=args.results_root,
        max_steps=args.max_steps,
    )
    if args.print_config:
        print(manifest["configs"][0]["config"])
    else:
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
