#!/usr/bin/env python3
"""
GPU Memory Requirements Estimator for Dendritic Models
======================================================

This script provides a command-line interface for the GPU memory estimation utilities.
It wraps the core GPUMemoryEstimator class from utils/gpu_estimation.py for easy CLI usage.

This script estimates GPU memory requirements for different dendritic model configurations
to help with resource planning and batch size optimization.

Usage:
    python estimate_gpu_requirements.py --config config.yaml [--output results.json]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Estimate GPU requirements for dendritic models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Estimate requirements for a config
    python scripts/estimate_gpu_requirements.py configs/cifar10_ei_network.yaml

    # Override batch size
    python scripts/estimate_gpu_requirements.py configs/imagenet_large.yaml --batch-size 256

    # Check distributed config
    python scripts/estimate_gpu_requirements.py configs/distributed/example_large_model_fsdp.yaml
""",
    )

    parser.add_argument("config_file", help="Path to configuration file")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size from config"
    )

    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found!")
        sys.exit(1)

    # Estimate GPU requirements
    try:
        from dendritic_modeling.utils.gpu_estimation import estimate_from_config

        estimator, model = estimate_from_config(args.config_file, args.batch_size)
    except Exception as e:
        print(f"Error analyzing configuration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
