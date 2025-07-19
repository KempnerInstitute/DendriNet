#!/usr/bin/env python3
"""
Estimate GPU memory requirements for dendritic models.

This script analyzes a configuration file and provides GPU recommendations
without actually starting training.

Usage:
    python scripts/estimate_gpu_requirements.py <config_file> [--batch-size N]
"""

import argparse
import sys
import os

from dendritic_modeling.utils.gpu_estimation import estimate_from_config


def main():
    parser = argparse.ArgumentParser(
        description='Estimate GPU requirements for dendritic models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Estimate requirements for a config
    python scripts/estimate_gpu_requirements.py configs/cifar10_ei_network.yaml
    
    # Override batch size
    python scripts/estimate_gpu_requirements.py configs/imagenet_large.yaml --batch-size 256
    
    # Check distributed config
    python scripts/estimate_gpu_requirements.py configs/distributed/example_large_model_fsdp.yaml
"""
    )

    parser.add_argument(
        'config_file',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )

    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found!")
        sys.exit(1)

    # Estimate GPU requirements
    try:
        estimator, model = estimate_from_config(args.config_file, args.batch_size)
    except Exception as e:
        print(f"Error analyzing configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 