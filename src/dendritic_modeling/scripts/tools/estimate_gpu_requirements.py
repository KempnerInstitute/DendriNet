#!/usr/bin/env python3
"""
GPU Memory Requirements Estimator for Dendritic Networks
========================================================

This script analyzes a configuration file and estimates the GPU memory requirements
for training dendritic networks. It provides detailed breakdowns of memory usage
including model parameters, activations, gradients, and optimizer states.

Usage:
    python estimate_gpu_requirements.py <config_file> [--batch-size <size>]

Examples:
    # Estimate requirements for a config
    python scripts/estimate_gpu_requirements.py configs/cifar10_ei_network.yaml

    # Override batch size
    python scripts/estimate_gpu_requirements.py configs/imagenet_large.yaml --batch-size 256

    # Check distributed config
    python scripts/estimate_gpu_requirements.py configs/distributed/example_large_model_fsdp.yaml

Output:
    The script outputs a detailed breakdown including:
    - Model parameter count and memory usage
    - Activation memory requirements per layer
    - Gradient memory requirements
    - Optimizer state memory (Adam/SGD)
    - Total estimated GPU memory usage
    - Recommended GPU models based on memory requirements
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dendritic_modeling.utils.gpu_estimation import estimate_from_config
from dendritic_modeling.utils.logging_config import setup_logging

# Configure module logger
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with enhanced validation."""
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory requirements for dendritic network configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    %(prog)s configs/cifar10_ei_network.yaml

    # Override batch size
    %(prog)s configs/imagenet_large.yaml --batch-size 256

    # With verbose output
    %(prog)s configs/distributed/fsdp_config.yaml --verbose

    # Save estimation details
    %(prog)s configs/my_model.yaml --output-file gpu_estimate.txt
""",
    )

    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config (useful for testing different batch sizes)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed breakdown",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=None,
        help="Save estimation results to file",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default=None,
        help="Override precision setting (default: use config value)",
    )

    return parser.parse_args()


def validate_config_file(config_path: Path) -> None:
    """Validate that the configuration file exists and is readable."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Path is not a file: {config_path}")

    if not os.access(config_path, os.R_OK):
        raise PermissionError(f"Cannot read configuration file: {config_path}")

    if config_path.suffix not in [".yaml", ".yml"]:
        logger.warning(
            f"Config file '{config_path}' does not have a standard YAML extension. "
            "Expected .yaml or .yml"
        )


def configure_logging(verbose: bool) -> None:
    """Print the package's records, DEBUG ones too when ``verbose``."""
    setup_logging(debug=verbose)


def main():
    """Main entry point for GPU requirements estimation."""
    # Parse arguments
    args = parse_arguments()

    # Configure logging
    configure_logging(args.verbose)

    try:
        # Validate configuration file
        validate_config_file(args.config_file)

        # Log startup information
        logger.info(f"Analyzing configuration: {args.config_file}")
        if args.batch_size:
            logger.info(f"Overriding batch size to: {args.batch_size}")
        if args.precision:
            logger.info(f"Overriding precision to: {args.precision}")

        # Perform GPU memory estimation
        _estimator, _model = estimate_from_config(
            str(args.config_file),
            batch_size=args.batch_size,
        )

        # Save results if requested
        if args.output_file:
            logger.info(f"Saving detailed results to: {args.output_file}")
            with open(args.output_file, "w") as f:
                # Redirect stdout temporarily to capture estimation output
                import contextlib
                import io

                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    # Re-run estimation to capture output
                    estimate_from_config(
                        str(args.config_file),
                        batch_size=args.batch_size,
                    )

                f.write(output_buffer.getvalue())
                logger.info("Results saved successfully")

        logger.info("GPU requirements estimation completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        return 1

    except ValueError as e:
        logger.error(f"Invalid configuration or parameters: {e}")
        if args.verbose:
            logger.debug("Detailed error:", exc_info=True)
        return 1

    except ImportError as e:
        logger.error(f"Import error - required module not found: {e}")
        logger.error("Please ensure dendritic_modeling package is properly installed")
        if args.verbose:
            logger.debug("Detailed error:", exc_info=True)
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        if args.verbose:
            logger.debug("Full traceback:", exc_info=True)
        else:
            logger.info("Run with --verbose flag for detailed error information")
        return 1


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
