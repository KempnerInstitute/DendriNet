#!/usr/bin/env python
"""Profile training to identify performance bottlenecks."""

import cProfile
import io
import pstats
import sys
import time
from contextlib import contextmanager


@contextmanager
def profile_section(name):
    """Context manager to profile a section of code."""
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"Profile: {name}")
        print(f"Elapsed time: {elapsed:.3f} seconds")
        print(f"{'='*60}")

        # Print top 20 functions by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(s.getvalue())


def main():
    # Import here to measure import time
    print("Starting profiling...")

    with profile_section("Imports"):
        from dendritic_modeling.scripts.training.train_experiments import (
            main as train_main,
        )

    # Run a short training session
    print("\nRunning training with profiling...")

    # Override sys.argv for the training script
    config_path = ".vscode/config_exp.yaml"
    sys.argv = [
        "train_experiments.py",
        config_path,
        "training.epochs=2",  # Just 2 epochs for profiling
        "experiment.enable_profiling=false",  # Disable PyTorch profiling
        "experiment.enable_hooks=true",  # Keep hooks enabled as in the slow run
    ]

    with profile_section("Full Training (2 epochs)"):
        train_main(config_path)


if __name__ == "__main__":
    main()
