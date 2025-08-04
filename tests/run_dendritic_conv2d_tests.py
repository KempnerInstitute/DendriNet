#!/usr/bin/env python3
"""
Run the complete dendritic convolution test suite.

This script executes all test stages to validate dendritic convolution
as a drop-in replacement for standard CNN layers.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from tests.test_dendritic_conv2d import run_all_tests

    print("Running Dendritic Convolution Test Suite...")
    print("=" * 60)

    try:
        run_all_tests()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise
