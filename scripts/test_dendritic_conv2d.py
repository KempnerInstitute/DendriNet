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
sys.path.append(str(project_root))

# Import and run tests
from tests.test_dendritic_conv2d import run_all_tests

if __name__ == "__main__":
    print("Running Dendritic Convolution Test Suite...")
    print("=" * 60)
    
    try:
        run_all_tests()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1) 