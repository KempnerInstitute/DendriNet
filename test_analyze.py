#!/usr/bin/env python
"""
Test script to verify that analyze_ei_sweep.py works correctly.
"""

import argparse
import os
import sys


def main():
    # Determine folder path - use first command line argument or default
    folder_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/n/holylabs/LABS/kempner_dev/Users/hsafaai/results/dendritic-modeling/outputs/ei_20250412010555"
    )

    print(f"Testing analyze_ei_sweep.py on folder: {folder_path}")

    # Run the analyze_ei_sweep.py script
    cmd = f"python scripts/analyze_ei_sweep.py --folder {folder_path} --report-missing"
    print(f"Running command: {cmd}")

    # Execute the command
    exit_code = os.system(cmd)

    if exit_code == 0:
        print("Success! analyze_ei_sweep.py completed without errors.")
    else:
        print(f"Error: analyze_ei_sweep.py exited with code {exit_code}")


if __name__ == "__main__":
    main()
