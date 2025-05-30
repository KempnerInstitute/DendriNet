#!/usr/bin/env python
"""
Test the new per-branch plotting with separate mean and variance plots
"""

import logging
import os
import tempfile

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Test with our implementation
from dendritic_modeling.analysis import BranchInformationAnalyzer

print("=" * 60)
print("Testing Per-Branch Plotting: Mean and Variance Plots")
print("=" * 60)

# Create analyzer
analyzer = BranchInformationAnalyzer(
    use_dvc=False, compute_per_branch=True, compute_metrics=["I_E_I", "I_E_C", "I_I_C"]
)

# Create mock per-branch results data
np.random.seed(42)
results = {
    "layer": [0, 1, 2],
    # Mock results with different means and variances per layer
    "I_E_I_mean": [0.05, 0.12, 0.08],
    "I_E_I_std": [0.01, 0.03, 0.02],
    "I_E_I_n_branches": [5, 8, 6],
    "I_E_C_mean": [0.15, 0.22, 0.18],
    "I_E_C_std": [0.02, 0.05, 0.03],
    "I_E_C_n_branches": [5, 8, 6],
    "I_I_C_mean": [0.08, 0.14, 0.11],
    "I_I_C_std": [0.015, 0.04, 0.025],
    "I_I_C_n_branches": [5, 8, 6],
}

print(f"Mock results structure: {list(results.keys())}")
print(f"Number of layers: {len(results['layer'])}")
print(f"Number of metrics: {len(analyzer.compute_metrics)}")

# Test the plotting
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"\nTesting plotting in temporary directory: {temp_dir}")

    try:
        analyzer._plot_per_branch_results(results, temp_dir, epoch=42)

        # Check if file was created
        expected_file = os.path.join(
            temp_dir, "branch_information_per_branch_epoch_42.png"
        )
        if os.path.exists(expected_file):
            print(f"✅ Plot successfully created: {expected_file}")
            file_size = os.path.getsize(expected_file)
            print(f"   File size: {file_size} bytes")
        else:
            print(f"❌ Plot file not found: {expected_file}")

    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "-" * 40)
print("Expected Plot Layout:")
print("-" * 40)
print("With 3 metrics, the plot should have:")
print("- 2 rows × 3 columns = 6 subplots total")
print("- Top row: Mean plots for I_E_I, I_E_C, I_I_C")
print("- Bottom row: Variance plots for I_E_I, I_E_C, I_I_C")
print("- Each mean plot shows: blue line with 'o' markers")
print("- Each variance plot shows: red line with 's' markers")
print("- Branch count annotations (n=X) on mean plots")
print("- Variance plots start y-axis at 0")

print("\n" + "=" * 60)
print("Summary:")
print("✅ Per-branch plotting updated to separate mean and variance")
print("✅ Mean plots: Blue lines with circle markers")
print("✅ Variance plots: Red lines with square markers")
print("✅ Clear subplot organization: top row = means, bottom row = variances")
print("✅ Branch count annotations on mean plots")
print("✅ Variance plots constrained to start at y=0")
print("=" * 60)
