#!/usr/bin/env python
"""Check import times to identify slow imports."""

import time


def time_import(module_name, import_statement):
    """Time a single import."""
    start = time.time()
    try:
        exec(import_statement)
        elapsed = time.time() - start
        print(f"{module_name:<60} {elapsed:>8.3f}s")
        return elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"{module_name:<60} {elapsed:>8.3f}s (FAILED: {e})")
        return elapsed


print("Timing imports...")
print("=" * 70)
print(f"{'Module':<60} {'Time':>8}")
print("=" * 70)

total_time = 0

# Core imports
total_time += time_import("torch", "import torch")
total_time += time_import("omegaconf", "from omegaconf import OmegaConf")
total_time += time_import("dendritic_modeling", "import dendritic_modeling")
total_time += time_import(
    "factory", "from dendritic_modeling.networks.architectures import factory"
)
total_time += time_import(
    "config_adapter",
    "from dendritic_modeling.networks.architectures.excitation_inhibition import config_adapter",
)
total_time += time_import(
    "setup_utils", "from dendritic_modeling.scripts.script_utils import setup_utils"
)
total_time += time_import(
    "config_utils", "from dendritic_modeling.scripts.script_utils import config_utils"
)

print("=" * 70)
print(f"{'TOTAL':<60} {total_time:>8.3f}s")

# Now check if there's an issue with circular imports or module-level initialization
print("\n\nChecking for module-level code execution...")
print("=" * 70)

import ast

# Check what's in __init__ files
import os


def check_module_init(module_path):
    """Check if a module's __init__.py has substantial code."""
    init_file = os.path.join(module_path, "__init__.py")
    if os.path.exists(init_file):
        with open(init_file) as f:
            content = f.read()
            # Parse and count non-import statements
            try:
                tree = ast.parse(content)
                import_count = sum(
                    1
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                )
                total_count = sum(
                    1 for node in ast.walk(tree) if isinstance(node, ast.stmt)
                )
                non_import = total_count - import_count
                if non_import > 5:  # More than 5 non-import statements
                    print(f"{module_path}: {non_import} non-import statements")
            except:
                print(f"{module_path}: Could not parse")


# Check key modules
base_path = "src/dendritic_modeling"
for module in [
    "",
    "networks",
    "networks/architectures",
    "networks/architectures/excitation_inhibition",
    "analysis",
    "analysis/core",
    "utils",
]:
    check_module_init(os.path.join(base_path, module))
