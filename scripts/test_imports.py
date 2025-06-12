#!/usr/bin/env python
"""
Test script to verify that all updated imports work correctly.

This script tests the key imports used by the main scripts to ensure
they work with the new codebase structure.
"""

import sys
import traceback
from pathlib import Path

def test_import(module_name, from_statement=None):
    """Test a single import statement"""
    try:
        if from_statement:
            exec(f"from {module_name} import {from_statement}")
            print(f"✅ Successfully imported: from {module_name} import {from_statement}")
        else:
            exec(f"import {module_name}")
            print(f"✅ Successfully imported: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import: {module_name}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Unexpected error importing {module_name}: {e}")
        return False

def main():
    """Test all key imports used by the scripts"""
    
    print("🧪 Testing Updated Script Imports\n")
    print("=" * 50)
    
    # Test basic framework imports
    print("\n📦 Testing Core Framework Imports:")
    test_import("dendritic_modeling", "logger")
    test_import("dendritic_modeling.config", "load_config")
    test_import("dendritic_modeling.models", "Classifier, ProbabilisticClassifier")
    test_import("dendritic_modeling.networks", "ExcitationInhibitionNetwork, MatchedParamMLP")
    
    # Test updated imports (these were changed)
    print("\n📦 Testing Updated Imports:")
    test_import("dendritic_modeling.datasets", "get_unified_datasets")  # Was synthetic_datasets
    test_import("dendritic_modeling.training.strategies", "TrainerMLE, get_trainer")  # Was learning_strategies
    test_import("dendritic_modeling.plotting.utils", "plot_accuracy_curves")  # Was plot_utils
    
    # Test utility imports
    print("\n📦 Testing Utility Imports:")
    test_import("dendritic_modeling.utils", "evaluate_accuracy, save_dict, set_seed")
    
    # Test analysis imports
    print("\n📦 Testing Analysis Imports:")
    test_import("dendritic_modeling.analysis", "AnalysisManager")
    
    # Test standard library imports
    print("\n📦 Testing Standard Library Imports:")
    test_import("argparse")
    test_import("json")
    test_import("os")
    test_import("sys")
    test_import("pathlib", "Path")
    test_import("omegaconf", "OmegaConf")
    
    print("\n" + "=" * 50)
    print("✅ Import testing complete!")
    print("\nIf any imports failed, check:")
    print("1. You're in the correct directory")
    print("2. The package is installed: pip install -e .")
    print("3. All dependencies are installed")

if __name__ == "__main__":
    main() 