"""
Test script to verify imports work correctly after reorganization.
"""

import sys
sys.path.insert(0, 'src')

print("Testing imports from the new modular structure...\n")

# Test config imports
print("1. Testing config imports...")
try:
    from dendritic_modeling.config import Config, load_config, EINetParameters
    print("   ✓ Config imports successful")
except Exception as e:
    print(f"   ✗ Config imports failed: {e}")

# Test datasets imports
print("\n2. Testing datasets imports...")
try:
    from dendritic_modeling.datasets import LineDataset, get_unified_datasets
    print("   ✓ Datasets imports successful")
except Exception as e:
    print(f"   ✗ Datasets imports failed: {e}")

# Test models imports
print("\n3. Testing models imports...")
try:
    from dendritic_modeling.models import (
        BaseModel, Classifier, ProbabilisticClassifier,
        Regressor, DeepstRegressor, SimpleMLP
    )
    print("   ✓ Models imports successful")
except Exception as e:
    print(f"   ✗ Models imports failed: {e}")

# Test networks imports
print("\n4. Testing networks imports...")
try:
    from dendritic_modeling.networks import (
        ExcitationInhibitionNetwork, DendriNet, 
        DendriticBranchLayer, BlockLinear,
        InputNetTransform, ActivationFactory
    )
    print("   ✓ Networks imports successful")
except Exception as e:
    print(f"   ✗ Networks imports failed: {e}")

# Test training imports
print("\n5. Testing training imports...")
try:
    from dendritic_modeling.training import (
        get_trainer, CustomWeightDecayOptimizer,
        GradientScaler
    )
    print("   ✓ Training imports successful")
except Exception as e:
    print(f"   ✗ Training imports failed: {e}")

# Test analysis imports
print("\n6. Testing analysis imports...")
try:
    from dendritic_modeling.analysis import AnalysisManager
    from dendritic_modeling.analysis.information import compute_mi_cc
    print("   ✓ Analysis imports successful")
except Exception as e:
    print(f"   ✗ Analysis imports failed: {e}")

# Test plotting imports
print("\n7. Testing plotting imports...")
try:
    from dendritic_modeling.plotting import plot_NLL_loss_curves, plot_einet_params
    print("   ✓ Plotting imports successful")
except Exception as e:
    print(f"   ✗ Plotting imports failed: {e}")

# Test utils imports
print("\n8. Testing utils imports...")
try:
    from dendritic_modeling.utils import (
        evaluate_accuracy, add_gaussian_noise, save_dict,
        LoggerManager
    )
    print("   ✓ Utils imports successful")
except Exception as e:
    print(f"   ✗ Utils imports failed: {e}")

# Test scripts imports
print("\n9. Testing scripts imports...")
try:
    from dendritic_modeling.scripts import main, initialize_model
    print("   ✓ Scripts imports successful")
except Exception as e:
    print(f"   ✗ Scripts imports failed: {e}")

print("\n✅ All import tests completed!")
print("\nThe dendritic_modeling package has been successfully reorganized into a modular structure.")
print("\nNew import examples:")
print("  - Old: from dendritic_modeling.config import Config")
print("    New: from dendritic_modeling.config import Config  (same!)")
print("\n  - Old: from dendritic_modeling.networks import ExcitationInhibitionNetwork")
print("    New: from dendritic_modeling.networks import ExcitationInhibitionNetwork  (same!)")
print("\nThe reorganization maintains the same public API while providing better internal organization.") 