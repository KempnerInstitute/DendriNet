# Visualization Manager Refactoring Specification

**Status**: ⏳ PENDING REFACTORING  
**Target File**: `src/dendritic_modeling/visualization_manager.py` (1781 lines)  
**Date Created**: June 2024  
**Priority**: HIGH (Large monolithic file)  

## Overview

The `visualization_manager.py` contains a large `PlotManager` class that needs to be refactored into the existing organized structure. This specification documents exactly what functionality needs to be extracted and where it should be moved.

## Current File Structure

```python
# visualization_manager.py (1781 lines)
class PlotManager:
    def __init__(self, model, data, args, device="cpu")                    # Line 42
    def run_plot(self, plot_name, epoch=None)                             # Line 66
    
    # PLOTTING METHODS (→ Already handled by plotting/ package)
    def _plot_weights(self)                                               # Line 103
    def _plot_activations(self)                                           # Line 121  
    def _plot_gradients(self)                                             # Line 137
    def _plot_profiles(self)                                              # Line 158
    
    # BRANCH ANALYSIS METHODS (→ Move to analysis/)
    def _branch_info(self)                                                # Line 179-445
    def _branch_ablation(self)                                            # Line 487-993
    
    # DENDRITIC ANALYSIS METHODS (→ Move to analysis/)
    def _dendritic_ablation_all(self)                                     # Line 994-1142
    def _dendritic_information(self)                                      # Line 1143-1420
    def _dendritic_information_inputs(self)                               # Line 1421-1563
    def _dendritic_information_activations(self)                         # Line 1564-1781
    
    # UTILITY METHODS (→ Move to utils/ or keep in PlotManager)
    def _train_mlp_classifier(self, train_dataset, valid_dataset, ...)   # Line 446-486
```

## Refactoring Plan

### 1. **Plotting Methods** ✅ ALREADY HANDLED
- **Status**: Already extracted to `plotting/` package
- **Target**: Keep references in PlotManager, delegate to plotting package
- **Methods**: `_plot_weights`, `_plot_activations`, `_plot_gradients`, `_plot_profiles`

### 2. **Branch Analysis Methods** → `analysis/branches/`
- **Target Directory**: `src/dendritic_modeling/analysis/branches/`
- **New Module**: `analysis/branches/interactive.py`
- **Class**: `InteractiveBranchAnalyzer`

```python
# → analysis/branches/interactive.py
class InteractiveBranchAnalyzer:
    def analyze_branch_info(self, model, data, device, save_path)         # From _branch_info
    def analyze_branch_ablation(self, model, data, device, save_path)     # From _branch_ablation
    def _train_mlp_classifier(self, train_dataset, valid_dataset, ...)    # Utility method
```

### 3. **Dendritic Analysis Methods** → `analysis/dendritic/`
- **Target Directory**: `src/dendritic_modeling/analysis/dendritic/`
- **New Module**: `analysis/dendritic/comprehensive.py`
- **Class**: `ComprehensiveDendriticAnalyzer`

```python
# → analysis/dendritic/comprehensive.py  
class ComprehensiveDendriticAnalyzer:
    def analyze_dendritic_ablation_all(self, model, data, save_path)      # From _dendritic_ablation_all
    def analyze_dendritic_information(self, model, data, save_path)       # From _dendritic_information
    def analyze_dendritic_information_inputs(self, model, data, save_path) # From _dendritic_information_inputs
    def analyze_dendritic_information_activations(self, model, data, save_path) # From _dendritic_information_activations
```

### 4. **Refactored PlotManager** → `analysis/managers/`
- **Target**: `src/dendritic_modeling/analysis/managers/plot_manager.py`
- **Class**: `PlotManager` (slimmed down)

```python
# → analysis/managers/plot_manager.py
class PlotManager:
    def __init__(self, model, data, args, device="cpu")
    def run_plot(self, plot_name, epoch=None)
    
    # Delegate to organized modules
    def _plot_weights(self)          # → plotting.network
    def _plot_activations(self)      # → plotting.analysis  
    def _plot_gradients(self)        # → plotting.network
    def _plot_profiles(self)         # → plotting.analysis
    def _branch_info(self)           # → analysis.branches.interactive
    def _branch_ablation(self)       # → analysis.branches.interactive
    def _dendritic_ablation_all(self) # → analysis.dendritic.comprehensive
    def _dendritic_information(self) # → analysis.dendritic.comprehensive
    # ... etc
```

## Dependencies to Update

### Files that import from visualization_manager.py:
```python
# These files will need import updates:
from dendritic_modeling.visualization_manager import PlotManager
# → 
from dendritic_modeling.analysis.managers.plot_manager import PlotManager
```

### New Import Structure:
```python
# For direct access to new analyzers:
from dendritic_modeling.analysis.branches.interactive import InteractiveBranchAnalyzer
from dendritic_modeling.analysis.dendritic.comprehensive import ComprehensiveDendriticAnalyzer
```

## File Size Reduction Target

- **Original**: `visualization_manager.py` (1781 lines)
- **Target**: `analysis/managers/plot_manager.py` (~100-200 lines, delegation only)
- **Extracted**: 
  - `analysis/branches/interactive.py` (~600-700 lines)
  - `analysis/dendritic/comprehensive.py` (~800-900 lines)
- **Reduction**: ~90% size reduction in main file

## Key Requirements

1. **Backward Compatibility**: Maintain existing `PlotManager` interface
2. **Code Organization**: Group related functionality into focused modules  
3. **Clean Separation**: Plotting vs Analysis vs Management concerns
4. **Documentation**: Each new module needs comprehensive docstrings
5. **Testing**: Ensure imports work after refactoring

## Validation Checklist

- [ ] Extract branch analysis methods to `analysis/branches/interactive.py`
- [ ] Extract dendritic analysis methods to `analysis/dendritic/comprehensive.py`
- [ ] Create slim `analysis/managers/plot_manager.py` with delegation
- [ ] Update imports in all dependent files
- [ ] Test that existing `PlotManager` interface still works
- [ ] Verify file size reduction (>90%)
- [ ] Add comprehensive documentation to all new modules
- [ ] Update `analysis/__init__.py` to export new classes

## Notes

- The `_train_mlp_classifier` utility method should probably stay with the `InteractiveBranchAnalyzer` since it's specifically used by branch analysis
- Consider whether some methods should use the existing organized analysis tools from `analysis/manager.py` instead of duplicating functionality
- The large amount of duplicate print statements and logging suggests opportunity for creating common analysis utilities 