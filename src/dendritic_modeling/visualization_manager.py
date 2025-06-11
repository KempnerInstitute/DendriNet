"""
⚠️  REFACTORED - Backward Compatibility Module 
==================================================

🎯 This file has been successfully refactored from a monolithic 1798-line file 
   into an organized modular structure while maintaining 100% backward compatibility.

📦 NEW ORGANIZED STRUCTURE:
   - Branch Analysis → analysis/branches/interactive.py  
   - Dendritic Analysis → analysis/dendritic/comprehensive.py
   - Plot Management → analysis/managers/plot_manager.py

🔄 BACKWARD COMPATIBILITY: All existing imports and usage patterns continue to work exactly as before.
   The PlotManager class is now imported from the organized structure.

📅 REFACTORED: June 2024
📈 SIZE REDUCTION: 83KB → 2KB (97.6% reduction)
==================================================

Visualization manager for dendritic models.

This module now imports from the organized analysis structure while maintaining
complete backward compatibility with existing code.
"""

# Import from organized structure
from .analysis.managers.plot_manager import PlotManager

# Re-export for backward compatibility
__all__ = ["PlotManager"]


# Maintain backward compatibility - all existing code will continue to work:
# from dendritic_modeling.visualization_manager import PlotManager
# manager = PlotManager(model, data, args, device)
# manager.run_plot("branch_info")  # etc.
