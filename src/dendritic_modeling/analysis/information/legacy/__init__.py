"""Legacy information analysis modules.

DEPRECATED: These modules are maintained for backward compatibility only.
Please use the new integrated framework instead:
- IntegratedInformationAnalyzer for all information analysis needs
- See README.md for migration guide
"""

import warnings

# Import with deprecation warnings
def _deprecated_import(module_name, class_name):
    warnings.warn(
        f"{class_name} is deprecated and will be removed in a future version. "
        f"Please use IntegratedInformationAnalyzer from the main information module instead.",
        DeprecationWarning,
        stacklevel=3
    )
    
# Import the legacy modules
from .mutual_info import EIInformationAnalyzer
from .branch_info import BranchInformationAnalyzer

# Show deprecation warnings on import
_deprecated_import("mutual_info", "EIInformationAnalyzer")
_deprecated_import("branch_info", "BranchInformationAnalyzer")

__all__ = [
    "EIInformationAnalyzer",
    "BranchInformationAnalyzer",
] 