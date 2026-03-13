# Information Utilities

This directory contains the low-level information-theoretic utilities used by
the higher-level analyzer in
`src/dendritic_modeling/analysis/core/information.py`.

## Scope

Use the code here when you need estimator primitives or PID helpers directly.
Use `InformationAnalyzer` when you want model-level analysis driven by the main
config system.

## Modules

- `nearest_neighbors.py`: k-NN mutual information and conditional mutual
  information helpers
- `estimators.py`: estimator wrappers and factory logic
- `pid.py`: PID availability checks and PID helper functions

## Public Exports

The package currently re-exports:

- `build_sr_matrix`
- `check_pid_available`
- `compute_cmi_ccc`
- `compute_ei_mi`
- `compute_mi_cc`
- `compute_mi_cd`
- `compute_pairwise_sr`
- `compute_pairwise_sr_with_mi`
- `compute_synaptic_information`
- `pid_synergy_redundancy`

## Minimal Example

```python
import numpy as np

from dendritic_modeling.utils.information import compute_mi_cc, compute_mi_cd

x = np.random.randn(1000, 4)
y = np.random.randn(1000, 4)
labels = np.random.randint(0, 10, size=1000)

mi_xy = compute_mi_cc(x, y, n_neighbors=10)
mi_xc = compute_mi_cd(x, labels, n_neighbors=10)
```

## Higher-Level Analysis

For model analysis, use:

- `dendritic_modeling.analysis.core.information.InformationAnalyzer`
- `dendritic_modeling.config.InformationAnalysisParams`

The maintained user-facing documentation for that workflow is in
`docs/analysis.rst` and `docs/CONFIGURATION_GUIDE.rst`.
