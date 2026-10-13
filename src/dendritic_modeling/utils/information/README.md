# Information utilities

Low-level information-theoretic estimators used by the `information` analyzer
(`dendritic_modeling.analysis.core.information.InformationAnalyzer`). Use these
functions to estimate information from arrays directly; configure the analyzer
under `analysis.information` to analyze a model (see the `information` section
of `docs/analyzers.rst`).

## Modules

- `nearest_neighbors.py`: k-nearest-neighbour estimators of mutual information
  between continuous variables (`compute_mi_cc`), between a continuous and a
  discrete variable (`compute_mi_cd`), of conditional mutual information
  (`compute_cmi_ccc`), and synaptic-information helpers built on them. Results
  are in nats; negative estimates are replaced by 0.
- `estimators.py`: the estimator classes behind `analysis.information.params.estimator.method`
  (Kraskov k-nearest-neighbour, binned, cross-validated decoder,
  scikit-learn and copula estimators) and `InformationEstimatorFactory`, which
  builds one from an `EstimatorConfig`.
- `pid.py`: partial information decomposition of two sources about a target
  with the minimum-mutual-information measure: redundancy
  `min(I(S1;T), I(S2;T))`, unique terms `I(Si;T)` minus the redundancy, and
  synergy `I(S1,S2;T) - I(S1;T) - I(S2;T) + redundancy`.

## Public exports

`dendritic_modeling.utils.information` exports:

- `PIDTerms`
- `build_sr_matrix`
- `compute_cmi_ccc`
- `compute_ei_mi`
- `compute_mi_cc`
- `compute_mi_cd`
- `compute_pairwise_sr`
- `compute_pairwise_sr_with_mi`
- `compute_synaptic_information`
- `mmi_pid`
- `partial_information_decomposition`

## Example

```python
import numpy as np

from dendritic_modeling.utils.information import (
    compute_mi_cc,
    compute_mi_cd,
    partial_information_decomposition,
)

rng = np.random.default_rng(0)
x = rng.normal(size=(1000, 4))
y = x + 0.5 * rng.normal(size=(1000, 4))
labels = rng.integers(0, 10, size=1000)

mi_xy = compute_mi_cc(x, y, n_neighbors=10)  # nats
mi_xc = compute_mi_cd(x, labels, n_neighbors=10)  # nats

terms = partial_information_decomposition(
    x[:, :2],
    x[:, 2:],
    labels,
    lambda a, b: compute_mi_cd(a, b, n_neighbors=10),
)
print(terms.redundancy, terms.synergy)
```

`partial_information_decomposition` accepts any estimator `f(x, y)` with
samples in rows. The terms inherit the estimator's bias: with estimated
informations the synergy can be slightly negative, and it is reported as is.
