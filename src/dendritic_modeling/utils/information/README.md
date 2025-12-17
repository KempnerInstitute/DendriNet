# Information Analysis Framework

This module provides a unified framework for computing information-theoretic metrics in dendritic networks using multiple estimation methods.

## Overview

The framework supports:
- Multiple estimation methods: Kraskov (k-NN), scikit-learn, copula-based
- Comprehensive metrics: MI, conditional MI, joint MI, PID
- Configurable analysis: Choose which metrics to compute
- Method comparison: Easy benchmarking of different estimators

## Quick Start

```python
from dendritic_modeling.analysis.information import (
    InformationAnalyzer,
    InformationAnalysisPreset
)

# Create analyzer with full analysis preset
config = InformationAnalysisPreset.full_analysis(method="kraskov")
analyzer = InformationAnalyzer(config)

# Run analysis on your data
results = analyzer.analyze(E, I, Vout, C, Vinf)

# Print summary
print(analyzer.get_summary(results))
```

## Available Metrics

### 1. Basic Mutual Information with C
- `I(E;C)` - MI between excitatory inputs and class labels
- `I(I;C)` - MI between inhibitory inputs and class labels
- `I(Vb;C)` - MI between branch inputs and class labels (hierarchical networks)
- `I(Vout;C)` - MI between branch outputs and class labels
- `I(Vinf;C)` - MI between soma activation and class labels (optional)
- `I(E,I;C)` - Joint MI between E&I and class labels

### 2. Pairwise Mutual Information
- `I(E;I)` - MI between excitatory and inhibitory inputs
- `I(E;Vb)` - MI between excitatory inputs and branch inputs
- `I(I;Vb)` - MI between inhibitory inputs and branch inputs
- `I(E;Vout)` - MI between excitatory inputs and outputs
- `I(I;Vout)` - MI between inhibitory inputs and outputs
- `I(Vb;Vout)` - MI between branch inputs and outputs
- `I(E;Vinf)`, `I(I;Vinf)` - MI with soma activation (optional)

### 3. Conditional Mutual Information
All pairwise metrics conditioned on C:
- `I(E;I|C)` - Conditional MI between E and I given C
- `I(E;Vb|C)` - Conditional MI between E and branch input given C
- `I(I;Vb|C)` - Conditional MI between I and branch input given C
- `I(E;Vout|C)` - Conditional MI between E and Vout given C
- `I(I;Vout|C)` - Conditional MI between I and Vout given C
- `I(Vb;Vout|C)` - Conditional MI between branch input and Vout given C

### 4. Partial Information Decomposition (PID)
For variable pairs (X,Y) → Z:
- Synergy - Information only available from X and Y together
- Redundancy - Information available from both X and Y
- Unique(X) - Information only from X
- Unique(Y) - Information only from Y

Computed for:
- `PID(E,I→C)` - How E and I contribute to class information
- `PID(E,Vout→C)` - How E and output contribute to class information
- `PID(I,Vout→C)` - How I and output contribute to class information
- `PID(E,I→Vout)` - How E and I contribute to output

## Estimation Methods

### Kraskov (k-NN) Estimator
- Best for: General continuous data
- Parameters: `n_neighbors` (default: 10)
- Pros: Non-parametric, handles multivariate well
- Cons: Can be slow for large datasets

### Scikit-learn Estimator
- Best for: Mixed discrete/continuous data
- Parameters: `n_bins` (default: 20) for histogram method
- Pros: Fast, handles discrete variables well
- Cons: Less accurate for high dimensions

### Copula Estimator
- Best for: Data with complex dependencies
- Parameters: `copula_type` (default: "gaussian")
- Pros: Captures non-linear dependencies
- Cons: Limited to bivariate currently

## Configuration

### Using Presets

```python
# Full analysis with all metrics
config = InformationAnalysisPreset.full_analysis(method="kraskov")

# Basic MI only
config = InformationAnalysisPreset.basic_mi_analysis(method="sklearn")

# Conditional MI focus
config = InformationAnalysisPreset.conditional_analysis(method="copula")

# PID focus
config = InformationAnalysisPreset.pid_analysis(method="kraskov")
```

### Using ConfigBuilder

```python
from dendritic_modeling.analysis.information import ConfigBuilder

config = ConfigBuilder() \
    .with_method("kraskov") \
    .with_metrics(basic=True, pairwise=True, conditional=True, pid=True) \
    .with_kraskov_params(n_neighbors=5) \
    .with_pid_params(binarize=True, binarize_method="median") \
    .with_options(normalize_mi=True, verbose=True) \
    .build()
```

### From Main Config System (Recommended)

```python
from dendritic_modeling.config import load_config
from dendritic_modeling.analysis.information import (
    InformationAnalyzer,
    create_analyzer_config_from_main_config
)

# Load YAML config
config = load_config("configs/example_information_analysis.yaml")

# Create analyzer from main config
info_config = create_analyzer_config_from_main_config(
    config.analysis.information_analysis
)
analyzer = InformationAnalyzer(info_config)

# Run analysis
results = analyzer.analyze(E, I, Vout, C, Vinf)
```

### From Dictionary

```python
config_dict = {
    "method": "sklearn",
    "n_bins": 30,
    "compute_basic_mi": True,
    "compute_pairwise_mi": True,
    "compute_conditional_mi": False,
    "compute_pid": True,
    "normalize_mi": True,
}

# Convert to analyzer config
from dendritic_modeling.analysis.information import (
    create_analyzer_config_from_main_config
)

info_config = create_analyzer_config_from_main_config(config_dict)
analyzer = InformationAnalyzer(info_config)
```

## Usage Examples

### Basic Analysis

```python
import numpy as np
from dendritic_modeling.analysis.information import (
    InformationAnalyzer,
    InformationAnalysisPreset
)

# Your data
E = excitatory_inputs      # Shape: (n_samples,) or (n_samples, n_features)
I = inhibitory_inputs      # Shape: (n_samples,) or (n_samples, n_features)
Vout = branch_outputs      # Shape: (n_samples,) or (n_samples, n_features)
C = class_labels          # Shape: (n_samples,)
Vinf = soma_activation    # Shape: (n_samples,) or (n_samples, n_features)

# Create and run analyzer
config = InformationAnalysisPreset.full_analysis()
analyzer = InformationAnalyzer(config)
analyzer.analyze(model, test_dataset, device, save_path, filename)
```

### Method Comparison

```python
methods = ["kraskov", "sklearn", "copula"]
results_by_method = {}

for method in methods:
    config = ConfigBuilder().with_method(method).build()
    analyzer = InformationAnalyzer(config)
    results_by_method[method] = analyzer.analyze(E, I, Vout, C)

# Compare results
for method, results in results_by_method.items():
    print(f"{method}: I(E;C) = {results['basic_mi']['I(E;C)']:.4f}")
```

### Custom Analysis Pipeline

```python
# Configure for specific metrics only
config = ConfigBuilder() \
    .with_method("kraskov") \
    .with_metrics(basic=False, pairwise=True, conditional=True, pid=False) \
    .with_kraskov_params(n_neighbors=15) \
    .build()

analyzer = InformationAnalyzer(config)

# Analyze multiple conditions
for condition in experimental_conditions:
    data = load_data(condition)
    results = analyzer.analyze(data.E, data.I, data.Vout, data.C)
    save_results(results, condition)
```

## Integration with Existing Code

The framework supports integration with existing analyzers:

```python
# Standard imports
from dendritic_modeling.analysis.information import (
    EIInformationAnalyzer,
    BranchInformationAnalyzer,
    compute_mi_cc,
    compute_mi_cd
)

# Using the framework instead
from dendritic_modeling.analysis.information import (
    InformationAnalyzer,
    KraskovEstimator,
    EstimatorConfig
)
```

## Advanced Features

### Branch Input Analysis

For hierarchical dendritic networks, the framework can analyze upstream dendritic inputs (branch inputs) in addition to excitatory and inhibitory synaptic inputs.

```python
config = ConfigBuilder() \
    .with_method("kraskov") \
    .with_metrics(basic=True, pairwise=True, conditional=True) \
    .build()

analyzer = InformationAnalyzer(config)

# Analyze with branch inputs
results = analyzer.analyze(E, I, Vout, C, Vinf=None, Vb=branch_inputs)
```

Branch input (Vb) represents signals from upstream branches in hierarchical networks. When enabled, the following additional metrics are computed:

- Basic: `I(Vb;C)`
- Pairwise: `I(E;Vb)`, `I(I;Vb)`, `I(Vb;Vout)`
- Conditional: `I(E;Vb|C)`, `I(I;Vb|C)`, `I(Vb;Vout|C)`

For single-layer networks without hierarchical structure, branch inputs are zero and these metrics provide baseline comparisons.

### Linear Discriminant Analysis Weights

The framework can compute alternative synaptic aggregations using Linear Discriminant Analysis (LDA) weights instead of network-learned weights.

```python
# Enable LDA-based weight computation
config = ConfigBuilder() \
    .with_method("kraskov") \
    .with_metrics(basic=True, pairwise=True) \
    .with_options(compute_lda_weights=True) \
    .build()

analyzer = InformationAnalyzer(config)
results = analyzer.analyze(E, I, Vout, C, Vinf=None, Vb=branch_inputs)
```

When LDA weights are enabled, the system:

1. Fits separate LDA models on raw synaptic inputs (treating each synapse as a feature)
2. Computes optimal linear combinations that maximize class separation
3. Creates three LDA-weighted signals: E_lin, I_lin, Vb_lin
4. Computes MI for both network-learned and LDA-optimized aggregations

Results include metrics for both versions:

Network-learned weights:
- `I(E;C)`, `I(I;C)`, `I(Vb;C)`
- Standard pairwise and conditional MI

LDA-optimized weights:
- `I(E_lin;C)`, `I(I_lin;C)`, `I(Vb_lin;C)`
- `I(E_lin;I_lin)`, `I(E_lin;Vb_lin)`, etc.
- `I(E_lin;I_lin|C)`, etc.

This comparison reveals whether the network has learned synaptic weights approaching the discriminant-optimal configuration or whether additional structure beyond class discrimination is encoded.

### Custom Estimators

Adding custom estimation methods:

```python
from dendritic_modeling.analysis.information import (
    BaseInformationEstimator,
    InformationEstimatorFactory,
    EstimationMethod
)

class MyCustomEstimator(BaseInformationEstimator):
    def mutual_information(self, x, y):
        # Your implementation
        pass

    def conditional_mutual_information(self, x, y, z):
        # Your implementation
        pass

# Register it
InformationEstimatorFactory.register_estimator(
    EstimationMethod.CUSTOM,
    MyCustomEstimator
)
```

### Batch Processing

```python
# Process multiple datasets efficiently
datasets = load_all_datasets()
analyzer = InformationAnalyzer(config)

results_list = []
for dataset in datasets:
    results = analyzer.analyze(
        dataset.E, dataset.I, dataset.Vout, dataset.C
    )
    results_list.append(results)

# Aggregate results
aggregate_results(results_list)
```

## API Reference

### Main Classes

- `InformationAnalyzer` - Main analyzer class
- `AnalyzerConfig` - Configuration dataclass
- `EstimatorConfig` - Estimator-specific configuration
- `BaseInformationEstimator` - Abstract base for estimators
- `ConfigBuilder` - Fluent configuration builder
- `InformationAnalysisPreset` - Common configuration presets

### Key Methods

- `analyzer.analyze(E, I, Vout, C, Vinf=None)` - Run full analysis
- `analyzer.get_summary(results)` - Get readable summary
- `ConfigBuilder.build()` - Build configuration
- `load_config_from_dict(dict)` - Load from dictionary
- `save_config_to_dict(config)` - Save to dictionary

## Performance Tips

1. Choose the right method:
   - Use Kraskov for general continuous data
   - Use sklearn for mixed/discrete data or speed
   - Use copula for complex dependencies

2. Optimize parameters:
   - Reduce `n_neighbors` for faster Kraskov estimation
   - Reduce `n_bins` for faster sklearn histogram method
   - Disable unnecessary metrics with `compute_*=False`

3. Handle large datasets:
   - Consider subsampling for initial exploration
   - Use sklearn method for faster computation
   - Process in batches if memory is limited

## Troubleshooting

### PID not available
If you see "PID package not available", ensure the external PID library is installed in the `external` directory.

### Slow computation
- Reduce `n_neighbors` for Kraskov method
- Switch to sklearn method for faster estimation
- Disable metrics you don't need

### Memory issues
- Process data in smaller batches
- Use single-feature arrays when possible
- Consider downsampling for large datasets
