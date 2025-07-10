# Information-Theoretic Analysis Guide for Dendritic Networks

This guide explains how to train Excitation-Inhibition (EI) networks and compute information-theoretic measures between network components.

## Quick Start

### 1. Basic Training with Information Analysis

To train an EI network and compute information values, use the quick-start configuration:

```bash
python -m dendritic_modeling.scripts.train_experiments \
    --config configs/ei_network_info_analysis_quickstart.yaml
```

This will:
- Train a small EI network on MNIST for 20 epochs
- Compute mutual information between E/I inputs, branch outputs, and class labels
- Save results to `./outputs/ei_info_quickstart/`

### 2. Understanding the Results

After training, check the output directory:

```
outputs/ei_info_quickstart/
├── models/           # Saved model checkpoints
├── logs/             # Training logs
└── analysis/         # Analysis results
    ├── information_analysis/
    │   ├── information_analysis_results.json  # Raw MI values
    │   └── information_analysis_summary.txt   # Human-readable summary
    └── performance_analysis/
        └── test_metrics.json                  # Final accuracy
```

The summary file will contain values like:
- `I(E;C)`: Mutual information between excitatory inputs and class labels
- `I(I;C)`: Mutual information between inhibitory inputs and class labels
- `I(E;I)`: Mutual information between E and I (their dependence)
- `I(Vout;C)`: Information carried by branch outputs about the class
- `I(Vinf;C)`: Information at the soma level

## Configuration Options

### Information Analysis Parameters

The key configuration section is `analysis.information_analysis`:

```yaml
analysis:
  information_analysis:
    # Estimation method
    method: kraskov      # Options: kraskov, sklearn, copula, auto
    
    # Method-specific parameters
    n_neighbors: 10      # For kraskov (k-NN estimator)
    n_bins: 20          # For sklearn (histogram bins)
    copula_type: gaussian # For copula
    
    # What to compute
    compute_basic_mi: true      # I(X;C) for all variables X
    compute_pairwise_mi: true   # I(X;Y) between all pairs
    compute_conditional_mi: true # I(X;Y|C) conditioned on class
    compute_pid: false          # Partial Information Decomposition
    
    # Additional options
    include_vinf: true     # Include soma activation
    normalize_mi: false    # Normalize by entropy
    verbose: true         # Print progress
```

### Choosing an Estimation Method

1. **Kraskov** (Recommended for most cases)
   - Best for: Continuous multivariate data
   - Pros: Non-parametric, handles high dimensions
   - Cons: Can be slow for large datasets
   ```yaml
   method: kraskov
   n_neighbors: 10  # 5-15 is typical
   ```

2. **Sklearn** (Fast but less accurate)
   - Best for: Quick exploration or mixed data types
   - Pros: Fast, handles discrete variables
   - Cons: Less accurate for complex distributions
   ```yaml
   method: sklearn
   n_bins: 20  # Adjust based on data size
   ```

3. **Copula** (For complex dependencies)
   - Best for: Non-linear relationships
   - Pros: Captures complex dependencies
   - Cons: Currently limited to bivariate
   ```yaml
   method: copula
   copula_type: gaussian  # or t, clayton, gumbel, frank
   ```

## Network Architecture for Information Analysis

For meaningful information analysis, configure your EI network appropriately:

```yaml
model:
  network:
    type: EINet
    parameters:
      # Use input_mode 0 for clear E/I separation
      input_mode: 0  # E gets input, I gets nothing
      
      # Enable shunting for biologically plausible computation
      use_shunting: true  # V = E/(E+I+1)
      
      # Ensure sufficient connectivity
      ee_synapses_per_branch_per_layer: [50]   # E→E
      ei_synapses_per_branch_per_layer: [100]  # E→I
      ie_synapses_per_branch_per_layer: [20]   # I→E
      ii_synapses_per_branch_per_layer: [0]    # I→I (usually 0)
```

## Advanced Use Cases

### 1. Detailed Analysis (Slow but Thorough)

```yaml
analysis:
  information_analysis:
    method: kraskov
    n_neighbors: 15
    compute_basic_mi: true
    compute_pairwise_mi: true
    compute_conditional_mi: true
    compute_pid: true  # Requires external PID package
    verbose: true
```

### 2. Quick Exploration (Fast)

```yaml
analysis:
  information_analysis:
    method: sklearn
    n_bins: 15
    compute_basic_mi: true
    compute_pairwise_mi: false
    compute_conditional_mi: false
    compute_pid: false
```

### 3. Focus on E/I Interactions

```yaml
analysis:
  information_analysis:
    method: kraskov
    n_neighbors: 10
    compute_basic_mi: false     # Skip I(X;C)
    compute_pairwise_mi: true   # Focus on I(E;I), etc.
    compute_conditional_mi: true # I(E;I|C)
    compute_pid: false
```

## Interpreting Results

### Basic Mutual Information
- **I(E;C)**: How much excitatory activity predicts the class
  - High values (>1.5 nats): Strong class information in E
  - Low values (<0.5 nats): Weak class information
  
- **I(I;C)**: How much inhibitory activity predicts the class
  - Often lower than I(E;C) if I serves modulatory role
  
- **I(Vout;C)**: Information at branch outputs
  - Should be ≥ max(I(E;C), I(I;C)) if integration is effective

### Pairwise Information
- **I(E;I)**: Dependence between E and I
  - High values: E and I are strongly coupled
  - Low values: E and I process information independently
  
- **I(E;Vout)**, **I(I;Vout)**: How inputs influence outputs
  - Indicates which pathway contributes more

### Conditional Information
- **I(E;I|C)**: E/I dependence after accounting for class
  - High values: E/I interact differently for different classes
  - Low values: E/I relationship is class-independent

## Common Issues and Solutions

### 1. Low Information Values
- **Problem**: All MI values are near zero
- **Solutions**:
  - Check if network is trained properly (accuracy > chance)
  - Increase connectivity (more synapses)
  - Use `input_mode: 0` for clear E/I separation
  - Ensure sufficient data samples

### 2. Estimation Errors
- **Problem**: Negative MI values or unrealistic numbers
- **Solutions**:
  - Increase `n_neighbors` for kraskov method
  - Use more data samples
  - Try different estimation method
  - Check for numerical issues in activations

### 3. Slow Computation
- **Problem**: Analysis takes too long
- **Solutions**:
  - Reduce test set size
  - Use sklearn method for quick estimates
  - Disable conditional MI or PID
  - Use fewer neighbors for kraskov

## Example Configurations

### Minimal Configuration
See: `configs/ei_network_info_analysis_quickstart.yaml`

### Detailed Analysis
See: `configs/example_information_analysis_documented.yaml`

### Parameter Sweeps
See: `.vscode/sweep_params_seeds_ei_documented.yaml`

## Running Parameter Sweeps

To systematically explore how network parameters affect information flow:

1. Create a sweep configuration (see documented examples)
2. Use SLURM scripts for HPC clusters
3. Aggregate results across seeds for statistical analysis

## Best Practices

1. **Always verify model performance first**
   - Ensure accuracy is above chance before analyzing information

2. **Use multiple seeds**
   - Information values can vary; use 3-5 seeds minimum

3. **Start simple**
   - Begin with basic MI before adding conditional/PID

4. **Validate with known cases**
   - Test on simple datasets where expected MI is known

5. **Document your configuration**
   - Use the documented YAML files as templates
   - Record all parameter choices for reproducibility 