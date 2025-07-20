# Unified Sweep Framework for Dendritic Modeling

This unified framework consolidates all parameter sweep functionality for the dendritic modeling project. It replaces the previous separate sweep systems with a single, cohesive interface.

##  Key Benefits

- **Single Interface**: One command-line tool for all sweep types
- **Consistent Configuration**: Standardized YAML format across all sweeps
- **Unified Analysis**: Common analysis pipeline with type-specific visualizations
- **Reduced Code Duplication**: ~60% reduction in codebase size
- **Better Error Handling**: Comprehensive validation and recovery
- **Backward Compatibility**: Legacy scripts still work with deprecation warnings

##  Framework Structure

```
scripts/sweeps/unified/
├── sweep_manager.py          # Main entry point for all operations
├── result_analyzer.py        # Unified result analysis
├── base_sweep.py            # Base classes and common utilities
├── sweep_types/             # Type-specific implementations
│   ├── ei_sweep.py          # EI parameter sweeps
│   ├── branch_sweep.py      # Branch architecture sweeps
│   ├── noise_sweep.py       # Noise robustness sweeps
│   └── general_sweep.py     # General parameter sweeps
└── README.md               # This documentation
```

##  Quick Start

### 1. Basic Usage

```bash
# Generate and run an EI sweep
python scripts/sweeps/unified/sweep_manager.py ei \
    --config configs/examples/unified_ei_sweep_example.yaml \
    --run

# Generate configs only (no job submission)
python scripts/sweeps/unified/sweep_manager.py branch \
    --config configs/branch_sweep.yaml \
    --generate-only

# Analyze existing results
python scripts/sweeps/unified/result_analyzer.py \
    --results-dir output/ei_20240101120000 \
    --auto-detect
```

### 2. Advanced Options

```bash
# Split sweep into multiple jobs
python scripts/sweeps/unified/sweep_manager.py general \
    --config configs/general_sweep.yaml \
    --split-jobs 10 \
    --run

# Dry run (show what would be submitted)
python scripts/sweeps/unified/sweep_manager.py ei \
    --config configs/ei_sweep.yaml \
    --run --dry-run

# Analyze multiple sweep results
python scripts/sweeps/unified/result_analyzer.py \
    --results-dirs output/ei_* output/branch_* \
    --auto-detect \
    --comparison-report comparison.txt
```

##  Configuration Format

All sweep types use a standardized configuration format:

```yaml
# Output and SLURM configuration
output_dir: "output"
slurm_config:
  job_name: "my_sweep"
  account: "kempner_dev"
  partition: "kempner"
  time: "02:00:00"
  # ... other SLURM options

# Experiment settings
experiment_settings:
  seeds_per_condition: 3
  base_seed: 42

# Parameters to sweep (type-specific)
sweep_config:
  parameter.path.1: [value1, value2, value3]
  parameter.path.2: [valueA, valueB]

# Base training configuration
base_config:
  task:
    dataset: "multixor"
  model:
    network:
      type: "EINet"
  train:
    optimizer: "adam"
    # ... other training settings
```

## 🔬 Sweep Types

### 1. EI (Excitation-Inhibition) Sweeps

**Purpose**: Explore E/I parameter spaces in dendritic networks.

**Key Parameters**:
- `model.network.parameters.ee_synapses_per_branch_per_layer`
- `model.network.parameters.ie_synapses_per_branch_per_layer`
- `model.network.parameters.use_shunting`

**Filtering Options**:
```yaml
filter_config:
  mode: "ei-ratio"  # "all", "ei-sum", "e-const", "i-const", "ei-ratio"
  value: 2.0
  tolerance: 0.1
```

**Analysis Features**:
- Performance heatmaps vs E/I parameters
- Network type comparison (EINet vs MLP)
- E/I ratio analysis
- Line plots for parameter exploration

### 2. Branch Sweeps

**Purpose**: Analyze branch architecture effects.

**Key Parameters**:
- `model.network.parameters.num_branches_per_layer`
- `model.network.parameters.num_layers`
- `model.network.parameters.excitatory_branch_factors`

**Analysis Features**:
- Branch count vs performance
- Architectural complexity analysis
- Shunting vs non-shunting comparison

### 3. Noise Sweeps

**Purpose**: Test robustness to different noise types.

**Key Parameters**:
- `noise.type`: ["gaussian", "uniform", "poisson", "frequency"]
- `noise.level`: [0.0, 0.1, 0.2, 0.3]
- `noise.application_mode`

**Analysis Features**:
- Robustness curves
- Performance degradation analysis
- Noise type comparison
- Area Under Robustness Curve (AURC) metrics

### 4. General Sweeps

**Purpose**: Sweep arbitrary parameters.

**Configuration**: Any parameter path with list of values.

**Analysis Features**:
- Parameter effect plots
- Correlation analysis
- Parameter heatmaps
- Performance distributions

## 📊 Analysis Pipeline

The unified analysis pipeline provides:

1. **Automatic Result Collection**: Finds and aggregates results across configurations
2. **Type-Specific Visualizations**: Generates plots tailored to each sweep type
3. **Summary Reports**: Creates comprehensive analysis summaries
4. **Best Configuration Identification**: Automatically finds top-performing setups
5. **Missing Data Detection**: Reports failed or incomplete experiments

##  Migration from Legacy Scripts

### Automatic Migration

Most legacy configurations can be automatically converted:

```bash
# Legacy EI sweep
python scripts/sweeps/ei_sweeps/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml

# Becomes:
python scripts/sweeps/unified/sweep_manager.py ei --config configs/sweep_params_seeds_ei.yaml --run
```

### Configuration Updates

Legacy configurations may need minor updates:

1. **Rename sections**: `run_config` → `slurm_config`
2. **Add sweep type**: Add `sweep_config` section
3. **Update paths**: Use dot notation for nested parameters

### Backward Compatibility

Legacy scripts still work but show deprecation warnings:

```bash
 python scripts/sweeps/ei_sweeps/generate_ei_sweep.py config.yaml
  WARNING: This script is deprecated. Use the unified framework:
   python scripts/sweeps/unified/sweep_manager.py ei --config config.yaml --run
```

## 📈 Performance Improvements

The unified framework provides significant improvements:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines of Code | ~10,300 | ~4,000 | 60% reduction |
| Number of Scripts | 10 | 2 main scripts | 80% reduction |
| Configuration Formats | 3 different | 1 unified | 100% consolidation |
| Error Handling | Inconsistent | Unified | Comprehensive |

## 🛠️ Development

### Adding New Sweep Types

1. Create new sweep implementation in `sweep_types/`:
```python
from base_sweep import BaseSweepGenerator, BaseSweepAnalyzer

class MySweepGenerator(BaseSweepGenerator):
    def __init__(self):
        super().__init__("my_sweep")
    
    def validate_config(self, config):
        # Validation logic
        pass
    
    def generate_configs(self, base_config):
        # Config generation logic
        pass

class MySweepAnalyzer(BaseSweepAnalyzer):
    # Analysis implementation
    pass
```

2. Register in `sweep_manager.py`:
```python
elif self.sweep_type == "my_sweep":
    from sweep_types.my_sweep import MySweepGenerator
    return MySweepGenerator()
```

### Testing

```bash
# Test configuration validation
python scripts/sweeps/unified/sweep_manager.py my_sweep \
    --config configs/my_sweep.yaml --generate-only

# Test with dry run
python scripts/sweeps/unified/sweep_manager.py my_sweep \
    --config configs/my_sweep.yaml --run --dry-run
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Check Python path and dependencies
2. **Configuration Validation Failures**: Verify required sections exist
3. **SLURM Job Failures**: Check account, partition, and resource limits
4. **Missing Results**: Use `--auto-detect` for analysis
5. **Path Issues**: Use absolute paths for configs when in doubt

### Debug Mode

Add verbose logging:
```bash
export PYTHONPATH="src:$PYTHONPATH"
python -v scripts/sweeps/unified/sweep_manager.py ...
```

### Getting Help

```bash
# General help
python scripts/sweeps/unified/sweep_manager.py --help

# Sweep-specific help
python scripts/sweeps/unified/sweep_manager.py ei --help

# Analysis help
python scripts/sweeps/unified/result_analyzer.py --help
```

##  Examples

See `configs/examples/` for complete configuration examples:

- `unified_ei_sweep_example.yaml`: EI parameter sweep
- `unified_branch_sweep_example.yaml`: Branch architecture sweep
- `unified_noise_sweep_example.yaml`: Noise robustness sweep
- `unified_general_sweep_example.yaml`: General parameter sweep

---

*For questions or issues, please refer to the project documentation or create an issue in the repository.* 