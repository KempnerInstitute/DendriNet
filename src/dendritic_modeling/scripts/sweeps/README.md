# Dendritic Modeling Sweep System

## Quick Start

**1. Run a Sweep:**
```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/unified_all_sweeps_example.yaml --run
```

**2. Analyze Results:**
```bash
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    --results-dir results/unified_20250101120000 --auto-detect
```

## Sweep Types

| Type | Purpose | Focus |
|------|---------|-------|
| `unified` | General parameter sweeps | E/I balance + architecture + training |
| `ei` | E/I balance studies | Excitation/inhibition ratios |
| `branch` | Architecture studies | Dendritic tree structures |
| `noise` | Robustness analysis | Noise perturbation effects |
| `general` | Custom studies | Any parameter combinations |

## Analysis Output

**Generated: 65+ comprehensive plots**

### Plot Categories
- **Performance**: Accuracy/loss vs E/I ratio, parameter scaling, heatmaps
- **Robustness**: Noise perturbation analysis (gaussian/uniform)
- **Architecture**: Ablation analysis (layer-wise contributions)
- **Information**: Mutual information analysis (global + layer-specific)

### Key Features
- **Layer-specific information flow**: I(E;C), I(I;E|C) vs E/I ratio by dendritic layer
- **Network comparisons**: Shunting vs non-shunting architectures
- **Statistical analysis**: Mean/std aggregation over multiple seeds
- **Publication-ready**: High-resolution plots with proper mathematical notation

## Configuration

**Template:** `configs/unified_all_sweeps_example.yaml`

```yaml
sweep_config:
  model.core_network.parameters.ee_synapses_per_branch_per_layer: [[1],[2],[3]]
  model.core_network.parameters.ie_synapses_per_branch_per_layer: [[1],[2],[3]]
  model.core_network.parameters.use_shunting: [true, false]

filter_config:
  mode: "ei-sum"  # E + I = value
  value: 4
```

**Result:** Automatic SLURM job submission + comprehensive analysis

## Help
```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --help
```