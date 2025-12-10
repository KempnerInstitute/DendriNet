# Sweep Analysis Framework

## Overview

Modular architecture for analyzing parameter sweeps in dendritic modeling experiments.

## Architecture

```
src/dendritic_modeling/scripts/sweeps/
├── result_analyzer.py          # Main CLI entry point
├── analyzers/                  # Sweep-type specific analyzers
│   ├── ei_analyzer.py
│   ├── local_learning_analyzer.py
│   ├── noise_analyzer.py
│   ├── ablation_analyzer.py
│   └── general_analyzer.py
├── collectors/                 # Data collection from results
│   ├── data_collector.py
│   └── config_extractor.py
├── plotters/                   # Specialized plotting
│   ├── performance_plotter.py
│   ├── information_plotter.py
│   └── weight_plotter.py
└── utils/                      # Shared utilities
    ├── metrics.py
    ├── aggregation.py
    └── formatting.py
```

## Usage

Basic usage (auto-detect sweep type):
```bash
python result_analyzer.py /path/to/sweep/results
```

Specify sweep type:
```bash
python result_analyzer.py /path/to/sweep --type ei
python result_analyzer.py /path/to/sweep --type local_learning
python result_analyzer.py /path/to/sweep --type noise
python result_analyzer.py /path/to/sweep --type ablation
```

List available types:
```bash
python result_analyzer.py --list-types
```

Python API:
```python
from dendritic_modeling.scripts.sweeps.analyzers import get_analyzer

analyzer = get_analyzer("ei")
results = analyzer.analyze(Path("/path/to/results"))
```

## Sweep Types

**ei** - E/I ratio sweeps
- Groups by: ee_value, ie_value, ei_ratio, use_shunting
- Unique plots: E/I heatmaps, shunting comparison
- Standard plots: Performance, information, weights, ablation, noise

**local_learning** - Local learning parameter sweeps
- Groups by: rule_variant, rho_mode, error_broadcast_mode, morphology params
- Unique plots: Rule variant comparison, parameter combination legend
- Standard plots: Performance, information, weights, ablation, noise

**branch** - Architecture/morphology sweeps
- Groups by: excitatory_layers, inhibitory_layers, branch_factors, nparams
- Unique plots: Performance vs parameter count (nparams scaling)
- Standard plots: Performance, information, weights, ablation, noise

**general** - Any other parameter sweep (fallback)
- Groups by: Auto-detected varying parameters
- Standard plots: Performance, information, weights, ablation, noise

Note: Ablation and noise analyses are automatically included when data is available.

## Directory Structure

Input:
```
sweep_results/
├── configs/config_*.yaml
└── results/config_*/
    ├── performance/final.json
    ├── information_analysis/final
    └── weight_analysis/final
```

Output:
```
plots/
├── {type}_processed_data.csv
├── {type}_aggregated_data.csv
├── performance/
├── information/
├── weights/
├── ablation/
├── noise/
└── {sweep_specific}/
```

## Testing

Run sweep tests:
```bash
python -m unittest tests.test_analyzers -v
python -m unittest tests.test_config_extractor -v
python -m unittest tests.test_aggregation -v
python -m unittest tests.test_data_collector -v
```

## Adding New Sweep Types

Create analyzer in `analyzers/custom_analyzer.py`:
```python
class CustomSweepAnalyzer(BaseSweepAnalyzer):
    def can_handle(self, results_dir: Path) -> bool:
        return True  # Detection logic

    def get_groupby_columns(self) -> List[str]:
        return ["param1", "param2"]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> List[Path]:
        plotter = PerformancePlotter()
        return plotter.generate_all(data, output_dir)
```

Register in `analyzers/__init__.py`:
```python
ANALYZER_REGISTRY["custom"] = CustomSweepAnalyzer
```

## Components

**BaseSweepAnalyzer** - Abstract base class with analysis pipeline

**Analyzers** - Sweep-type specific logic and plotting

**DataCollector** - Collects results from canonical directory structure

**ConfigExtractor** - Extracts parameters from config files

**Plotters** - Generate plots (performance, information, weights)

**Utils** - Metric definitions, aggregation, formatting

See docstrings in each module for detailed API documentation.
