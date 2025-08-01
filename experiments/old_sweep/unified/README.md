# Unified Sweep Framework (Legacy)

Legacy parameter sweep functionality. **Note**: Active development moved to `src/dendritic_modeling/scripts/sweeps/unified/`.

## Quick Usage

```bash
# EI parameter sweep
python sweep_manager.py ei --config configs/ei_sweep.yaml --run

# Branch architecture sweep  
python sweep_manager.py branch --config configs/branch_sweep.yaml --run

# Analyze results
python result_analyzer.py --results-dir output/ei_20240101120000 --auto-detect
```

## Components

- **sweep_manager.py** - Main entry point for all sweep operations
- **result_analyzer.py** - Unified result analysis
- **base_sweep.py** - Base classes and utilities
- **sweep_types/** - Type-specific implementations (EI, branch, noise, general)

## Migration Note

This is a legacy backup. For current sweep functionality, use:
```bash
python src/dendritic_modeling/scripts/sweeps/unified/sweep_manager.py
``` 