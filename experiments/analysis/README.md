# Analysis Tools

Analysis scripts for dendritic modeling experiments.

## Scripts

- **`analyze_pixel_information.py`** - Pixel-level information analysis using mutual information
- **`analyze_multi_run_results.py`** - Multi-run result analysis and comparison  
- **`run_pixel_analysis.sh`** - Shell wrapper for pixel analysis

## Quick Usage

```bash
# Basic pixel analysis
python analyze_pixel_information.py --dataset mnist --config configs/analysis.yaml

# Multi-run analysis
python analyze_multi_run_results.py --results_dir results/

# Shell script wrapper
bash run_pixel_analysis.sh mnist
```

## Requirements

```bash
pip install -e .
pip install dit broja  # For information theory analysis
``` 