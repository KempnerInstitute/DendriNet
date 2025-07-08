# Scripts Directory

This directory contains various scripts for experiments, analysis, and utilities related to the dendritic modeling project.

## Organization

### vision_experiments/
Scripts for visual neuroscience experiments:
- `gabor_orientation.py` - Gabor filter orientation selectivity experiments
- `motion_DS.py` - Motion direction selectivity experiments
- `cnn_equiv_edge.py` - CNN edge detection equivalence demonstrations

### neuron_modeling/
Scripts for fitting and modeling real neurons:
- `fit_real_neuron.py` - Fit dendritic models to real neuron data

### documentation_tools/
Tools for generating documentation assets:
- `generate_documentation_figures.py` - Generate all figures for the documentation

### sweeps/
Parameter sweep experiments:
- `ei_sweeps/` - Excitation/Inhibition ratio sweeps
- `experiment_sweeps/` - General experiment parameter sweeps
- `general_sweeps/` - Other parameter sweep utilities

### analysis/
Data analysis scripts:
- `analyze_multi_run_results.py` - Analyze results from multiple experimental runs
- `analyze_pixel_information.py` - Pixel-level information analysis
- `run_pixel_analysis.sh` - Shell script to run pixel analysis

### dendritic_cnn/
CNN implementations using dendritic layers (standalone experiments)

### dendrinet_simulations/
DendriNet architecture simulations and experiments

### synaptic_correlation/
Synaptic correlation analysis experiments

## Usage Examples

### Running Vision Experiments
```bash
python vision_experiments/gabor_orientation.py
python vision_experiments/motion_DS.py
```

### Running Parameter Sweeps
```bash
python sweeps/ei_sweeps/run_sweep.py --config configs/ei_sweep.yaml
```

### Analyzing Results
```bash
python analysis/analyze_multi_run_results.py --results_dir results/
```

### Generating Documentation Figures
```bash
python documentation_tools/generate_documentation_figures.py
```

## Dependencies

Most scripts require the dendritic-modeling package to be installed:
```bash
pip install -e .
```

Additional requirements may vary by script - check individual script headers for specific dependencies. 