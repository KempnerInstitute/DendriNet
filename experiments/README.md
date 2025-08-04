# Experiments Directory

Contains experiment scripts and analysis tools for the dendritic modeling project.

## Organization

- **analysis/** - Multi-run result analysis and pixel-level information analysis
- **dendrinet_simulations/** - DendriNet architecture experiments
- **neuron_modeling/** - Real neuron fitting scripts
- **old_sweep/** - Alternative parameter sweep implementations
- **slurm/** - SLURM job scripts for cluster computing
- **synaptic_correlation/** - Synaptic correlation analysis
- **vision_experiments/** - Vision neuroscience experiments

## Quick Usage

```bash
# Run analysis
python analysis/analyze_multi_run_results.py --results_dir results/

# Run vision experiments
python vision_experiments/gabor_orientation.py

# Submit SLURM jobs
sbatch slurm/cifar10_16gpu_realistic.sbatch
```

## Requirements

Install the main package: `pip install -e .` 