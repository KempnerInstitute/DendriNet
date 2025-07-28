# Scripts Folder

This folder contains executable scripts for running experiments and tools.

## Structure

- **training/**: Scripts for training models and running experiments
  - `train_experiments.py` - Main training script for experiments
  - `train_experiments_fsdp.py` - FSDP (Fully Sharded Data Parallel) variant for distributed training
  - `train_experiments_fstp.py` - FSTP variant for specialized training
  - `train_encoder_network.py` - Script for training encoder networks
  
- **tools/**: Utility tools for development and analysis
  - `estimate_gpu_requirements.py` - GPU approximation and requirement estimation tool
  - `generate_slurm_job.py` - Script to generate SLURM job files for cluster runs
  - `calculate_ei_parameters.py` - Tool to calculate E/I network parameters and model size
  
- **script_utils/**: Helper utilities specific to scripts
  - `setup_utils.py` - Setup utilities for training environment initialization

## Usage

To run a script from the project root:

```bash
# Training experiments
python src/dendritic_modeling/scripts/training/train_experiments.py [args]

# GPU estimation
python src/dendritic_modeling/scripts/tools/estimate_gpu_requirements.py [args]

# Generate SLURM jobs
python src/dendritic_modeling/scripts/tools/generate_slurm_job.py [args]
```

## Import Structure

Scripts can be imported as modules:

```python
from dendritic_modeling.scripts.training.train_experiments import run_dendritic_modeling_experiments
from dendritic_modeling.scripts.tools.estimate_gpu_requirements import main as estimate_gpu
``` 