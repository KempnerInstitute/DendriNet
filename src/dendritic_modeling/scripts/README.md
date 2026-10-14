# Scripts

The command-line programs of DendriNet. Installing the package installs them
as console scripts (`[project.scripts]` in `pyproject.toml`);
`docs/cli_reference.rst` lists every command with its options. Reusable logic
belongs in the library modules under `dendritic_modeling.*`, not here.

## Layout

- `training/`: `train_experiments.py` (`dendritic-train`; DDP under
  `torchrun`) and `train_experiments_fsdp.py` (FSDP under `torchrun`).
- `sweeps/`: sweep generation and submission (`dendritic-sweep`), sweep
  result analysis (`dendritic-analyze`), metric collection
  (`dendritic-collect-sweep-metrics`) and paired comparisons; see
  `sweeps/README.md`.
- `analysis/`: checkpoint analysis (`dendritic-analyze-checkpoint`,
  `dendritic-submit-checkpoint-analysis`), the `dendritic-collect-*`
  collectors that gather analyzer outputs across checkpoints, and epoch
  selection (`dendritic-select-validation-epochs`).
- `deployment/`: compact export of trained models
  (`dendritic-compact-checkpoint`).
- `tools/`: Slurm job writers (`generate_slurm_job.py`, `submit_fsdp_job.py`),
  a GPU memory estimate (`estimate_gpu_requirements.py`) and an E/I parameter
  count (`calculate_ei_parameters.py`).
- `benchmarks/`: timing and profiling of configured models and indexed
  kernels.
- `verification/`: comparison of numerical probes between two checkouts.
- `docs/`: generators for documentation pages.
- `script_utils/`: run setup and model construction shared by the programs
  (`setup_utils.setup_environment`, `setup_utils.initialize_model`).

## Common commands

Run from the repository root:

```bash
# train
dendritic-train configs/config_exp.yaml

# distributed training (one process per GPU)
torchrun --nproc_per_node=4 -m dendritic_modeling.scripts.training.train_experiments configs/config_exp.yaml
torchrun --nproc_per_node=4 -m dendritic_modeling.scripts.training.train_experiments_fsdp configs/config_exp_fsdp.yaml

# generate a sweep without submitting it, then summarize finished results
dendritic-sweep --config configs/sweep_config.yaml --generate-only
dendritic-analyze <sweep directory>

# estimate the GPU memory of a configuration
python -m dendritic_modeling.scripts.tools.estimate_gpu_requirements configs/config_exp.yaml
```

`scripts/train_fsdp.sh` at the repository root wraps the FSDP command in a
Slurm job; pass your site's partition and account to `sbatch`. The tool
modules without a console script run with `python -m
dendritic_modeling.scripts.<package>.<module>`.
