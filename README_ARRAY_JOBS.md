# Dendritic Network Simulation with Array Jobs

This README explains how to use the array job scripts to run the dendrinet simulation on a computing cluster using SLURM array jobs.

## Overview

The array job system consists of:

1. **Configuration files**: YAML files that define parameter sweeps
2. **Generation script**: Creates individual config files and job scripts for array submission
3. **Simulation script**: Runs a single simulation with parameters from a config file
4. **Analysis script**: Combines and visualizes results from all jobs

## Two-Level Parameter Sweep Approach

To efficiently use computational resources, we use a two-level parameter sweep approach:

1. **Array Job Level**: Parameters like `N_E`, `N_I`, and `rho` create separate array jobs.
   - Each unique combination of these parameters becomes one array job task
   - This level determines the total number of array jobs

2. **Within-Job Level**: Parameters like correlation combinations and group combinations are processed within each job.
   - Each job internally loops over all combinations of:
     - Correlation combinations (`cEE`, `cII`, `cEI`)
     - Group information assignments (`E_group`, `I_group`)
     - Random seeds
   - This approach significantly reduces the number of array jobs while still exploring the full parameter space

For example, with:
- 3 values each for N_E, N_I, and rho (27 combinations)
- 7 correlation combinations
- 9 group combinations
- 3 seeds

The old approach would create 27 × 7 × 9 × 3 = 5,103 array jobs
The new approach creates only 27 array jobs, each processing 7 × 9 × 3 = 189 combinations internally

## Directory Structure

When you run the generation script, it creates a timestamped directory with this structure:

```
dendrinet_20240501123456/
├── configs/                   # Individual configuration files for each job
│   ├── dendrinet_sim_config_0.yaml
│   ├── dendrinet_sim_config_1.yaml
│   └── ...
├── results/                   # Results from each job (created during execution)
│   └── [job_id]/
│       └── task_[task_id]/
│           ├── simulation_results.csv
│           ├── simulation_results.json
│           ├── weights.npz
│           └── ...
├── plots/                     # Analysis results (created by analysis script)
│   ├── parameter_effects/
│   ├── correlation_effects/
│   ├── group_effects/
│   ├── synergy_analysis/
│   └── ...
├── run_dendrinet_sweep.sh     # Main job submission script
└── analyze_dendrinet_results.sh  # Analysis script
```

## Step 1: Create Configuration File

Create a YAML configuration file that defines the parameters to sweep over. Example:

```yaml
# configs/dendrinet_simulation_config.yaml
slurm_config:
  account: kempner_dev
  run_name: dendrinet_sim
  output_dir: /path/to/outputs
  # ... other SLURM settings ...

experiment_settings:
  seeds_per_condition: 3
  base_seed: 42

simulation_params:
  D: 30  # Feature dimensions
  C: 3   # Number of classes
  # ... default simulation parameters ...

sweep_config:
  # Array job level parameters
  N_E: [5, 10, 15]  # Each value creates a separate array job
  N_I: [1, 5, 10]   # Each value creates a separate array job
  rho: [0.1, 0.3, 0.5]  # Each value creates a separate array job

  # Within-job loop parameters (processed inside each job)
  correlation_combos:
    - [0.0, 0.0, 0.0]
    - [0.3, 0.3, 0.3]
    # ... more correlation combinations ...

  group_combos:
    - [0, 0]  # Both E and I from no-info group
    - [2, 2]  # Both E and I from high-info group
    # ... more group combinations ...
```

## Step 2: Generate Jobs

Generate the job scripts and individual configuration files:

```bash
python scripts/generate_dendrinet_sweep.py configs/dendrinet_simulation_config.yaml
```

This will create a timestamped directory with all necessary files and print instructions on how to submit the jobs.

**Options:**
- `--split-jobs N`: Split into N separate job scripts
- `--configs-per-job N`: Limit each job script to N configurations
- `--dryrun`: Show what would be generated without creating files

## Step 3: Submit Jobs

Submit the generated job script to SLURM:

```bash
sbatch dendrinet_20240501123456/run_dendrinet_sweep.sh
```

If multiple job scripts were generated, submit each one:

```bash
sbatch dendrinet_20240501123456/run_dendrinet_sweep_1.sh
sbatch dendrinet_20240501123456/run_dendrinet_sweep_2.sh
# ...
```

## Step 4: Analyze Results

After all jobs have completed, run the analysis script:

```bash
./dendrinet_20240501123456/analyze_dendrinet_results.sh
```

Or run it directly:

```bash
python scripts/analyze_dendrinet_results.py --timestamp 20240501123456
```

**Options:**
- `--folder /path/to/experiment`: Specify experiment folder directly
- `--all`: Analyze all experiment folders
- `--show-plots`: Display plots in addition to saving them
- `--report-missing`: Show detailed information about missing jobs

## Examples

### Basic Parameter Sweep

```bash
# Generate jobs
python scripts/generate_dendrinet_sweep.py configs/dendrinet_simulation_config.yaml

# Submit jobs
sbatch dendrinet_20240501123456/run_dendrinet_sweep.sh

# Analyze results
python scripts/analyze_dendrinet_results.py --timestamp 20240501123456
```

### Large Parameter Sweep Split into Multiple Jobs

```bash
# Generate jobs split into 5 separate scripts
python scripts/generate_dendrinet_sweep.py configs/dendrinet_simulation_config.yaml --split-jobs 5

# Submit all jobs
for i in {1..5}; do
  sbatch dendrinet_20240501123456/run_dendrinet_sweep_${i}.sh
done

# Analyze results
python scripts/analyze_dendrinet_results.py --timestamp 20240501123456
```

## Tips for Efficient Usage

1. **Test with a small subset first**: Use a small parameter sweep to verify everything works before launching large jobs.

2. **Monitor jobs**: Check job status with `squeue -u <username>` and examine logs in the results directory.

3. **Rerun failed jobs**: If some jobs fail, you can manually resubmit them after examining logs and fixing any issues.

4. **Use the dryrun option**: Before submitting, use `--dryrun` to check how many configurations will be generated.

5. **Adjust resources**: Modify the `slurm_config` section in your config file to request appropriate resources for your simulations.

6. **Balance array jobs vs. within-job processing**: For very large parameter spaces, you can adjust which parameters are processed at the array job level versus within each job.

## Troubleshooting

- **Missing results**: Use the `--report-missing` option with the analysis script to identify jobs that didn't complete.

- **Memory issues**: If jobs are failing due to memory limits, increase the `mem` parameter in the `slurm_config` section.

- **Runtime issues**: If jobs time out, increase the `time` parameter in the `slurm_config` section.

- **Invalid correlation parameters**: If jobs fail due to non-PSD covariance matrices, adjust the correlation parameters to ensure the matrix remains positive-semidefinite.

## Adding New Parameters

To add new parameters to sweep over:

1. Add them to the appropriate section of `sweep_config` in your YAML file:
   - Array job level parameters: Create separate array jobs
   - Within-job loop parameters: Processed inside each job
2. Update the `simulation_params` section with default values
3. Make sure the simulation script (`simulate_dendrinet_array.py`) handles these parameters correctly
