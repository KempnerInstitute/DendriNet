.. _parameter_sweeps:

Parameter Sweeps
===============

The Dendritic Modeling framework provides powerful tools for conducting systematic parameter sweeps. This guide focuses on the EI sweep functionality, which allows you to explore how different combinations of excitatory-excitatory (EE) and inhibitory-excitatory (IE) synapse counts affect model performance.

EI Sweep Workflow
----------------

The EI sweep workflow consists of three main steps:

1. Generate sweep configurations with different EE and IE values
2. Run the sweeps using SLURM
3. Analyze results and generate plots

Step 1: Generate Sweep Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``generate_ei_sweep.py`` to create all required configuration files:

.. code-block:: bash

    # Basic usage - generates one big array job
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml

    # Split into 10 separate job scripts
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --split-jobs 10

    # Limit to 50 configurations per job
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --configs-per-job 50

This script:

- Creates a timestamped directory (e.g., ``ei_20250411145829``)
- Generates individual YAML configuration files for each parameter combination and seed
- Produces SLURM job scripts for batch execution
- Optionally splits the sweep into multiple smaller job scripts

The ``--split-jobs`` option divides the parameter sweep into the specified number of job scripts, each handling a subset of configurations. The ``--configs-per-job`` option limits each job script to at most the specified number of configurations.

The ``sweep_params_seeds_ei.yaml`` file defines:

- SLURM configuration (account, resources, partition)
- Parameter ranges for EE and IE synapses
- Number of seeds per parameter combination
- Base network architecture and training parameters

Example Configuration
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # sweep_params_seeds_ei.yaml
    slurm_config:
      account: kempner_dev
      run_name: dendritic_ei_sweep
      output_dir: /path/to/outputs
      nodes: 1
      gpus_per_node: 1
      time: 3:00:00
      # ... other SLURM parameters

    experiment_settings:
      seeds_per_condition: 5  # Use 5 seeds per parameter combination
      base_seed: 42

    train_config:
      # Base configuration for the model
      model:
        network:
          type: "EINet"
          parameters:
            # ... base architecture parameters

    sweep_config:
      # Parameters to sweep - will generate all combinations
      model.network.parameters.ee_synapses_per_branch_per_layer: [
        [1], [6], [11], [16], [21], [26], [31], [36], [41], [46], [51]
      ]
      model.network.parameters.ie_synapses_per_branch_per_layer: [
        [1], [6], [11], [16], [21], [26], [31], [36], [41], [46], [51]
      ]

Directory Structure
^^^^^^^^^^^^^^^^^

The ``generate_ei_sweep.py`` script creates the following directory structure:

.. code-block:: text

    outputs/ei_[timestamp]/
    ├── configs/           # Individual configuration files
    ├── results/           # Will contain results from each job
    └── plots/             # Will contain analysis plots

Step 2: Run the Sweeps
~~~~~~~~~~~~~~~~~~~~~

Run the sweep using the generated job scripts:

.. code-block:: bash

    # If using a single job script (default behavior)
    sbatch run.sh

    # If using multiple job scripts
    sbatch run_ei_sweep_0.sh
    sbatch run_ei_sweep_1.sh
    # etc.

    # Or use the generated helper script (edit first to uncomment commands)
    ./run_all_jobs.sh

When splitting a sweep into multiple jobs, a helper script ``run_all_jobs.sh`` is generated to make it easier to submit all jobs. The script contains commented-out commands that you can uncomment to run jobs all at once or one after another.

The generated job scripts:

- Launch SLURM array jobs with their respective subsets of configurations
- Each task runs the dendritic model with specific EE/IE values and seed
- Results are saved to individual directories within the experiment folder

Job Splitting Benefits
^^^^^^^^^^^^^^^^^^^^

Splitting a large sweep into multiple smaller job scripts offers several advantages:

1. **Better Queue Management**: Smaller jobs may get scheduled faster
2. **Improved Resource Utilization**: More efficient use of cluster resources
3. **Reliability**: If one job fails, others can still complete
4. **Flexibility**: Run different parts of the sweep on different partitions or with different resource requirements
5. **Time Management**: Easier to fit within walltime limits

Step 3: Analyze Results
~~~~~~~~~~~~~~~~~~~~~

After jobs complete, analyze results with:

.. code-block:: bash

    # Analyze a specific experiment
    python scripts/analyze_ei_sweep.py --timestamp 20250411145829

    # Analyze all experiments with 'ei_' prefix
    python scripts/analyze_ei_sweep.py --all

    # Compare multiple experiments
    python scripts/analyze_ei_sweep.py --multiple 20250411145829 20250412123456

The analysis script supports several options:

.. code-block:: text

    --timestamp TIMESTAMP  Timestamp ID of the experiment (e.g. 20250411144518)
    --folder FOLDER        Full path to the experiment folder
    --all                  Analyze all ei_ folders in the output directory
    --multiple IDs         List of timestamp IDs to analyze and compare
    --output_base DIR      Base directory for outputs
    --prefix PREFIX        Prefix used for the output folder (default: ei_)
    --save_plots           Save plots to the plots directory
    --show_plots           Display plots (not just save them)
    --summary_dir DIR      Directory to save comparison plots

The analysis script automatically combines results from all jobs, regardless of whether they were run as a single job or multiple smaller jobs.

Generated Plots and Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

The analysis script generates various plots to help understand the impact of EE and IE synapse counts:

1. **Heatmap of Test Accuracy**: Shows how test accuracy varies with different EE/IE combinations

   .. code-block:: text

       ee_ie_heatmap.png

2. **Line Plots**: Show trends in accuracy for different parameter values

   .. code-block:: text

       ee_vs_accuracy.png  # Accuracy vs EE values for different IE values
       ie_vs_accuracy.png  # Accuracy vs IE values for different EE values

3. **Bar Chart of Top Combinations**: Compares the best-performing parameter combinations

   .. code-block:: text

       top_combinations.png

4. **Summary Files**:

   .. code-block:: text

       ei_sweep_results.csv  # All results in tabular format
       summary.txt  # Text summary with the best parameter combination

When comparing multiple experiments, additional plots are generated:

1. **Box Plot Comparison**: Shows distribution of test accuracy across experiments
2. **Best Parameter Comparison**: Bar chart of the best parameter combination for each experiment
3. **Parameter Combination Line Plot**: Shows how top combinations perform across experiments

Output Structure
--------------

After running the complete workflow, the output directory will have this structure:

.. code-block:: text

    outputs/ei_[timestamp]/
    ├── configs/           # Configuration files for each parameter combination
    ├── results/           # Results from each SLURM job
    │   └── [job_id]/
    │       └── task_[task_id]/
    │           ├── output.log
    │           ├── error.log
    │           └── performance/
    │               └── final.json
    └── plots/             # Generated analysis plots
        ├── ee_ie_heatmap.png
        ├── ee_vs_accuracy.png
        ├── ie_vs_accuracy.png
        └── top_combinations.png

When comparing multiple experiments, a separate summary directory is created with comparison plots.

Key Parameters
------------

In ``sweep_params_seeds_ei.yaml``:

- ``model.network.parameters.ee_synapses_per_branch_per_layer``: Excitatory-excitatory synapses
- ``model.network.parameters.ie_synapses_per_branch_per_layer``: Inhibitory-excitatory synapses
- ``seeds_per_condition``: Number of random seeds to use for each parameter combination
- ``slurm_config.max_concurrent_jobs``: Maximum number of jobs to run simultaneously

Tips for Effective Sweeps
-----------------------

1. **Parameter Range**: Start with a wide range and then refine with narrower sweeps
2. **Multiple Seeds**: Always use multiple seeds (3-5) to ensure robust results
3. **Resource Management**: Set appropriate time limits and memory requirements
4. **Job Splitting**: For very large sweeps (>100 configurations), use ``--split-jobs`` or ``--configs-per-job``
5. **Sequential Submission**: Submit job scripts sequentially to better manage cluster resources
6. **Result Tracking**: Use W&B integration for additional tracking and visualization
7. **Incremental Analysis**: You can analyze partial results while jobs are still running

Extending for Custom Sweeps
-------------------------

The EI sweep framework can be extended to explore other parameter combinations:

1. Modify the ``sweep_config`` section in your YAML file to include different parameters
2. Update the analysis script to extract and visualize the relevant parameters

See the source code of ``generate_ei_sweep.py`` and ``analyze_ei_sweep.py`` for implementation details.
