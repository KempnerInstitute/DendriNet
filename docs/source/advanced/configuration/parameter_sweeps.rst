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

    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml

This script:

- Creates a timestamped directory (e.g., ``ei_20250411145829``)
- Generates individual YAML configuration files for each parameter combination and seed
- Produces a custom ``run.sh`` SLURM script for batch execution

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

Submit the generated SLURM job:

.. code-block:: bash

    sbatch run.sh

The generated ``run.sh`` script:

- Launches a SLURM array job with one task per configuration
- Each task runs the dendritic model with specific EE/IE values and seed
- Results are saved to individual directories within the experiment folder

Once submitted, SLURM will manage the execution of all configurations, respecting the maximum number of concurrent jobs specified in the configuration.

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
4. **Result Tracking**: Use W&B integration for additional tracking and visualization
5. **Incremental Analysis**: You can analyze partial results while jobs are still running

Extending for Custom Sweeps
-------------------------

The EI sweep framework can be extended to explore other parameter combinations:

1. Modify the ``sweep_config`` section in your YAML file to include different parameters
2. Update the analysis script to extract and visualize the relevant parameters

See the source code of ``generate_ei_sweep.py`` and ``analyze_ei_sweep.py`` for implementation details.
