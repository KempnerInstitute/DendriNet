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

    # Filter combinations where E+I=52
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-sum --filter-value 52

This script:

- Creates a timestamped directory (e.g., ``ei_20250411145829``)
- Generates individual YAML configuration files for each parameter combination and seed
- Creates job scripts for batch execution and stores them in a ``jobs`` directory
- Produces a main ``run.sh`` script in the workspace root for easy submission
- Optionally splits the sweep into multiple smaller job scripts
- Optionally filters E-I combinations based on specified constraints

The ``--split-jobs`` option divides the parameter sweep into the specified number of job scripts, each handling a subset of configurations. The ``--configs-per-job`` option limits each job script to at most the specified number of configurations.

Filtering Options
^^^^^^^^^^^^^^^^^

The script now supports advanced filtering options to focus on specific E-I relationships:

.. code-block:: bash

    # Generate only combinations where E+I=52
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-sum --filter-value 52

    # Generate only combinations where E=26
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode e-const --filter-value 26

    # Generate only combinations where I=26
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode i-const --filter-value 26

    # Generate only combinations where E/I ratio is approximately 2
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-ratio --filter-value 2

    # Customize the tolerance for ratio filtering
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-ratio --filter-value 2 --filter-tolerance 0.2

Available filtering modes:

- ``all``: Generate all E-I combinations (default)
- ``ei-sum``: Only include combinations where E+I equals a constant (E+I=K)
- ``e-const``: Only include combinations with a specific constant E value
- ``i-const``: Only include combinations with a specific constant I value
- ``ei-ratio``: Only include combinations where E/I approximates a constant ratio (within tolerance)

This filtering functionality allows you to focus your computational resources on the most interesting regions of the parameter space or study specific E-I relationships in depth.

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
      model.network.parameters.use_shunting: [true, false, "MLP"]  # "MLP" will create MLP networks

Note that the special value ``"MLP"`` in the ``use_shunting`` parameter will create configurations with network type set to MLP instead of EINet, which allows you to compare EINet models (with shunting enabled or disabled) against standard MLP models in the same sweep. The MLP networks will use the values specified in ``ee_synapses_per_branch_per_layer`` and ``ie_synapses_per_branch_per_layer`` from the sweep to define their structure and parameter counts, ensuring fair comparisons between the architectures.

Directory Structure
^^^^^^^^^^^^^^^^^

The ``generate_ei_sweep.py`` script creates the following directory structure:

.. code-block:: text

    outputs/ei_[timestamp]/
    ├── configs/           # Individual configuration files
    ├── jobs/              # SLURM job scripts
    ├── results/           # Will contain results from each job
    └── plots/             # Will contain analysis plots

Step 2: Run the Sweeps
~~~~~~~~~~~~~~~~~~~~~

Run the sweep using the generated main script:

.. code-block:: bash

    # Run all jobs with automatic sequencing
    ./run.sh

The main ``run.sh`` script:

- Finds all job scripts in the output directory's ``jobs`` folder
- Submits them sequentially with a small delay between submissions
- Provides status messages and job tracking information

The individual job scripts in the output directory's ``jobs`` folder:

- Launch SLURM array jobs with their respective subsets of configurations
- Each task runs the dendritic model with specific EE/IE values and seed
- Results are saved to individual directories within the experiment folder

Job Organization Benefits
^^^^^^^^^^^^^^^^^^^^

The new job organization provides several advantages:

1. **Clean Workspace**: All job scripts are stored in the output directory, keeping your workspace clean
2. **Automatic Submission**: The main script handles submitting all jobs in the correct order
3. **Everything in One Place**: All files related to a sweep (configs, jobs, results) are stored together
4. **Simplified Workflow**: A single command starts the entire sweep process

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
    --output-base DIR      Base directory for outputs (default: ./output)
    --prefix PREFIX        Prefix used for the output folder (default: ei_)
    --show-plots           Display plots in addition to saving them
    --report-missing       Report detail of missing or unfinished runs

The analysis script automatically combines results from all jobs, regardless of whether they were run as a single job or multiple smaller jobs. It also handles missing or incomplete runs gracefully, showing these as grey areas in heatmaps with "N/A" annotations.

Generated Plots and Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

The analysis script generates various plots to help understand the impact of EE and IE synapse counts:

1. **Heatmap of Accuracy**: Shows how train, validation, and test accuracy varies with different EE/IE combinations

   .. code-block:: text

       heatmap_train_acc.png
       heatmap_val_acc.png
       heatmap_test_acc.png

2. **Line Plots**: Show trends in accuracy for different parameter values

   .. code-block:: text

       sweep_by_ee.png  # Test accuracy vs EE values for different IE values
       sweep_by_ie.png  # Test accuracy vs IE values for different EE values

3. **Bar Chart of Top Combinations**: Compares the best-performing parameter combinations

   .. code-block:: text

       top_combinations.png

4. **Summary Files**:

   .. code-block:: text

       ei_sweep_results.csv  # All results in tabular format
       summary.txt  # Text summary with the best parameter combination
       missing_runs.csv  # (if missing runs exist) Details about missing or incomplete runs

When comparing multiple experiments with the ``--multiple`` option, additional files are generated in a comparison directory (named ``ei_comparison_[timestamp]``):

.. code-block:: text

    experiment_comparison.png  # Bar chart comparing best results across experiments
    multi_experiment_summary.txt  # Text summary of the experiment comparison

The ``--report-missing`` flag provides detailed information about any missing or incomplete runs, which can be helpful for debugging or planning additional runs to complete the parameter sweep.

Output Structure
--------------

After running the complete workflow, the output directory will have this structure:

.. code-block:: text

    outputs/ei_[timestamp]/
    ├── configs/           # Configuration files for each parameter combination
    ├── jobs/              # SLURM job scripts
    ├── results/           # Results from each SLURM job
    │   └── [job_id]/
    │       └── task_[task_id]/
    │           ├── output.log
    │           ├── error.log
    │           └── performance/
    │               └── final.json
    └── plots/             # Generated analysis plots
        ├── heatmap_train_acc.png
        ├── heatmap_val_acc.png
        ├── heatmap_test_acc.png
        ├── sweep_by_ee.png
        ├── sweep_by_ie.png
        └── top_combinations.png

When comparing multiple experiments, a separate summary directory is created with comparison plots.

Key Parameters
------------

In ``sweep_params_seeds_ei.yaml``:

- ``model.network.parameters.ee_synapses_per_branch_per_layer``: Excitatory-excitatory synapses
- ``model.network.parameters.ie_synapses_per_branch_per_layer``: Inhibitory-excitatory synapses
- ``model.network.parameters.use_shunting``: Network type (true for EINet, false for standard MLP, "MLP" for MLP networks)
- ``seeds_per_condition``: Number of random seeds to use for each parameter combination
- ``slurm_config.max_concurrent_jobs``: Maximum number of jobs to run simultaneously

Tips for Effective Sweeps
-----------------------

1. **Parameter Range**: Start with a wide range and then refine with narrower sweeps
2. **Multiple Seeds**: Always use multiple seeds (3-5) to ensure robust results
3. **Resource Management**: Set appropriate time limits and memory requirements
4. **Job Splitting**: For very large sweeps (>100 configurations), use ``--split-jobs`` or ``--configs-per-job``
5. **Organized Outputs**: All job scripts and configs are organized by timestamp to avoid confusion between sweeps
6. **Result Tracking**: Use W&B integration for additional tracking and visualization
7. **Incremental Analysis**: You can analyze partial results while jobs are still running

Extending for Custom Sweeps
-------------------------

The EI sweep framework can be extended to explore other parameter combinations:

1. Modify the ``sweep_config`` section in your YAML file to include different parameters
2. Update the analysis script to extract and visualize the relevant parameters

See the source code of ``generate_ei_sweep.py`` and ``analyze_ei_sweep.py`` for implementation details.
