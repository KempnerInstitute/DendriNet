.. _ei_sweep_guide:

Running and Analyzing EI Sweeps
===============================

Overview
--------

The dendritic modeling framework includes specialized tools for running sweeps over excitatory (E) and inhibitory (I) parameters to optimize neural network performance. This guide explains how to generate, run, and analyze EI parameter sweeps using the dedicated scripts.

EI sweeps allow you to systematically explore how different configurations of excitatory and inhibitory synapse values affect model performance, providing insights into the optimal balance of excitation and inhibition for a given task.

.. important::
   The workflow described in this guide is the standard approach for parameter sweeps in this codebase. The proper workflow uses the scripts described below:

   1. ``generate_ei_sweep.py`` with ``sweep_params_seeds_ei.yaml`` to generate configurations and run scripts
   2. Execute the generated ``run_ei_sweep.sh`` to run all experiments
   3. Use ``analyze_ei_sweep.py`` to analyze results and generate visualizations

Generating EI Sweeps
--------------------

The ``generate_ei_sweep.py`` script uses the configuration from ``sweep_params_seeds_ei.yaml`` to create a set of configuration files and job scripts for running parameter sweeps over excitatory and inhibitory synapse counts.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml

This will:

1. Use the sweep configuration in ``sweep_params_seeds_ei.yaml``
2. Generate a set of configurations with varying E/I parameters and seeds
3. Create a directory structure for the sweep experiment
4. Generate a Slurm batch script (``run_ei_sweep.sh``) for running the sweep

Configuration Settings
~~~~~~~~~~~~~~~~~~~~~~

The ``sweep_params_seeds_ei.yaml`` file contains all the settings needed for the EI sweep:

.. code-block:: yaml

    # sweep_params_seeds_ei.yaml
    slurm_config:
      account: your_account
      run_name: dendritic_ei_sweep
      output_dir:
      create_subdir_with_timestamp: true
      subdir_prefix: "ei_"
      run_subfolders:
        - configs
        - results
        - plots
      # SLURM settings
      nodes: 1
      ntasks_per_node: 1
      gpus_per_node: 1
      cpus_per_task: 10
      time: 0:30:00
      mem: 400GB
      partition: gpu
      max_concurrent_jobs: 8
      # Other settings
      training_script: src/dendritic_modeling/train_experiments.py
      config_filename_init: ei_sweep_config

    experiment_settings:
      # Use multiple seeds per parameter combination
      seeds_per_condition: 5
      base_seed: 42

    train_config:
      # Base model configuration here
      # ...

    sweep_config:
      # EE and IE synapse parameters to sweep
      model.network.parameters.ee_synapses_per_branch_per_layer: [
        [1], [6], [11], [16], [21], [26], [31], [36], [41], [46], [51]
      ]
      model.network.parameters.ie_synapses_per_branch_per_layer: [
        [1], [6], [11], [16], [21], [26], [31], [36], [41], [46], [51]
      ]

Key sections in this configuration file:

1. **slurm_config**: Settings for SLURM job submission
2. **experiment_settings**: Controls seeds and experiment structure
3. **train_config**: Base model configuration used for all runs
4. **sweep_config**: Parameters to sweep over (EE and IE synapse values)

Output Structure
~~~~~~~~~~~~~~

The script generates a directory structure organized as follows:

.. code-block:: text

    output/
    └── ei_<timestamp>/
        ├── configs/
        │   ├── ei_sweep_config_0.yaml
        │   ├── ei_sweep_config_1.yaml
        │   └── ...
        ├── jobs/
        │   ├── run_ei_sweep_0.sh
        │   ├── run_ei_sweep_1.sh
        │   └── ...
        ├── results/
        │   └── (empty, will contain experimental results)
        └── run_ei_sweep.sh (master script to submit all jobs)

Running the Sweep
----------------

After generating the sweep configuration, run it using the master script:

.. code-block:: bash

    cd /path/to/output/ei_<timestamp>
    ./run_ei_sweep.sh

This submits all jobs to the SLURM scheduler. Each job:

1. Loads the corresponding configuration file
2. Runs the training experiment with specific E/I parameter values
3. Saves results in the results directory with appropriate subdirectories

Analyzing Results
----------------

Once your sweep experiments are complete, analyze the results using ``analyze_ei_sweep.py``.

Basic Usage
~~~~~~~~~~

.. code-block:: bash

    python scripts/analyze_ei_sweep.py --folder /path/to/output/ei_<timestamp>

This will:

1. Scan the experiment directory for results
2. Extract metrics (train accuracy, validation accuracy, test accuracy)
3. Generate visualizations showing the relationship between E/I parameters and performance
4. Identify the optimal parameter configuration

Command-Line Arguments
~~~~~~~~~~~~~~~~~~~~

The ``analyze_ei_sweep.py`` script supports several command-line arguments:

.. code-block:: bash

    python scripts/analyze_ei_sweep.py --timestamp 20250412010555 \
      --show-plots \
      --report-missing

Key arguments include:

* ``--folder``: Path to the experiment folder
* ``--timestamp``: Timestamp ID of the experiment (alternative to --folder)
* ``--show-plots``: Display plots interactively in addition to saving them
* ``--report-missing``: Report details of missing or unfinished runs
* ``--all``: Analyze all ei_ folders in the output directory
* ``--multiple``: List of timestamps or folders to analyze and compare together

Visualization Types
~~~~~~~~~~~~~~~~~

The script generates various plots to help you understand the results:

1. **Heatmaps** (``*_heatmap.png``): 2D visualization showing how metrics (train_acc, val_acc, test_acc) vary with EE/IE parameter combinations. Colors represent performance values with annotations.

2. **3D Surface plots** (``*_surface.png``): 3D visualization showing the performance landscape across the parameter space, providing a clearer view of performance trends and optimal regions.

3. **Line plots**:
   - ``*_ie_lines.png``: Shows effect of IE values on performance for different fixed EE values
   - ``*_ee_lines.png``: Shows effect of EE values on performance for different fixed IE values

4. **Scatter plots** (``*_scatter.png``): Points colored by performance metrics, useful for identifying patterns across the parameter space.

5. **Comparison plots** (when analyzing multiple experiments):
   - Bar charts comparing experiments
   - Line plots showing performance across experiments for specific EE/IE combinations

Output Structure
~~~~~~~~~~~~~~

The analysis results are saved in the plots directory:

.. code-block:: text

    output/
    └── ei_<timestamp>/
        ├── ...
        └── plots/
            ├── ei_sweep_results.csv
            ├── ei_sweep_train_acc_heatmap.png
            ├── ei_sweep_val_acc_heatmap.png
            ├── ei_sweep_test_acc_heatmap.png
            ├── ei_sweep_train_acc_surface.png
            ├── ei_sweep_val_acc_surface.png
            ├── ei_sweep_test_acc_surface.png
            ├── ei_sweep_test_acc_ie_lines.png
            ├── ei_sweep_test_acc_ee_lines.png
            ├── ...
            └── summary.txt

Comparing Multiple Sweeps
------------------------

To compare results from multiple EI sweeps:

.. code-block:: bash

    python scripts/analyze_ei_sweep.py --multiple \
      /path/to/output/ei_20250412010555 \
      /path/to/output/ei_20250413153045

This will:

1. Process each experiment individually
2. Create comparison visualizations
3. Generate a summary of the best parameters across experiments

The comparison outputs are saved in a new directory named ``ei_comparison_<timestamp>``.

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Missing Results**:

   If the analysis shows missing results, check:
   - SLURM job logs for errors (``slurm-*.out`` files)
   - Task output logs in the results directories
   - Whether jobs are still running with ``squeue``

2. **No Metrics Extracted**:

   If the script doesn't extract metrics:
   - Check log file format in your experiment
   - Ensure metrics are reported with standard naming (e.g., "accuracy", "train_acc")
   - Use the ``--report-missing`` flag to see detailed information

3. **Poor Visualizations**:

   If heatmaps or plots look incomplete:
   - Ensure sufficient parameter combinations completed successfully
   - Check if there's enough variation in the results
   - Try using a subset of the data with the most complete results

Example Workflow
--------------

Here's a complete workflow for running and analyzing an EI sweep:

1. **Prepare Configuration File**:

   Edit ``configs/sweep_params_seeds_ei.yaml`` to set your desired EE/IE values, seeds, and model configuration.

2. **Generate Sweep**:

   .. code-block:: bash

       python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml

3. **Run Sweep**:

   .. code-block:: bash

       cd output/ei_<timestamp>
       ./run_ei_sweep.sh

4. **Analyze Results**:

   .. code-block:: bash

       python scripts/analyze_ei_sweep.py --folder output/ei_<timestamp> --show-plots

5. **Identify Optimal Parameters**:

   From the analysis output, identify the best EE and IE values for your task.

6. **Fine-tune with Optimal Values**:

   Use the identified optimal values in your final model configuration.

Conclusion
---------

EI parameter sweeps provide a powerful method for optimizing the excitatory-inhibitory balance in dendritic networks. The specialized scripts in the framework automate the process of generating, running, and analyzing these sweeps, helping you find the optimal parameter configuration for your specific task.
