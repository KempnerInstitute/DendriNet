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

Filtering EI Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~

You can generate sweeps with filtered E-I combinations using the filtering options:

.. code-block:: bash

    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode MODE --filter-value VALUE

Available filtering modes:

- ``all``: Generate all E-I combinations (default)
- ``ei-sum``: Only include combinations where E+I equals a constant (E+I=K)
- ``e-const``: Only include combinations with a specific constant E value
- ``i-const``: Only include combinations with a specific constant I value
- ``ei-ratio``: Only include combinations where E/I approximates a constant ratio

Examples:

.. code-block:: bash

    # Generate only combinations where E+I=52
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-sum --filter-value 52

    # Generate only combinations where E=26
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode e-const --filter-value 26

    # Generate only combinations where I=26
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode i-const --filter-value 26

    # Generate only combinations where E/I ratio is approximately 2 (with default tolerance of 0.1)
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-ratio --filter-value 2

    # Generate only combinations where E/I ratio is approximately 2 with a custom tolerance of 0.2
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-ratio --filter-value 2 --filter-tolerance 0.2

This filtering functionality allows you to focus your experiments on specific regions of the parameter space, which is useful for studying particular E-I relationships.

Comparing Network Types
~~~~~~~~~~~~~~~~~~~~~~

You can generate sweeps that compare EINet architecture (with and without shunting) against standard MLP networks by using the special value ``"MLP"`` for the ``use_shunting`` parameter:

.. code-block:: yaml

    sweep_config:
      model.network.parameters.ee_synapses_per_branch_per_layer: [
        [1], [6], [11], [16], [21], [26]
      ]
      model.network.parameters.ie_synapses_per_branch_per_layer: [
        [1], [6], [11], [16], [21], [26]
      ]
      model.network.parameters.use_shunting: [true, false, "MLP"]

This configuration will generate three types of networks for each E-I combination:

- ``use_shunting=true``: EINet with shunting inhibition enabled
- ``use_shunting=false``: EINet with shunting inhibition disabled
- ``use_shunting="MLP"``: Standard MLP network using the same E-I parameters

When using the ``"MLP"`` option, the script automatically:

1. Sets the network type to "MLP" instead of "EINet"
2. Uses the specified E-I parameters to configure the MLP architecture
3. Sets ``use_shunting=false`` for the MLP configuration

After running the sweep, the ``analyze_ei_sweep.py`` script will automatically detect the different network types and generate comparative visualizations grouped by network category (EINet with shunting, EINet without shunting, and MLP). This enables direct comparison between dendritic and non-dendritic architectures using the same parameter values.

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
      model.network.parameters.use_shunting: [true, false, "MLP"]

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
2. Extract metrics (train accuracy, validation accuracy, test accuracy) and network configuration details
3. Generate visualizations showing the relationship between E/I parameters and performance, grouped by network type
4. Identify the optimal parameter configuration
5. Create a ``network_comparison_summary.txt`` file listing key comparison plots

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

The script generates various plots to help you understand the results, primarily focusing on comparing different network configurations (EINet with Shunting, EINet without Shunting, MLP):

1. **Heatmaps** (e.g., ``*_heatmap_EINet_shunting.png``, ``*_heatmap_MLP.png``, ``*_heatmap.png``):
   2D visualizations showing how metrics (train_acc, val_acc, test_acc) vary with EE/IE parameter combinations. Separate heatmaps are generated for each network category and for the overall dataset.

2. **Grouped Line Plots** (Metric vs. E/I parameter, grouped by Network Type):
   * **Metric vs. IE (lines per EE value)**: ``*_ie_lines_by_network_per_ee.png``
       Shows how performance varies with IE, for up to 5 representative EE values. Separate sets of these lines are plotted for each network category.
   * **Metric vs. EE (lines per IE value)**: ``*_ee_lines_by_network_per_ie.png``
       Shows how performance varies with EE, for up to 5 representative IE values. Separate sets for each network category.
   * **Metric vs. E/I Ratio**: ``*_ei_ratio_by_network.png``
       Shows how performance varies with the E/I ratio. One line is plotted for each network category.

3. **Top Combinations Bar Chart** (``top_combinations.png``):
   Shows the top-performing parameter combinations (EE and IE values) based on validation and test accuracy.

(Note: General 3D surface plots and non-grouped scatter plots are no longer generated by default to maintain focus and clarity.)

Output Structure
~~~~~~~~~~~~~~

The analysis results are saved in the ``plots`` directory within your experiment folder (e.g., ``output/ei_<timestamp>/plots/``):

.. code-block:: text

    output/
    └── ei_<timestamp>/
        ├── ...
        └── plots/
            ├── ei_sweep_results_original.csv  (Raw collected data)
            ├── ei_sweep_results.csv           (Processed data for plotting)
            ├── ei_sweep_train_acc_heatmap.png (Overall heatmap)
            ├── ei_sweep_train_acc_heatmap_EINet_shunting.png (Category-specific heatmap)
            ├── ei_sweep_train_acc_heatmap_MLP.png
            ├── ei_sweep_val_acc_ie_lines_by_network_per_ee.png
            ├── ei_sweep_val_acc_ee_lines_by_network_per_ie.png
            ├── ei_sweep_val_acc_ei_ratio_by_network.png
            ├── top_combinations.png
            ├── network_comparison_summary.txt (Summary of comparison plots)
            ├── ... (other specific data files like _clean.csv, _pivot.csv)
            └── summary.txt (Overall best parameters)

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

Rerunning Missing Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If some configurations failed or were not completed in the sweep, you can use the ``rerun_missing_configs.py`` script to generate a new batch job that only runs the missing configurations:

.. code-block:: bash

    python scripts/rerun_missing_configs.py --timestamp 20250412010555
    sbatch rerun_missing.sh

This will:

1. Scan the experiment directory to identify which configurations have results
2. Generate a new SLURM job script that only runs the missing configurations
3. Save the generated script as ``rerun_missing.sh`` (or a custom name with ``--output-file``)

Command-Line Arguments
^^^^^^^^^^^^^^^^^^^^^^

The ``rerun_missing_configs.py`` script supports several command-line arguments:

.. code-block:: bash

    python scripts/rerun_missing_configs.py --timestamp 20250412010555 \
      --config configs/sweep_params_seeds_ei.yaml \
      --max-concurrent 20 \
      --rerun-partial-failures \
      --force-config-ids 8,15,23 \
      --identify-failed-directories

Key arguments include:

* ``--folder``: Full path to the ei_TIMESTAMP folder
* ``--timestamp``: Timestamp ID of the experiment (alternative to --folder)
* ``--config``: Path to the sweep configuration file to get output_dir
* ``--base-dir``: Base directory where ei_TIMESTAMP folders are located (overridden if config file has output_dir)
* ``--output-file``: Path to save the generated SLURM script (default: rerun_missing.sh)
* ``--max-concurrent``: Maximum number of concurrent jobs (default: 40)
* ``--check-log-content``: Check log content to verify task completion rather than just file existence
* ``--rerun-partial-failures``: Rerun configs that have any failed attempts, even if they also have successful runs
* ``--force-config-ids``: Comma-separated list of config IDs to forcibly rerun (e.g., '8,15,23')
* ``--identify-failed-directories``: Show specific job directories with failed tasks

Common Use Cases
^^^^^^^^^^^^^^^

1. **Rerun Completely Failed Configurations**:

   .. code-block:: bash

       python scripts/rerun_missing_configs.py --timestamp 20250412010555

   This runs only configurations that have no successful results.

2. **Rerun Configurations with Any Failed Attempts**:

   .. code-block:: bash

       python scripts/rerun_missing_configs.py --timestamp 20250412010555 --rerun-partial-failures

   This also includes configurations that have some successful runs but also have some failed attempts.

3. **Force Specific Configurations to Rerun**:

   .. code-block:: bash

       python scripts/rerun_missing_configs.py --timestamp 20250412010555 --force-config-ids 8,15,23

   This forcibly reruns specific configurations regardless of their current status.

4. **Identify Failed Task Directories**:

   .. code-block:: bash

       python scripts/rerun_missing_configs.py --timestamp 20250412010555 --identify-failed-directories

   This shows exactly which job/task directories have failed runs.

The script maintains the same directory structure and configuration as the original sweep, ensuring that results are stored consistently. This makes it easy to fill in missing results without rerunning the entire sweep.

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

4. **Check for Missing Results** (Optional):

   .. code-block:: bash

       python scripts/analyze_ei_sweep.py --folder output/ei_<timestamp> --report-missing

   If there are missing results, rerun only those configurations:

   .. code-block:: bash

       python scripts/rerun_missing_configs.py --timestamp <timestamp>
       sbatch rerun_missing.sh

5. **Analyze Results**:

   .. code-block:: bash

       python scripts/analyze_ei_sweep.py --folder output/ei_<timestamp> --show-plots

6. **Identify Optimal Parameters**:

   From the analysis output, identify the best EE and IE values for your task.

7. **Fine-tune with Optimal Values**:

   Use the identified optimal values in your final model configuration.

Conclusion
---------

EI parameter sweeps provide a powerful method for optimizing the excitatory-inhibitory balance in dendritic networks. The specialized scripts in the framework automate the process of generating, running, and analyzing these sweeps, helping you find the optimal parameter configuration for your specific task.
