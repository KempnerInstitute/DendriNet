.. _ei_sweep_analysis:

EI Sweep Analysis
================

The ``analyze_ei_sweep.py`` script provides comprehensive analysis and visualization tools for Excitatory-Inhibitory (EI) sweep experiments. This tool helps researchers understand how different combinations of excitatory and inhibitory parameters affect model performance.

Overview
--------

After running EI sweep experiments (see :ref:`parameter_sweeps`), you'll have results from many different parameter combinations. The ``analyze_ei_sweep.py`` script helps you:

1. Aggregate results from all jobs (potentially across multiple SLURM array jobs)
2. Generate visualizations to understand parameter impact on performance
3. Identify optimal parameter combinations
4. Track missing or incomplete runs
5. Compare results across multiple experiments

Command-Line Arguments
--------------------

The script supports various command-line arguments to customize the analysis:

.. code-block:: text

    python scripts/analyze_ei_sweep.py [OPTION]...

    Required arguments (one of these must be specified):
      --timestamp TIMESTAMP  Timestamp ID of the experiment (e.g. 20250411144518)
      --folder FOLDER        Full path to the experiment folder
      --all                  Analyze all ei_ folders in the output directory
      --multiple TIMESTAMPS  List of timestamps or folder paths to analyze together

    Optional arguments:
      --output-base DIR      Base output directory (default: ./output)
      --prefix PREFIX        Prefix of experiment folders to look for (default: ei_)
      --show-plots           Display plots in addition to saving them
      --report-missing       Report detail of missing or unfinished runs

Visualizations Explained
----------------------

The script generates several types of visualizations to help you understand parameter relationships:

Heatmaps (Accuracy)
^^^^^^^^^^^^^^^^^^

The heatmaps show how model performance varies with different combinations of EE and IE parameters:

- X-axis: IE parameter values
- Y-axis: EE parameter values
- Color: Accuracy value (from 0.0 to 1.0)
- Missing values: Shown as grey cells with "N/A" annotation

Separate heatmaps are generated for train, validation, and test sets (e.g., ``heatmap_train_acc.png``).

Parameter Sweep Line Plots (Grouped by Network Type)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When your sweep includes different network types (via ``model.network.parameters.use_shunting: [true, false, "MLP"]``), the script automatically generates additional visualizations for comparing them. These plots show the relationship between specific E/I parameters and model performance, with separate lines for each network category (EINet with Shunting, EINet without Shunting, MLP):

- **Network-Grouped Heatmaps**: Separate heatmaps for each network category (e.g., ``*_heatmap_EINet_shunting.png``, ``*_heatmap_MLP.png``) are generated, showing performance against EE and IE values for that specific category.

- **Comparison Line Plots** (Accuracy vs. an E/I parameter, lines for the other E/I parameter, grouped by Network Type):
  - ``*_ie_lines_by_network_per_ee.png``: Compares how different network types respond to varying **IE values**. To maintain clarity, lines are drawn for up to 5 representative **EE values**.
  - ``*_ee_lines_by_network_per_ie.png``: Compares how different network types respond to varying **EE values**. Similarly, lines are drawn for up to 5 representative **IE values**.
  - ``*_ei_ratio_by_network.png``: Compares how different network types respond to varying **E/I ratios**. Each network category will have its own line on this plot.

These visualizations make it easy to compare the performance of dendritic networks (EINet with and without shunting) against standard MLP networks using the same parameter values.

Top Combinations Bar Chart
^^^^^^^^^^^^^^^^^^^^^^^^

The ``top_combinations.png`` chart shows the top-performing parameter combinations:

- X-axis: Parameter combinations (EE and IE values)
- Y-axis: Accuracy
- Bars: Paired bars showing validation and test accuracy
- Sorting: Combinations are sorted by validation accuracy

This visualization helps quickly identify the best parameter settings.

Multi-Experiment Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^

When analyzing multiple experiments with the ``--multiple`` option:

- ``experiment_comparison.png``: Bar chart comparing the best results from each experiment
- ``multi_experiment_summary.txt``: Text summary of the comparison

Output Files
-----------

The script generates several output files in the ``plots`` subdirectory of your experiment folder:

**Data Files**:

- ``ei_sweep_results_original.csv``: Raw collected data before extensive cleaning/pivoting.
- ``ei_sweep_results.csv``: Processed results in tabular format, used for some plots.
- ``ei_sweep_<metric>_clean.csv``: Data subset used for plotting a specific metric after NaN removal for EE/IE.
- ``ei_sweep_<metric>_pivot.csv``: Pivot table for a specific metric (EE vs IE).
- ``missing_runs.csv``: Details about missing or incomplete runs (if any).
- ``network_comparison_summary.txt``: A summary listing all generated network comparison plots.

**Plot Files (Examples)**:

- ``ei_sweep_train_acc_heatmap.png`` (and similar for val/test, and per-network-category)
- ``ei_sweep_test_acc_ie_lines_by_network_per_ee.png``
- ``ei_sweep_test_acc_ee_lines_by_network_per_ie.png``
- ``ei_sweep_test_acc_ei_ratio_by_network.png``
- ``top_combinations.png``

(Note: General 3D surface plots and non-grouped scatter plots are no longer generated by default.)

Example Usage
-----------

**Analyzing a Single Experiment**:

.. code-block:: bash

    # Using timestamp ID
    python scripts/analyze_ei_sweep.py --timestamp 20250411144518

    # Using full path
    python scripts/analyze_ei_sweep.py --folder /path/to/outputs/ei_20250411144518

    # With additional options
    python scripts/analyze_ei_sweep.py --timestamp 20250411144518 --show-plots --report-missing

**Analyzing Multiple Experiments**:

.. code-block:: bash

    # Compare two experiments using timestamp IDs
    python scripts/analyze_ei_sweep.py --multiple 20250411144518 20250412123456

    # Compare using full paths
    python scripts/analyze_ei_sweep.py --multiple /path/to/exp1 /path/to/exp2

**Analyzing All Experiments**:

.. code-block:: bash

    # Analyze all experiments with default prefix
    python scripts/analyze_ei_sweep.py --all

    # Analyze all experiments with custom prefix
    python scripts/analyze_ei_sweep.py --all --prefix custom_ei_

Handling Missing Data
-------------------

The script automatically identifies and reports on missing or incomplete runs:

- **Missing**: No task directory found for the configuration
- **Incomplete**: Task directory exists but no final.json performance file
- **Error**: Exception occurred while processing the results

Use the ``--report-missing`` flag to get detailed information about these runs, which can help you:

1. Identify failed jobs
2. Determine which configurations need to be rerun
3. Debug issues with specific parameter combinations

Tips for Effective Analysis
-------------------------

1. **Start with Overview**: First analyze without ``--show-plots`` to get a quick summary
2. **Check for Missing Data**: Use ``--report-missing`` to identify gaps
3. **Compare Validation vs Test**: Look for discrepancies that might indicate overfitting
4. **Examine Parameter Trends**: Use line plots to understand how parameters interact
5. **Multi-Experiment Analysis**: Use ``--multiple`` to track progress across experiments

Implementation Details
-------------------

The script follows this workflow:

1. Locate experiment directories based on command-line arguments.
2. For each experiment (`process_single_experiment`):
   - Collect parameter data from original config files (`collect_params_data`).
   - Collect results by parsing `output.log` files from `config_X` directories. Crucially, this step also loads the original config for each run to extract `network_type` and `use_shunting` to determine the `network_category` (`collect_results`).
   - Generate visualizations and summary files (`generate_plots`). This includes averaging results for repeated E/I and network configurations (e.g., across different seeds) when creating pivot tables and line plot data points.
3. If analyzing multiple experiments, combine results and generate comparative visualizations (`compare_multiple_experiments`).

The implementation handles various edge cases including experiments with different parameter ranges, missing or incomplete runs, and multiple random seeds for each parameter combination. The primary grouping for comparative plots is by `network_category`.

See the source code in ``scripts/analyze_ei_sweep.py`` for full implementation details.
