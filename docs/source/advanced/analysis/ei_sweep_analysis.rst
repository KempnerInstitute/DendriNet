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

Separate heatmaps are generated for train, validation, and test sets:

- ``heatmap_train_acc.png``
- ``heatmap_val_acc.png``
- ``heatmap_test_acc.png``

Parameter Sweep Line Plots
^^^^^^^^^^^^^^^^^^^^^^^^

Line plots show the relationship between specific parameters and model performance:

- ``sweep_by_ee.png``: How test accuracy changes with EE values, shown as separate lines for each IE value
- ``sweep_by_ie.png``: How test accuracy changes with IE values, shown as separate lines for each EE value

These plots help identify trends and interactions between parameters.

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

The script generates several output files:

**Data Files**:

- ``ei_sweep_results.csv``: All results in tabular format
- ``missing_runs.csv``: Details about missing or incomplete runs (if any)

**Summary Files**:

- ``summary.txt``: Text summary with the best parameter combination and performance metrics

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

1. Locate experiment directories based on command-line arguments
2. For each experiment:
   - Collect parameter data from config files
   - Collect results from all job directories
   - Generate visualizations and summary files
3. If analyzing multiple experiments:
   - Combine results for comparison
   - Generate comparative visualizations

The implementation handles various edge cases including:
- Experiments with different parameter ranges
- Missing or incomplete runs
- Multiple random seeds for each parameter combination

See the source code in ``scripts/analyze_ei_sweep.py`` for implementation details.
