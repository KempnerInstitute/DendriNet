Parameter Sweeps
================

Systematic exploration of hyperparameter spaces with automated job submission and analysis.

Quick Start
-----------

Run a parameter sweep:

.. code-block:: bash

   # Generate and run sweep (default behavior)
   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/unified_all_sweeps_example.yaml

   # Generate configs only (no job submission)
   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/unified_all_sweeps_example.yaml --generate-only

   # Analyze results with auto-detection
   python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --results-dir output/sweep_20240101120000 --auto-detect

Configuration
-------------

Basic sweep configuration:

.. code-block:: yaml

   sweep_config:
     model.core_network.parameters.ee_synapses_per_branch_per_layer: [[1], [2], [3]]
     model.core_network.parameters.ie_synapses_per_branch_per_layer: [[1], [2], [3]]
     model.core_network.parameters.use_shunting: [true, false]
     train.main.param_groups.lr: [0.001, 0.01, 0.1]

   filter_config:
     mode: "ei-sum"
     value: 4

   experiment_settings:
     seeds_per_condition: 3
     base_seed: 42

   slurm_config:
     account: "kempner_dev"
     partition: "kempner"
     time: "02:00:00"
     gpus_per_node: 1

Sweep Types
-----------

The framework supports unified parameter sweeps with filtering:

- EI balance studies (excitatory/inhibitory ratios)
- Architecture exploration (layer sizes, branch factors)
- Training parameter optimization (learning rates, batch sizes)
- Noise robustness analysis

Analysis Output
---------------

The result analyzer generates comprehensive plots:

- Performance vs parameter heatmaps
- Statistical analysis across seeds
- Architecture comparisons
- Information theory analysis (when enabled)

Results are saved in timestamped directories:

.. code-block:: text

   output/sweep_20240101120000/
   ├── configs/              # Generated configurations
   ├── analysis/             # Analysis plots and data
   ├── submit_all_jobs.sh   # Job submission script
   └── sweep_summary.json   # Sweep metadata

Examples
--------

See example configurations in ``configs/unified_all_sweeps_example.yaml`` and refer to the sweep manager help:

.. code-block:: bash

   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help
   python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --help