Parameter Sweeps
================

Generate sweep configs, submit jobs, and analyze finished sweep results.

Quick Start
-----------

Run a parameter sweep:

.. code-block:: bash

   # Generate and run a sweep
   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/sweep_config.yaml

   # Generate configs only (no job submission)
   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/sweep_config.yaml --generate-only

   # Analyze an existing sweep directory
   python src/dendritic_modeling/scripts/sweeps/result_analyzer.py /path/to/sweep/results

Configuration
-------------

Basic sweep configuration:

.. code-block:: yaml

   sweep_config:
     model.core.connectivity.ee_synapses_per_branch_per_layer: [[1], [2], [3]]
     model.core.connectivity.ie_synapses_per_branch_per_layer: [[1], [2], [3]]
     model.core.morphology.use_shunting: [true, false]
     training.main.common.param_groups.lr: [0.001, 0.01, 0.1]

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

Current analyzers cover:

- EI balance studies (excitatory/inhibitory ratios)
- Branch and morphology sweeps
- Local-learning sweeps
- Stochastic-dataset sweeps
- General parameter sweeps

Analysis Output
---------------

The analyzer writes processed CSVs and plot directories under the sweep output tree.

Typical layout:

.. code-block:: text

   <sweep_dir>/
   |-- configs/
   |-- results/
   |-- plots/
   `-- jobs/

Examples
--------

See example configurations in ``configs/sweep_config.yaml`` and the
specialized examples under ``configs/sweeps/``.

.. code-block:: bash

   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help
   python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --help
