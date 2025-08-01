Sweeps Guide
============

Systematic parameter exploration across different configurations.

Sweep Types
-----------

1. EI Sweep: Explore excitatory/inhibitory synapse combinations
2. Branch Sweep: Vary branch architecture parameters  
3. Noise Sweep: Test robustness to different noise types
4. General Sweep: Arbitrary parameter combinations

Usage
-----

.. code-block:: bash

   # Generate and run sweep
   python -m dendritic_modeling.scripts.sweeps.unified.sweep_manager \
     --config configs/examples/general_sweep_example.yaml --run

   # Generate only (no job submission)
   python -m dendritic_modeling.scripts.sweeps.unified.sweep_manager \
     --config configs/examples/general_sweep_example.yaml --generate-only

   # Analyze results
   python -m dendritic_modeling.scripts.sweeps.unified.sweep_manager \
     --analyze --results-dir output/sweep_20240101_120000

Configuration
------------

Basic sweep configuration:

.. code-block:: yaml

   sweep_config:
     train.main.trainer.lr: [0.0001, 0.001, 0.01]
     train.main.trainer.batch_size: [32, 64, 128]
     model.core_network.parameters.excitatory_layer_sizes: [[10], [20], [30]]

   experiment_settings:
     seeds_per_condition: 3
     base_seed: 42

   slurm_config:
     account: "kempner_dev"
     partition: "kempner"
     time: "02:00:00"
     gpus_per_node: 1

Results are saved in timestamped directories with heatmaps and statistical analysis.

See ``configs/examples/`` for example sweep configurations. 