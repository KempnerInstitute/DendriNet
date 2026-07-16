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
     model.core.population_network.layers.0.population_defaults.ff_excitatory_synapses: [20, 30]
     model.core.population_network.layers.0.population_defaults.ff_inhibitory_synapses: [10, 15]
     model.core.population_network.layers.0.populations.1.branch_factors:
       - [3, 3]
       - [4, 4]
     training.main.common.param_groups.lr: [0.001, 0.01, 0.1]

   filter_config:
     mode: "ei-sum"
     value: 40

   experiment_settings:
     seeds_per_condition: 3
     base_seed: 42

   slurm_config:
     account: ""
     partition: ""
     time: "02:00:00"
     gpus_per_node: 1

Sweep Types
-----------

Current analyzers cover:

- EI balance studies (excitatory/inhibitory ratios)
- Population-network branch and morphology sweeps
- Local-learning sweeps
- Stochastic-dataset sweeps
- General parameter sweeps

Analysis Output
---------------

The analyzer writes processed CSVs and plot directories under the sweep output
tree. For general sweeps it records swept parameters from ``_sweep_metadata``
and creates parameter-effect plots, heatmaps for pairs of swept parameters, and
correlation plots when performance columns such as ``valid_acc`` or ``test_acc``
are present. Model-level analyses requested in ``base_config.analysis`` still
run inside each training job, so population-network sweeps can also collect
routing, spike-train, synaptic-activation, and other configured analysis
artifacts per condition.

The shared ``DataCollector`` also exports compact convergence fields from
``training_summary.json``: best validation epoch/loss, epochs completed, and
the final train/validation losses. Full epoch histories are deliberately not
copied into tabular sweep outputs. Compare ``training_best_epoch`` with the
declared ``epochs`` horizon when auditing undertraining; do not select a
mechanism or condition from test accuracy.

Typical layout:

.. code-block:: text

   <sweep_dir>/
   |-- configs/
   |-- results/
   |-- plots/
   `-- jobs/

Examples
--------

Use ``configs/sweep_config.yaml`` as the maintained generic sweep template.
Project-specific sweep packs should live outside the main config tree
(for example under ``drafts/*/configs/sweeps/`` or local ignored folders).

Population-network sweep paths use normal dot notation plus numeric list
indices. For example:

.. code-block:: yaml

   sweep_config:
     # First layer population defaults
     model.core.population_network.layers.0.population_defaults.tau_base: [20.0, 50.0]

     # Excitatory/readout population in the first layer
     model.core.population_network.layers.0.populations.1.n_neurons: [64, 128]

     # Fourth connection in the first layer
     model.core.population_network.layers.0.connections.3.probability: [0.3, 0.6]

Use ``filter_config.mode: "ei-sum"`` / ``"ei-ratio"`` / ``"e-const"`` /
``"i-const"`` with the population-network E/I synapse paths:

.. code-block:: yaml

   sweep_config:
     model.core.population_network.layers.0.population_defaults.ff_excitatory_synapses: [20, 30, 40]
     model.core.population_network.layers.0.population_defaults.ff_inhibitory_synapses: [10, 20, 30]
   filter_config:
     mode: "ei-sum"
     value: 50

Branch sweeps update excitatory ``branch_factors`` in population-network
layers. For heterogeneous layers, use explicit dot-path sweeps when you want to
change only one named population.

.. code-block:: bash

   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help
   python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --help
