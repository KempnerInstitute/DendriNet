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
     launcher: "python"

Each array element normally launches one Python training process. For a
multi-GPU DDP condition, set ``gpus_per_node`` to the number of local GPUs,
``launcher: "torchrun"``, and ``base_config.distributed.mode: ddp``. The sweep
manager then launches one ``torch.distributed.run`` process group per array
element; it does not treat ranks as independent sweep replicates.

Use ``sweep_variants`` when a comparison requires correlated changes to
several config paths. Each named variant is one condition and its overrides are
applied together. Ordinary ``sweep_config`` axes are then crossed with every
variant and may refine settings shared across variants::

   sweep_variants:
     - name: indexed_dendritic
       overrides:
         model.core.type: einet
         model.core.sparsity.type: indexed
     - name: matched_dense_mlp
       overrides:
         model.core.type: point_mlp
         model.core.sparsity.type: indexed

The generated YAML stores ``_sweep_variant`` and the standard collector exports
it as the ``sweep_variant`` column. Variant names must be unique; override keys
use the same dot-path syntax as ``sweep_config``.

When the experimental seed must also select a fixed topology, declare those
paths once instead of creating a second seed axis::

   experiment_settings:
     seeds_per_condition: 5
     base_seed: 41
     seed_paths:
       - model.core.sparsity.indexed.seed
       - model.core.connectivity.structured.seed

For every replicate, the generator writes the same seed to
``experiment.seed`` and all ``seed_paths``. This is appropriate when topology
is part of the experimental unit; omit ``seed_paths`` to hold connectivity
fixed while varying initialization and data order.

Project-specific launch profiles can extend a maintained benchmark instead of
copying it. ``extends`` is resolved relative to the child YAML, mappings are
deep-merged, and child lists replace parent lists::

   extends: ../../configs/vision/alexnet_fc6_replacement_benchmark.yaml
   slurm_config:
     account: my_account
     partition: my_partition
   base_config:
     data:
       base_dir: /datasets/imagenet
   execution_results_dir: /project/results/fc6_benchmark

The frozen sweep manifest hashes both files and the fully merged
``original_config.yaml``. Cyclic or empty ``extends`` declarations fail before
any job script is created.

By default, generated YAMLs, SLURM scripts, logs, checkpoints, and metrics use
one timestamped sweep directory. Set top-level ``execution_results_dir`` to
keep the lightweight sweep control plane in a code or project workspace while
placing each condition's checkpoints and metrics under a separate large-data
root. The generated configs receive
``<execution_results_dir>/config_<index>``; generation does not create the
external root, so it may be mounted only on compute nodes. The staged
``results`` entry is a symbolic link to that root, preserving the standard
sweep layout expected by ``dendritic-collect-sweep-metrics``.

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

The trainer's exact ``model_resources.json`` accounting
separates full-model and replacement-core parameter counts
(``total_parameters``/``core_total_parameters`` and their trainable
counterparts) from candidate and active synaptic counts. Use the core fields
for replacement-layer size comparisons and the synaptic fields for sparse
connectivity comparisons; they answer different resource questions.

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
