Quick Start
===========

Train an E-I network on MNIST
------------------------------

.. code-block:: bash

   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

This runs a single experiment using the default configuration. Results are
saved to the run output directory configured under ``outputs``.

Override parameters from the command line
-----------------------------------------

.. code-block:: bash

   python src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/config_exp.yaml --seed 42 --output_dir results/my_run --run_name mnist_seed42

Distributed training (DDP / FSDP)
----------------------------------

Set ``distributed.mode`` in your YAML config and use the unified launcher:

.. code-block:: bash

   # Auto-selects entrypoint based on distributed.mode in YAML
   sbatch scripts/train.sh configs/my_config.yaml

   # Multi-node (override nodes at submit time)
   sbatch --nodes=2 scripts/train.sh configs/my_config.yaml

   # Or use dedicated launchers directly
   sbatch scripts/train_ddp.sh configs/my_config.yaml     # DDP
   sbatch scripts/train_fsdp.sh configs/my_config.yaml    # FSDP

   # Local (non-SLURM) single-node DDP/FSDP
   bash scripts/train.sh configs/my_config.yaml

See :doc:`configuration_guide` for details on the ``distributed:`` config block.

Run a parameter sweep
---------------------

.. code-block:: bash

   # Generate configs and submit jobs
   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
     --config configs/sweep_config.yaml

   # Analyze results
   python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
     /path/to/sweep/results

Example model runs
------------------

The repository includes a curated set of population-network examples plus
specialized vision and transformer replacement configs. See :doc:`examples`
for the full list.

.. code-block:: bash

   # MNIST, single-layer E/I population network
   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_feedforward.yaml

   # CIFAR-10, one 128E/32I layer, [3, 3, 3] morphology, 40E/10I synapses
   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_cifar10.yaml

   # Two-layer population network with explicit cross-layer routing
   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_two_layer.yaml

   # Memory-efficient indexed rewiring on MNIST
   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_indexed_rewire_mnist.yaml

   # AlexNet fc7 replacement with a 500E-only dendritic tensor-map core
   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/vision/alexnet_fc7_dendritic_500.yaml

   # Named Pyr/PV/SST recurrent population network
   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_recurrent_pyr_pv_sst.yaml

   # Spiking soma with rate-decoded spike readout and spike-train analysis
   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_spiking.yaml

Output structure
----------------

A typical experiment produces:

.. code-block:: text

   output/experiment_name/
   +-- config.json          # Saved resolved configuration
   +-- final_model.pt       # Final network weights
   +-- performance/         # Training metrics and plots
   +-- analysis/            # Optional analysis outputs
   `-- checkpoints/         # Optional checkpoints

WandB integration
-----------------

.. code-block:: yaml

   wandb:
     use_wandb: true
     project: dendritic-modeling

Minimal configuration example
-----------------------------

.. code-block:: yaml

   model:
     task: classification
     encoder:
       type: Identity
     core:
       type: population_network
       population_network:
         layers:
           - name: layer0
             readout_population: e
             population_defaults:
               ff_excitatory_synapses: 20
               ff_inhibitory_synapses: 5
               rec_excitatory_synapses: 0
               rec_inhibitory_synapses: 0
               somatic_synapses: true
               reactivate: true
               reactivation_init_policy: occupancy_quantile
               weight_transform: softplus
             populations:
               - name: i
                 polarity: inhibitory
                 n_neurons: 16
                 branch_factors: [1]
               - name: e
                 polarity: excitatory
                 n_neurons: 64
                 branch_factors: [3, 3]
             connections:
               - source: input
                 target: i
               - source: input
                 target: e
               - source: i
                 target: e
     decoder:
       type: Identity

   training:
     main:
       strategy: standard
       common:
         epochs: 50
         batch_size: 128
         loss_function: ce

   data:
     dataset_name: mnist
     processing:
       flatten: true
