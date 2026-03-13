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

Distributed training with FSDP
------------------------------

.. code-block:: bash

   # Generate and submit a SLURM job
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml

   # Or generate the script without submitting
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --generate-only

Run a parameter sweep
---------------------

.. code-block:: bash

   # Generate configs and submit jobs
   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
     --config configs/EI_sweeps_example.yaml

   # Analyze results
   python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
     /path/to/sweep/results

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
     enabled: true
     project: dendritic-modeling
     name: my_experiment

Minimal configuration example
-----------------------------

.. code-block:: yaml

   model:
     task: classification
     encoder:
       type: Identity
     core:
       type: dendritic_shunting
       architecture:
         excitatory_layer_sizes: [64, 32]
         inhibitory_layer_sizes: [16, 8]
       connectivity:
         ee_synapses_per_branch_per_layer: [20]
         ie_synapses_per_branch_per_layer: [5]
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
