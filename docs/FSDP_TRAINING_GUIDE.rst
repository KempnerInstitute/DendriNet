FSDP Training Guide
==================

FSDP (Fully Sharded Data Parallel) training for large dendritic models across multiple GPUs.

Overview
--------

FSDP enables training of large models by sharding parameters across GPUs, using mixed precision, and optimizing communication. The framework provides three approaches:

1. Automated job submission with resource estimation
2. Direct FSDP script for manual training
3. Manual workflow for step-by-step control

Automated Job Submission
------------------------

The submit_fsdp_job.py script automates GPU requirement estimation, SLURM script generation, and job submission.

Basic Usage:

.. code-block:: bash

   # Generate SLURM script only (default)
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml

   # Generate and submit job
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit

   # Dry run (see what would be submitted)
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit --dry-run

Override SLURM Settings:

.. code-block:: bash

   # Force 4-GPU training with specific cluster settings
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit \
     --gpu-per-node 4 --partition kempner_h100 --account my_account --time "12:00:00"

The script outputs model parameters, memory requirements, and generated script paths.

Direct FSDP Training
-------------------

Single Node (4 GPUs):

.. code-block:: bash

   torchrun --nproc_per_node=4 \
     src/dendritic_modeling/scripts/training/train_experiments_fsdp.py \
     configs/config_exp_fsdp.yaml

Multi-Node (2 nodes, 4 GPUs each):

.. code-block:: bash

   # Node 0 (master)
   torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
     --master_addr=NODE0_IP --master_port=29500 \
     src/dendritic_modeling/scripts/training/train_experiments_fsdp.py \
     configs/config_exp_fsdp.yaml

   # Node 1 (worker)
   torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
     --master_addr=NODE0_IP --master_port=29500 \
     src/dendritic_modeling/scripts/training/train_experiments_fsdp.py \
     configs/config_exp_fsdp.yaml

FSDP Configuration
-----------------

Required FSDP settings:

.. code-block:: yaml

   train:
     # Enable FSDP
     use_fsdp: true
     
     # FSDP-specific configuration
     fsdp_config:
       sharding_strategy: "FULL_SHARD"       # Full parameter sharding
       mixed_precision: true                 # Use FP16 for memory efficiency
       cpu_offload: false                    # Keep on GPU for speed
       min_num_params: 10000                # Wrap layers with more params
       reduce_communication_overhead: true   # Optimize communication
       limit_all_gathers: true              # Reduce memory pressure
       forward_prefetch: true               # Prefetch for forward pass
       sync_module_states: true             # Sync across ranks
       use_orig_params: true                # Important for dendritic models
       backward_prefetch: "BACKWARD_PRE"    # Prefetch for backward
       gradient_checkpointing: false        # Trade compute for memory

   # Optional: SLURM configuration for automated submission
   slurm_config:
     account: kempner_dev
     partition: kempner_h100
     nodes: 1
     gpu_per_node: 4
     training_script: src/dendritic_modeling/scripts/training/train_experiments_fsdp.py
     modules_to_load: []
     conda_env_path: dendritic_env

Model Configuration for FSDP:

.. code-block:: yaml

   model:
     core_network:
       type: EINet
       parameters:
         excitatory_layer_sizes: [100, 80, 60, 40]
         inhibitory_layer_sizes: [25, 20, 15, 10]
         excitatory_branch_factors: [8, 6, 6, 4]
         inhibitory_branch_factors: [4, 3, 3, 2]
         ee_synapses_per_branch_per_layer: [60, 50, 40, 30]
         ei_synapses_per_branch_per_layer: [20, 15, 12, 8]
         ie_synapses_per_branch_per_layer: [30, 25, 20, 15]
         ii_synapses_per_branch_per_layer: [10, 8, 6, 4]

Manual Workflow
--------------

Step-by-step control:

.. code-block:: bash

   # Step 1: Estimate requirements
   python src/dendritic_modeling/scripts/tools/estimate_gpu_requirements.py configs/config_exp_fsdp.yaml

   # Step 2: Generate SLURM script
   python src/dendritic_modeling/scripts/tools/generate_slurm_job.py configs/config_exp_fsdp.yaml

   # Step 3: Submit job
   sbatch src/dendritic_modeling/scripts/tools/slurm/config_exp_fsdp.sbatch

Common Issues
------------

Memory errors: Reduce batch_size or enable cpu_offload
Communication timeouts: Check network connectivity
Model initialization errors: Use meta device initialization