Distributed Training Guide
==========================

Quick Start
-----------

Set ``distributed.mode`` in your YAML config and use the unified launcher:

.. code-block:: bash

   # DDP (data-parallel, model fits on one GPU)
   sbatch scripts/train.sh configs/my_config.yaml

   # FSDP (model too large for one GPU)
   sbatch scripts/train.sh configs/my_config.yaml

   # Multi-node (2 nodes)
   sbatch --nodes=2 scripts/train.sh configs/my_config.yaml

Or use the dedicated launchers directly:

.. code-block:: bash

   sbatch scripts/train_ddp.sh configs/my_config.yaml
   sbatch scripts/train_fsdp.sh configs/my_config.yaml

When to Use What
----------------

.. list-table::
   :header-rows: 1

   * - Mode
     - When to use
   * - ``none``
     - Model fits on one GPU; single-process debugging
   * - ``ddp``
     - Model fits on one GPU; want data-parallel speedup across GPUs
   * - ``fsdp``
     - Model does NOT fit on one GPU; parameters must be sharded

Configuration
-------------

.. code-block:: yaml

   distributed:
     mode: fsdp             # none | ddp | fsdp
     fsdp:                  # FSDP-specific knobs (only used when mode=fsdp)
       sharding_strategy: "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
       mixed_precision: true
       cpu_offload: false
       min_num_params: 1000000
       sync_module_states: true
       use_orig_params: true
       backward_prefetch: "BACKWARD_PRE"
       gradient_checkpointing: false

Node and GPU counts are controlled by the SLURM allocation
(``--nodes``, ``--gpus-per-node``), not by the YAML config.

Common Issues
-------------

Network Timeout
~~~~~~~~~~~~~~~

Symptoms: NCCL timeout errors

Solutions:

- The launch scripts set ``NCCL_SOCKET_IFNAME=ib0,eth0`` for InfiniBand.
  If your cluster uses a different interface, override this in your script.
- Use smaller batch sizes.
- Check network connectivity between nodes.

Memory Issues
~~~~~~~~~~~~~

Symptoms: CUDA out of memory

Solutions:

- Reduce ``training.main.common.batch_size``.
- Enable ``fsdp.gradient_checkpointing: true``.
- Use ``fsdp.cpu_offload: true`` (slow but saves GPU memory).
- Use ``fsdp.sharding_strategy: FULL_SHARD`` (maximum sharding).

FSDP Validation
~~~~~~~~~~~~~~~

FSDP validation uses ``PaddedDistributedEvalSampler`` which pads each
rank to equal batch counts (required by FSDP all-gather) but excludes
padding samples from the metric reduction.  This produces results
numerically identical to a non-distributed validation run.

FSDP Final Analysis
~~~~~~~~~~~~~~~~~~~

After training, the FSDP script gathers the full (unsharded) state dict
from all ranks, saves it as a checkpoint, then loads it into a plain
non-FSDP model on rank 0 for analysis.  This avoids running analysis
on sharded parameters.

Autoencoder Pretraining
~~~~~~~~~~~~~~~~~~~~~~~

FSDP does **not** support autoencoder pretraining.  If you need a
pretrained encoder, train it separately with the standard single-GPU
script, then point the config at the saved checkpoint.

Monitoring
----------

.. code-block:: bash

   squeue -u $USER
   tail -f logs/train_fsdp_*.out
   nvidia-smi
