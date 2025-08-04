FSDP Training Guide
==================

Quick Start
-----------

Use the automated FSDP workflow for large models:

.. code-block:: bash

   # Generate and submit FSDP job (recommended)
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml

   # Generate SLURM script only
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --generate-only

When to Use FSDP
-----------------

- Model doesn't fit on single GPU
- Multiple GPUs available 
- Large datasets requiring distributed training

When to Use Single GPU
----------------------

- Model fits comfortably on single GPU
- Development and debugging
- Quick experiments

.. code-block:: bash

   # Standard single-GPU training
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

Common Issues
-------------

Network Timeout
~~~~~~~~~~~~~~~

Symptoms: NCCL timeout errors

Solutions:
- Add to environment: ``export NCCL_IB_DISABLE=1``
- Use smaller batch sizes
- Check network connectivity

Memory Issues
~~~~~~~~~~~~~

Symptoms: CUDA out of memory

Solutions:
- Reduce batch size in config
- Enable gradient checkpointing
- Use smaller model variants

Job Submission Failed
~~~~~~~~~~~~~~~~~~~~

Symptoms: SLURM submission errors

Solutions:
- Check SLURM partition availability
- Verify account permissions
- Use ``--dry-run`` to debug

Configuration
-------------

FSDP settings in ``configs/config_exp_fsdp.yaml``:

.. code-block:: yaml

   fsdp:
     enabled: true
     sharding_strategy: "FULL_SHARD"
     auto_wrap_policy: "transformer"
     cpu_offload: false

Monitoring
----------

Check job status:

.. code-block:: bash

   squeue -u $USER
   tail -f slurm-*.out

Monitor GPU usage:

.. code-block:: bash

   nvidia-smi

Additional Help
---------------

- Use ``--help`` with any script for full options
- Check generated SLURM scripts in ``scripts/tools/slurm/``
- Contact support for persistent issues