.. _distributed-training:

Distributed training
====================

DendriNet trains on several GPUs with one of two PyTorch runtimes:

- **DDP** (``DistributedDataParallel``): every process holds a complete copy
  of the model and trains on its own shard of the training set; the gradients
  are averaged after every backward pass. ``dendritic-train`` runs it when
  ``torchrun`` launches it with several processes.
- **FSDP** (``FullyShardedDataParallel``): the parameters, gradients and
  optimizer state are sharded across the processes, so a model larger than
  one GPU can train. The separate entry point
  ``dendritic_modeling.scripts.training.train_experiments_fsdp`` runs it, with
  a plain backpropagation loop.

Both runtimes start one process per GPU and use the NCCL process-group
backend, so they need CUDA GPUs.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - ``distributed.mode``
     - Entry point
     - When to use it
   * - ``none`` (default)
     - ``dendritic-train``
     - The model fits on one GPU; one process.
   * - ``ddp``
     - ``dendritic-train`` under ``torchrun``
     - The model fits on one GPU and more GPUs should process more samples
       per step. Runs most strategies.
   * - ``fsdp``
     - ``train_experiments_fsdp`` under ``torchrun``
     - The model does not fit on one GPU. Runs ``standard`` and
       ``recurrent`` backpropagation only.

.. contents:: On this page
   :local:
   :depth: 1

Launching a run
---------------

``torchrun`` starts the processes. On one node with four GPUs:

.. code-block:: bash

   # DDP
   torchrun --nproc_per_node=4 \
     -m dendritic_modeling.scripts.training.train_experiments \
     configs/my_config.yaml

   # FSDP
   torchrun --nproc_per_node=4 \
     -m dendritic_modeling.scripts.training.train_experiments_fsdp \
     configs/config_exp_fsdp.yaml

On several nodes, start ``torchrun`` on every node with the node count, the
node's rank and the address of the first node:

.. code-block:: bash

   torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
     --master_addr=<first-node-address> --master_port=29500 \
     -m dendritic_modeling.scripts.training.train_experiments_fsdp \
     configs/my_config.yaml

Both entry points take the options of ``dendritic-train``
(:ref:`cli-reference`), such as ``--output_dir``, ``--seed`` and
``--epochs``. ``--validate-only`` under ``torchrun`` builds the datasets and
the model, wraps the model (DDP or FSDP), builds the optimizer and the
trainer, writes the construction records and exits without training. The
launcher and the scheduler allocation set the number of nodes and GPUs, not
the configuration file.

For Slurm clusters the library includes job helpers:

.. code-block:: bash

   # Write a Slurm script for one run; --dry-run prints it, --submit submits it
   python -m dendritic_modeling.scripts.tools.generate_slurm_job \
     configs/config_exp_fsdp.yaml --account ACCOUNT --partition PARTITION --dry-run

- ``dendritic_modeling.scripts.tools.generate_slurm_job`` writes a script for
  the entry point that ``distributed.mode`` selects: one GPU with ``python``
  for ``none``, and one process per GPU with ``torchrun`` on one node for
  ``ddp`` (``--gpus-per-node``, at least 2, is required) and ``fsdp`` (by
  default the smallest power of two GPUs whose sharded memory estimate fits
  ``--gpu-memory-gb``, at most ``--max-gpus-per-node``). ``--account`` and
  ``--partition`` are required.
- ``dendritic_modeling.scripts.tools.submit_fsdp_job`` does the same for an
  FSDP configuration and submits by default (``--generate-only`` writes the
  script only).
- ``dendritic_modeling.scripts.tools.estimate_gpu_requirements`` estimates the
  GPU memory of a configuration (``--batch-size``, ``--precision``).
- ``scripts/train_fsdp.sh`` in the repository is an ``sbatch`` script that
  starts ``train_experiments_fsdp`` under ``torchrun`` on every node of the
  allocation, for multi-node FSDP runs. It takes the configuration and the
  number of GPUs per node, runs the checkout it is submitted from, and names
  no partition or account: pass them to ``sbatch``.

Sweeps launch DDP and FSDP conditions through ``slurm_config`` (:doc:`sweeps`).

.. _distributed-training-section:

The distributed section
-----------------------

The ``distributed`` section of an FSDP configuration, with the two training
options that every FSDP run of a model with sparse synapses needs
(:ref:`distributed-training-fsdp-restrictions`):

.. code-block:: yaml

   distributed:
     mode: fsdp                 # none | ddp | fsdp
     timeout_seconds: 3600      # null: 600 s under train_experiments, 1800 s under train_experiments_fsdp
     fsdp:
       sharding_strategy: FULL_SHARD
       auto_wrap_policy: indexed_synapses
       mixed_precision: true
       mixed_precision_dtype: bf16
   training:
     main:
       common:
         synapse_decay_rate: 0.0    # FSDP runs no sparse-synapse maintenance
         weight_boosting: false

``distributed.mode`` declares the runtime, and each entry point checks it
against its launch. ``train_experiments`` trains in one process (``none``) or
with DDP under ``torchrun`` (``ddp``); ``train_experiments_fsdp`` trains with
FSDP and needs ``torchrun``. A mode the entry point cannot run raises:
``fsdp`` under ``train_experiments``, ``ddp`` under the FSDP entry point, and
``ddp`` without ``torchrun``. Otherwise the mode the run actually used (for
example ``none`` for ``mode: ddp`` launched with one process) replaces the
declared one, with a warning, and ``config.json`` records it. An unknown mode
fails when the configuration loads, and so does a top-level ``fsdp`` block:
FSDP settings are read only under ``distributed.fsdp``
(``dendritic-upgrade-config`` removes a top-level block that repeats them or
has ``use_fsdp: false``, and reports any other).

``distributed.timeout_seconds`` bounds how long a collective operation may
wait before the process group aborts the run. ``null`` keeps PyTorch's NCCL
default of 600 seconds under ``train_experiments`` and uses 1800 seconds
under ``train_experiments_fsdp``. Work that only rank 0 does runs while the
other ranks wait in their next collective: training-time analyses after each
DDP epoch, and the data-driven gate calibration before FSDP wraps the model.
Raise the timeout when those phases can take longer. Network settings of
NCCL (interfaces, InfiniBand) belong to the launch environment, not the
configuration.

Batches, samplers and validation
--------------------------------

``training.main.common.batch_size`` is per process: with ``W`` processes one
optimizer step sees ``W * batch_size`` samples, an epoch has ``1/W`` as many
steps, and the learning rate is not rescaled. Schedules counted in optimizer
steps or forward passes (indexed rewiring, dense-to-sparse schedules)
therefore reach a given step at a different epoch when ``W`` changes.
``training_summary.json`` records the layout under ``data_parallel``
(``world_size``, ``batch_size_per_rank``, ``global_batch_size``,
``train_steps_per_epoch``), and ``runtime_environment.json`` records the
process group.

The training set is split across the ranks by a ``DistributedSampler`` seeded
with the loader seed; it reshuffles every epoch. Validation pads every rank to
the same number of batches (``PaddedDistributedEvalSampler``) and leaves the
padding out of the metrics, so the reported loss and accuracy are the
sample-weighted means over the validation set, as in a single-process run, up
to floating-point rounding and the configured mixed precision.

Each rank trains on its own batch. Synapses whose training forward pass keeps
statistics of the batch (the input moments of variance Top-K, the forward
impact and source activity of release synapses) reduce their batch sums and
counts over all ranks under DDP and FSDP, one all-reduce per such synapse and
training forward pass. Every rank then keeps the statistics of the global
batch and makes the same selections and rewiring decisions, as one process
training on the combined batch would.

DDP
---

``dendritic-train`` wraps the model in ``DistributedDataParallel`` when
``torchrun`` starts it with several processes. Rank 0 creates the run
directory and writes the run records, analyses and checkpoints; the other
ranks train. Data-driven gate calibration runs before training, and rank 0's
calibrated gates are broadcast to every rank.

DDP registers the parameters that require gradients when it wraps the model,
and waits for a gradient of each of them in every backward pass. Options that
change this set after wrapping, or train without such a backward pass, raise
before the first step:

- ``freeze_layers``, ``freeze_branches``, ``freeze_branch_kl``,
  ``multi_stage`` and ``presentation_stream`` run in a single process only.
- ``voltage_stabilization``, ``homeostatic_control`` and ``ei_equilibrium``
  raise, because their warm-up loss gives the decoder no gradient.
- ``reactivation_update_mode: quantile`` or ``frozen`` raises for the
  strategies whose gradients DDP reduces (``standard``, ``recurrent``,
  ``soma_dfa``). ``fa``, ``dfa`` and ``local_ca`` compute their gradients
  themselves, average them across the ranks before clipping, and accept it.
- ``training.main.common.impact_rewiring`` raises.
- ``credit_deepst`` synapses raise (their rewiring uses credit recorded
  before gradients are averaged), and so do DeepST synapses without a
  topology seed (``model.core.sparsity.indexed.seed``, or ``indexed_seed`` in
  a population network), which would rewire differently on each rank.

The supported strategies are ``standard``, ``recurrent``, ``fa``, ``dfa``,
``soma_dfa`` and ``local_ca`` (:ref:`training-strategies`). Feedback-alignment
matrices are drawn from the run seed and the block names, so every rank
builds the same ones. MuonH relies on DDP averaging the gradients before its
nonlinear update; for the first ``muon_rank_consistency_check_steps`` steps it
compares the updated matrices across the ranks and raises if they differ.

FSDP
----

``train_experiments_fsdp`` sets a run up as ``train_experiments`` does: the
same command-line options, configuration checks, run directory, seeds,
datasets, model (including ``model.initial_checkpoint``, which every rank
loads before wrapping) and run records, so one configuration builds the same
model under both entry points. The model stays on the CPU until FSDP wraps
it, and each wrapped unit moves to its GPU as it is sharded. The FSDP trainer
builds its optimizer from ``training.main.optimizer`` and the parameter groups
of single-process training (trainable parameters only), so every parameter
trains at the rate it would have without FSDP (:ref:`parameter-groups`).

FSDP options
~~~~~~~~~~~~

``distributed.fsdp`` takes these options:

.. list-table::
   :header-rows: 1
   :widths: 26 17 57

   * - Option
     - Default
     - Meaning
   * - ``sharding_strategy``
     - ``FULL_SHARD``
     - ``FULL_SHARD``, ``SHARD_GRAD_OP``, ``NO_SHARD`` or ``HYBRID_SHARD``
       (case-insensitive).
   * - ``auto_wrap_policy``
     - ``none``
     - How the model is split into FSDP units: ``none`` (the whole model is
       one unit), ``indexed_synapses`` (one unit per indexed synapse module)
       or ``size`` (PyTorch's size-based policy).
   * - ``min_num_params``
     - ``1000000``
     - Smallest unit of ``auto_wrap_policy: size``; any other policy rejects
       a different value.
   * - ``mixed_precision``
     - ``true``
     - Compute in reduced precision.
   * - ``mixed_precision_dtype``
     - ``bf16``
     - The dtype of mixed precision for parameters, gradient reduction and
       buffers, unless set separately: ``bf16``/``bfloat16``,
       ``fp16``/``float16``/``half`` or ``fp32``/``float32``.
   * - ``param_dtype``, ``reduce_dtype``, ``buffer_dtype``
     - ``null``
     - ``param_dtype`` defaults to ``mixed_precision_dtype``;
       ``reduce_dtype`` (gradient reduction) and ``buffer_dtype`` default to
       ``param_dtype``. ``buffer_dtype: keep`` leaves buffers in their own
       dtype.
   * - ``cpu_offload``
     - ``false``
     - Keep the sharded parameters in CPU memory (slower, less GPU memory).
   * - ``sync_module_states``
     - ``true``
     - Broadcast rank 0's parameters and buffers when wrapping.
   * - ``use_orig_params``
     - ``true``
     - Keep the original parameters, which the parameter groups hold.
   * - ``limit_all_gathers``, ``forward_prefetch``
     - ``true``, ``true``
     - Passed to ``FullyShardedDataParallel``.
   * - ``backward_prefetch``
     - ``BACKWARD_PRE``
     - ``BACKWARD_PRE`` or ``BACKWARD_POST``.
   * - ``gradient_checkpointing``
     - ``false``
     - Recompute activations in the backward pass
       (:ref:`distributed-training-memory`).
   * - ``freeze_core``
     - ``false``
     - Readout-only training: freeze every core parameter before wrapping;
       the core still runs its full forward dynamics.
   * - ``cache_frozen_core_outputs``
     - ``false``
     - With ``freeze_core``, compute the core outputs of each rank's data
       once and train the readout on them in later epochs. It needs a frozen
       or parameter-free encoder.
   * - ``cached_core_batch_size``
     - ``null``
     - Batch size of the epochs that use the cached core outputs.

.. _distributed-training-fsdp-restrictions:

Options the FSDP trainer rejects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FSDP loop is plain backpropagation with an optimizer step, so the entry
point raises ``ValueError``, listing every problem, for options that it would
not apply:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Option
     - Reason
   * - ``training.main.strategy`` other than ``standard`` or ``recurrent``
     - The loop is backpropagation.
   * - ``lr_schedule``, ``lr_warmup_epochs``, and ``strategies.recurrent``
       overrides that differ from ``common``
     - There is no learning-rate scheduler; ``use_amp`` and
       ``grad_clip_value`` come from ``common``.
   * - ``training.main.regularization``, ``training.main.pruning``
     - No auxiliary losses and no pruning step.
   * - ``reactivation_update_mode`` other than ``backprop``,
       ``recalibrate_reactivation_every > 0``
     - No gate recalibration.
   * - ``synapse_decay_rate > 0`` or ``weight_boosting: true`` with Top-K,
       indexed or DeepST synapses
     - No sparse-synapse maintenance. The default ``synapse_decay_rate`` is
       0.01, so set it to 0.
   * - Rewiring synapses (``indexed_rewire``, DeepST, dense-to-sparse with
       ``advance_on_forward: false``), ``impact_rewiring``
     - No rewiring step and no topology controllers.
   * - ``best_state_device`` or ``save_best_checkpoint`` other than their
       defaults
     - The trainer keeps its final weights and tracks no best state.
   * - ``shuffle: false``
     - The FSDP sampler always shuffles.
   * - ``enable_nan_checking``, ``profile_dataloader``
     - Not implemented in the FSDP step.
   * - ``optimizer.name: muonh`` unless ``sharding_strategy: NO_SHARD`` and
       ``use_orig_params: true``
     - Sharded units expose 1-D views of the matrices that MuonH
       orthogonalizes.
   * - ``fp16`` parameters or reductions without ``use_amp: true``
     - fp16 gradients underflow without the sharded gradient scaler.
   * - ``use_orig_params: false`` with ``param_groups.split_params``
     - The parameter groups hold module parameters, which FSDP would replace
       with flat parameters.
   * - Parameters with zero dimensions
     - FSDP shards only parameters with at least one dimension. The learned
       global output scale (``model.output_scale_mode: global``) is stored as
       a one-element vector and trains under FSDP.

Losses that need the model itself, the sequence losses and autoencoder
encoders also raise (:ref:`loss-functions`). Because the default
``synapse_decay_rate`` is positive, an FSDP configuration of a model with
sparse synapses sets ``synapse_decay_rate: 0.0`` and ``weight_boosting:
false`` explicitly, as the example in :ref:`distributed-training-section`
does; ``configs/config_exp_fsdp.yaml`` is a complete FSDP configuration of a
population network with indexed synapses. The FSDP trainer clips the global
gradient norm unless ``grad_clip_mode`` says otherwise. It keeps the final
weights: with ``load_best_state_dict: true`` it logs a warning and records
``final_weights``, ``load_best_state_dict_requested`` and
``load_best_state_dict_applied`` in ``training_summary.json``.
``log_first_step_diagnostics: true`` records statistics of the first
optimizer step.

Wrapping and construction
~~~~~~~~~~~~~~~~~~~~~~~~~

``auto_wrap_policy`` decides the FSDP units. With ``none`` the whole model is
one unit, so every GPU holds the complete model while it computes.
``indexed_synapses`` makes every indexed synapse module (``indexed`` and
``indexed_dynamic`` synapses and their subclasses) its own unit, so the
parameter all-gathers of a contact-heavy model stay bounded; ``size`` wraps
every submodule with at least ``min_num_params`` parameters.

Data-driven gate calibration (for example ``occupancy_quantile``) runs on rank
0 before wrapping, whether or not analyses are enabled, and reaches the other
ranks through ``sync_module_states``; a data-driven policy with
``sync_module_states: false`` raises.

Every rank must hold the same indexed topology. With the default
``init_mode: per_rank`` each rank samples it from the same seed; with
``init_mode: rank0_broadcast`` (``model.core.sparsity.indexed.init_mode``, or
``indexed_init_mode`` in a population network) only rank 0 samples it, and
the other ranks receive it when FSDP wraps the model with
``sync_module_states: true``. Layers with ``persistent_indices: false``
regenerate their indices from an explicit seed when a checkpoint loads and
check them against a stored ``topology_fingerprint``
(:ref:`indexed-topology`).

Mixed precision
~~~~~~~~~~~~~~~

Mixed precision defaults to bf16. fp16 needs loss scaling
(``training.main.common.use_amp: true``) and raises without it. FSDP casts
every floating buffer to ``buffer_dtype`` for the rest of the run and keeps no
full-precision copy, so buffers updated in place (variance Top-K statistics)
lose precision and checkpoints store the rounded values; ``buffer_dtype:
keep`` leaves buffers in their own dtype. Recurrent layers compute their
decay factors :math:`\exp(-\Delta t/\tau)` in the parameter dtype whatever
``buffer_dtype`` is: in bf16 every :math:`\tau` above about 511 steps gives a
factor of exactly 1, and that level stops responding to input. The entry point
warns about such buffers (use ``mixed_precision_dtype: fp32`` for these time
constants) and records them under ``frozen_decay_buffers``, with the resolved
dtypes under ``fsdp_mixed_precision``, in ``training_summary.json``.

Checkpoints and final weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the end of training the ranks gather the weights once; rank 0 saves them
as a plain state dict, ``final_model.pt`` in the run directory, as
single-process training does, and runs the final analyses on an unsharded copy
of the model. The FSDP trainer runs no analyses during training, so analyzers
with ``training: true`` run only on the final model.

With ``experiment.checkpointing.enabled: true`` rank 0 also writes, under
``<run_dir>/checkpoints/``:

- ``checkpoint_epoch_<e>.pt`` every ``save_every_n_epochs`` epochs, where
  ``e`` counts epochs from 0; none is written after the last epoch;
- ``best_model.pt`` at every validation improvement, when
  ``early_stopping`` is true;
- ``final_checkpoint.pt`` at the end of training.

These files come from ``training.fsdp_utils.save_fsdp_checkpoint``: the
unsharded model and optimizer state, the epoch, and the random generators of
every rank (Python, NumPy, CPU and the rank's CUDA device).
``training.fsdp_utils.load_fsdp_checkpoint(model, optimizer, path)`` restores
them into a newly wrapped model and optimizer and returns the epoch, so
training continued on the same number of ranks draws the same dropout masks
and stochastic synapse selections as the uninterrupted run. A checkpoint from
a different number of ranks, or one without generator state, raises unless
``restore_random_state=False``. The training scripts have no resume option:
call ``load_fsdp_checkpoint`` from your own launcher.

.. _distributed-training-memory:

Memory
~~~~~~

When a run does not fit, in roughly this order:

- Reduce ``training.main.common.batch_size``.
- Use ``auto_wrap_policy: indexed_synapses`` (or ``size``) so that no GPU
  holds the whole model at once.
- Keep ``sharding_strategy: FULL_SHARD``, which shards parameters, gradients
  and optimizer state.
- Enable ``distributed.fsdp.gradient_checkpointing: true``. It recomputes, in
  the backward pass, the outermost dendritic branch layers, E/I layers and
  networks, and ``nn.Linear`` layers with more than 1000 inputs. Models whose
  synapses change state in their training forward pass (dense-to-sparse
  schedules advanced on the forward pass, variance Top-K, release synapses)
  reject it, because the recomputation would replay those updates and compute
  gradients for a different mask.
- Enable ``distributed.fsdp.cpu_offload: true`` (slower).
- Train only the readout with ``freeze_core: true`` and, for recurrent cores,
  ``cache_frozen_core_outputs: true``.

The entry point records the peak GPU memory, the maximum over the ranks, in
``model_resources.json``.

Troubleshooting
---------------

NCCL timeouts
   Raise ``distributed.timeout_seconds`` when rank 0 runs long analyses or
   calibration while the other ranks wait. If NCCL selects the wrong network
   interface, set ``NCCL_SOCKET_IFNAME`` (and, where needed,
   ``NCCL_IB_DISABLE``) in the launch environment, and check the network
   connectivity between the nodes.

Out of GPU memory
   See :ref:`distributed-training-memory`.

An option is rejected at start
   The FSDP entry point lists every option it would not apply in one error;
   remove them or train with ``train_experiments``. DDP rejects the
   strategies and options listed under DDP.

Autoencoder encoders
   Encoder pretraining (:ref:`encoder-pretraining`) runs in single-process
   training. A DDP run starts from the checkpoint of a single-process run
   (``model.initial_checkpoint``); the FSDP entry point rejects autoencoder
   encoders.

Monitoring
   Rank 0 writes the run's log files, ``dendritic_modeling.log`` and
   ``train.log``, into the run directory (:ref:`logging`). Watch GPU memory
   with ``nvidia-smi``; when a scheduler runs the job, follow its job output
   file.
