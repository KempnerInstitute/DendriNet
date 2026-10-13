Sparse execution and performance
================================

An indexed synapse layer stores, for each output, only its active contacts:
``K`` presynaptic input indices and ``K`` weights, with no dense
``[out_features, in_features]`` matrix. Its projection is

.. math::

   y_{m,j} = \sum_{k=1}^{K} w_{j,k}\, x_{m,\,i_{j,k}},

where :math:`i_{j,k}` is the fixed index of contact :math:`k` of output
:math:`j` and :math:`m` runs over the rows of the flattened input. The
synapse types that store their contacts this way are ``indexed``,
``indexed_dynamic``, ``indexed_rewire`` and ``indexed_release``
(:ref:`synapse-variants` describes them).

One option, ``projection_backend``, selects how the projection runs. It never
changes the sampled topology or the model equation: the backends differ in
speed, memory, determinism of the input gradient and the rounding of the
result. The option sits at a different path in each model family:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Model
     - Configuration path
   * - Structured E/I cores (``einet`` and its aliases, the recurrent E/I
       cores built from them, spatial dendritic cores)
     - ``model.core.sparsity.indexed.projection_backend``
   * - Population networks
     - ``model.core.population_network.layers[].population_defaults.indexed_projection_backend``,
       or per population
       ``model.core.population_network.layers[].populations[].population.indexed_projection_backend``
   * - ``unified_ei`` cores
     - ``model.core.unified_ei.layers[].excitatory.indexed_projection_backend``
       and ``model.core.unified_ei.layers[].inhibitory.indexed_projection_backend``
   * - Heterogeneous-leak CTRNN
     - ``model.core.heterogeneous_leak_ctrnn.indexed_projection_backend``
   * - Legendre memory
     - ``model.core.legendre_memory.indexed_projection_backend``
   * - Affine MLP control with ``mode: fixed``
     - ``model.core.affine_mlp.projection_backend``

The default is ``eager`` in every family. :ref:`indexed-defaults` gives the
recommended starting settings.

.. _indexed-backends:

Projection backends
-------------------

``projection_backend`` accepts ten values. "Atomic" in the gradient column
means the input gradient is accumulated with atomic additions, so repeated
runs can differ in the last bits.

.. list-table::
   :header-rows: 1
   :widths: 16 34 22 16 12

   * - Value
     - Implementation
     - Requirements
     - Input gradient
     - Chosen by ``auto``
   * - ``eager``
     - PyTorch gather, multiply and sum over bounded chunks of output rows.
       Autograd keeps the gathered inputs of each chunk for the backward pass.
       The reference for comparisons, and the only backend of
       ``indexed_release``.
     - Any device and floating-point dtype.
     - PyTorch (see the note below).
     - No.
   * - ``recompute``
     - The same chunked computation in a custom autograd function. Forward
       saves the input, the weights and the indices; backward gathers again,
       one chunk at a time. Float16 and bfloat16 input gradients accumulate in
       float32. Supports second derivatives, not forward-mode derivatives.
     - Any device and floating-point dtype.
     - PyTorch (see the note below).
     - Yes, where no Triton kernel is available or fits.
   * - ``contiguous_blocks``
     - Batched matrix multiply for the layout in which every output row reads
       one aligned block of ``K`` consecutive inputs, in order.
     - ``in_features`` a multiple of ``K`` and indices in that layout;
       other indices raise ``ValueError``.
     - PyTorch (see the note below).
     - No.
   * - ``scatter_dense``
     - Scatters each row's contacts into a dense ``[out_features,
       in_features]`` matrix and runs the forward pass and both gradients as
       dense matrix multiplies. Suited to rows whose ``K`` is a large fraction
       of the input width.
     - Any device. Allocates the dense matrix on every call.
     - PyTorch (see the note below).
     - No.
   * - ``triton_transposed``
     - Triton kernels that read the input in transposed, batch-contiguous
       layout. The input gradient uses a two-stage sorted (CSR) reduction
       without atomics.
     - Triton and a CUDA device; float16, bfloat16 or float32 inputs and
       weights; the transposed int32 limits.
     - Deterministic. Atomic, with a one-time ``RuntimeWarning``, when one
       ``[in_features, batch]`` float32 block exceeds 2 GiB or
       ``out_features * K`` reaches ``2^31``.
     - Yes, below 100,000 inputs (and above, when ``triton_fused`` does not
       fit or deterministic algorithms are required).
   * - ``triton_transposed_atomic``
     - The ``triton_transposed`` forward and weight-gradient kernels with an
       atomic input-gradient scatter.
     - As ``triton_transposed``.
     - Atomic.
     - No.
   * - ``triton_transposed_bf16_compat``
     - Transposed kernels that round every product to bfloat16 before the
       float32 sum, in the forward pass and both gradients. This matches the
       product rounding of a bfloat16 model on ``eager``; use it to compare the
       two executions of the same bfloat16 model.
     - Triton and a CUDA device; bfloat16 inputs and bfloat16 weights only
       (other dtypes raise); the transposed int32 limits.
     - Deterministic, with the same atomic fallback as
       ``triton_transposed``.
     - No.
   * - ``triton_fused``
     - One fused kernel body computes the forward pass, and a second launch
       computes both gradients from shared index and weight reads. Designed
       for input widths of at least 100,000 at small batches (up to 16
       flattened input rows).
     - Triton and a CUDA device; float16, bfloat16 or float32;
       ``in_features * batch < 2^31``.
     - Atomic; the weight gradient is atomic too above 16 flattened rows.
     - Yes, at 100,000 inputs or more, unless deterministic algorithms are
       required.
   * - ``triton_ell``
     - Fused ELL gather-GEMM kernel with a feature-major copy of the input on
       each call. The weight gradient has no atomics.
     - Triton and a CUDA device; the transposed int32 limits.
     - Atomic.
     - No.
   * - ``auto``
     - Chooses one of the backends above per layer; see below.
     - None.
     - That of the chosen backend.
     - Not applicable.

Note on determinism: the gradients of the PyTorch backends (``eager``,
``recompute``, ``contiguous_blocks`` and ``scatter_dense``) come from PyTorch
operations, so on a CUDA device their run-to-run determinism follows PyTorch's
deterministic-algorithms setting, which ``experiment.strict_deterministic:
true`` enables (:ref:`random-seeds`). The Triton backends follow the same
setting. Under ``torch.use_deterministic_algorithms(True)``, ``auto`` resolves
only to backends whose gradients are deterministic, and a projection that an
explicitly selected backend would differentiate with atomic additions raises
``RuntimeError``: ``triton_transposed_atomic``, ``triton_fused`` and
``triton_ell`` when the input needs a gradient, ``triton_fused`` also for the
weight gradient above 16 flattened rows, and the transposed kernels past
their deterministic workspace limits. Projections without gradients run on
every backend, since all forward passes are deterministic. In PyTorch's
``warn_only`` mode these cases warn and run.
``nondeterministic_gradient_reason`` in
``dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels``
reports whether a backend's gradients are deterministic at a given shape.

All Triton backends accumulate in float32. The output has the dtype of the
input, and each gradient has the dtype of the tensor it belongs to. Indices
may be stored as int32 or int64; the transposed and fused kernels convert
them to int32.
``pip install -e '.[gpu]'`` installs Triton; CUDA builds of PyTorch on Linux
usually include it.

.. _indexed-backend-resolution:

How ``auto`` chooses
~~~~~~~~~~~~~~~~~~~~

On a CUDA device where Triton and the kernel modules are available, ``auto``
selects ``triton_fused`` for a layer with at least 100,000 inputs
(``in_features``) and ``triton_transposed`` otherwise. On any other device, or
without Triton, it selects ``recompute``. ``auto`` never selects ``eager``,
``contiguous_blocks``, ``scatter_dense``, ``triton_transposed_atomic``,
``triton_transposed_bf16_compat`` or ``triton_ell``.

Under deterministic algorithms (``experiment.strict_deterministic``),
``auto`` does not select ``triton_fused``, and it selects ``recompute``
instead of ``triton_transposed`` when the deterministic input gradient cannot
run the layer's shape.

Each layer resolves its backend at its first call and reuses the choice for
later calls on the same device. Moving the layer to another device, or
assigning another value to its ``projection_backend`` attribute, makes it
resolve again. When one call has a shape that the cached kernel's int32 limits
do not allow, that call alone runs on a kernel that fits: ``triton_fused``
when the layer has at least 100,000 inputs and its limit allows, otherwise
``triton_transposed`` when it fits, otherwise ``recompute``. Likewise, a call
that would differentiate the cached kernel with atomic additions while
deterministic algorithms are required runs on the deterministic choice.

``auto`` is a portability rule, not a guarantee that the chosen kernel is the
fastest for a given model, batch size and GPU. Kernel launch and first-use
compilation can dominate small recurrent workloads. Measure the alternatives
with the tools in :ref:`indexed-benchmarks`. To reproduce a published result,
pin the backend and software environment recorded with it.

.. _indexed-int32-limits:

int32 offset limits
~~~~~~~~~~~~~~~~~~~

Some kernels compute element offsets in 32-bit integers. Here ``batch`` is the
number of rows of the flattened input (all leading dimensions multiplied, for
example batch size times image positions).

- ``triton_transposed``, ``triton_transposed_atomic``,
  ``triton_transposed_bf16_compat`` and ``triton_ell`` need
  ``in_features * batch``, ``batch * out_features`` and
  ``out_features * K`` each below ``2^31``.
- ``triton_fused`` needs ``in_features * batch`` below ``2^31``.

An explicitly selected kernel raises ``ValueError`` for a call that exceeds
its limits; ``auto`` falls back for that call as described above. In Python,
``indexed_kernel_shape_violations(backend, batch_size=..., in_features=...,
out_features=..., synapses_per_output=...)`` returns the products a shape
exceeds, and ``resolve_indexed_projection_backend`` applies the same limits
when it is given the full shape. Both are in
``dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels``.

Explicit selection, older names and checking the choice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only ``auto`` changes the algorithm on its own. An explicitly selected
backend fails instead of falling back:

- a Triton backend on a CPU, without CUDA or without Triton raises
  ``RuntimeError`` at the layer's first call;
- a shape beyond the backend's int32 limits raises ``ValueError``;
- ``contiguous_blocks`` raises ``ValueError`` when the indices are not in its
  block layout.

The configuration loader accepts two retired values with a
``DeprecationWarning``: ``triton`` loads as ``triton_transposed`` and
``cuda`` as ``auto``. ``recompute_backward: true``
(``model.core.sparsity.indexed.recompute_backward``, or the flat field
``indexed_recompute_backward``) is an older spelling of
``projection_backend: recompute``. It combines only with
``projection_backend`` ``eager`` (the default) or ``recompute``; with any other
value the configuration raises. Unknown keys under
``model.core.sparsity.indexed`` raise when the configuration loads.

Each indexed layer logs the backend it resolved at its first call, and again
whenever the resolution changes, as an INFO record such as
``IndexedSparseLinear projection backend resolved: requested=auto,
resolved=recompute, device=cpu`` (``IndexedDynamicTopKLinear`` for
``indexed_dynamic`` layers). Training runs keep these records in
``dendritic_modeling.log`` in the run directory (:ref:`logging`).
``benchmark_configured_step`` also counts the layers on each configured and
each resolved backend (:ref:`indexed-benchmarks`).

Some analyses need second-order or forward-mode derivatives. The Triton
kernels compute the same operator, but their backward passes are not
differentiable, and ``recompute`` has no forward-mode rule. Run such
computations on ``eager`` with a context manager that switches every indexed
layer of a module and restores the configured backends on exit:

.. code-block:: python

   from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.backend import (
       indexed_projection_backend_override,
   )

   with indexed_projection_backend_override(model, "eager"):
       outputs = model(inputs)

.. _indexed-defaults:

Default and recommended settings
--------------------------------

The configuration default is ``projection_backend: eager`` with
``index_dtype: int64``. For a model with indexed synapses, start from
``configs/templates/indexed_runtime_template.yaml``: it sets a topology seed,
``projection_backend: auto``, ``index_dtype: auto`` (int32 indices whenever
they fit) and memory-efficient ``param_tanh`` reactivation. The following
experiment excerpt applies those settings to a small structured E/I core:

.. code-block:: yaml

   model:
     core:
       type: einet
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: [16]
         excitatory_branch_factors: [4, 4]
       connectivity:
         ee_synapses_per_branch_per_layer: [32]
         ei_synapses_per_branch_per_layer: [32]
         ie_synapses_per_branch_per_layer: [8]
       sparsity:
         type: indexed
         indexed:
           seed: 42
           index_dtype: auto
           projection_backend: auto
       reactivation:
         type: param_tanh
         memory_efficient: true

Population networks and ``unified_ei`` populations use the flat field names.
This ``population_defaults`` excerpt sets the same options for every
population of a layer:

.. code-block:: yaml

   population_defaults:
     topk_type: indexed
     indexed_seed: 42
     indexed_index_dtype: auto
     indexed_projection_backend: auto
     reactivation_type: param_tanh
     reactivation_memory_efficient: true

Keep the defaults of the other options unless a measurement shows that a
change helps. For runs without a GPU, ``auto`` resolves to ``recompute``.

.. _indexed-options:

Indexed synapse options
-----------------------

These keys sit under ``model.core.sparsity.indexed`` for structured E/I
cores. Population networks and ``unified_ei`` populations use the flat field
in the second column, in
``model.core.population_network.layers[].population_defaults`` (or a
population's ``population`` mapping) and in
``model.core.unified_ei.layers[].excitatory`` / ``.inhibitory``.

.. list-table::
   :header-rows: 1
   :widths: 24 30 12 34

   * - Key
     - Flat field
     - Default
     - Meaning
   * - ``projection_backend``
     - ``indexed_projection_backend``
     - ``eager``
     - Execution backend (:ref:`indexed-backends`).
   * - ``index_dtype``
     - ``indexed_index_dtype``
     - ``int64``
     - Stored index dtype: ``int32``, ``int64``, or ``auto`` (int32 when
       ``in_features`` is at most ``2^31 - 1``).
   * - ``seed``
     - ``indexed_seed``
     - ``null``
     - Topology seed. ``dendritic-train`` fills an unset seed from the run's
       topology seed (:ref:`random-seeds`).
   * - ``output_chunk_size``
     - ``indexed_output_chunk_size``
     - ``2048``
     - Output rows per chunk of the ``eager`` and ``recompute`` gathers.
   * - ``workspace_mb``
     - ``indexed_workspace_mb``
     - ``null``
     - Memory budget in MiB for one gathered chunk of ``eager`` and
       ``recompute``; ``null`` sets no budget.
   * - ``cache_transformed_weights``
     - ``indexed_cache_transformed_weights``
     - ``false``
     - In recurrent cores, transform the weights once per unroll instead of
       at every time step.
   * - ``persistent_indices``
     - ``indexed_persistent_indices``
     - ``true``
     - Save the indices in checkpoints; ``false`` regenerates them from the
       seed (:ref:`indexed-topology`).
   * - ``init_mode``
     - ``indexed_init_mode``
     - ``per_rank``
     - ``per_rank``: every process samples the topology from the seed.
       ``rank0_broadcast``: rank 0 samples it and FSDP broadcasts it.
   * - ``recompute_backward``
     - ``indexed_recompute_backward``
     - ``false``
     - Older spelling of ``projection_backend: recompute``.
   * - ``candidate_size``
     - ``indexed_candidate_size``
     - ``null``
     - ``indexed_dynamic``: candidate contacts per output; ``null`` means
       ``min(in_features, 4 * K)``.
   * - ``selection``
     - ``indexed_selection``
     - ``standard``
     - ``indexed_dynamic``: ``standard`` or ``stochastic`` (alias
       ``rank_probabilistic``) selection among the candidates.
   * - ``rewire_frequency``
     - ``indexed_rewire_frequency``
     - ``100``
     - ``indexed_rewire``: optimizer steps between rewiring events.
   * - ``rewire_quantile``
     - ``indexed_rewire_quantile``
     - ``0.05``
     - ``indexed_rewire``: fraction of each row's contacts replaced at an
       event (at least one contact; ``0`` disables rewiring).
   * - ``rewire_until_step``
     - ``indexed_rewire_until_step``
     - ``null``
     - ``indexed_rewire``: last step that can rewire; ``null`` sets no
       limit.
   * - ``release_*``
     - the same names
     - see link
     - Options of ``indexed_release`` (:ref:`conductance-indexed-release`).

These keys apply to the indexed synapse types only. The other ``sparsity.type``
(``topk_type``) values load them and ignore them without an error.
``indexed_rewire`` and ``indexed_release`` require ``persistent_indices:
true``; ``indexed_release`` also requires ``projection_backend: eager`` and
``cache_transformed_weights: false``.

Other model families read a subset under their own section:

- ``model.core.heterogeneous_leak_ctrnn`` and ``model.core.legendre_memory``:
  ``indexed_projection_backend``, ``indexed_output_chunk_size``,
  ``indexed_workspace_mb``, ``indexed_cache_transformed_weights``,
  ``indexed_recompute_backward`` and ``indexed_persistent_indices``, with the
  defaults above. Their topology seed is ``topology_seed``.
- ``model.core.affine_mlp`` with ``mode: fixed``: ``projection_backend``
  (``eager``), ``workspace_mb`` (``128.0``) and ``index_dtype`` (``int32``);
  :ref:`baseline-models` describes the control.

.. _indexed-memory:

Memory controls
---------------

Gather workspace
   ``eager`` and ``recompute`` gather ``[batch, chunk, K]`` input values for
   one chunk of output rows at a time. ``output_chunk_size`` sets the number
   of rows per chunk; ``workspace_mb`` lowers it, down to one row, so that a
   chunk fits the budget. The same chunking bounds the frozen-layer tiles
   below and the weight gradients of local learning rules on indexed layers.
   The Triton, ``contiguous_blocks`` and ``scatter_dense`` projections ignore
   both options. Neither option changes the sampled topology.

Frozen layers
   A layer whose weights do not require gradients (or a core frozen with
   ``distributed.fsdp.freeze_core``), called with an input that does not
   require gradients, projects in tiles that also bound the number of input
   rows gathered at once. Each tile stays within ``workspace_mb``, or 256 MiB
   when it is unset. Tiling applies to the ``eager``, ``recompute`` and
   ``auto`` selections, so a frozen ``auto`` layer runs this tiled path on
   every device. It does not apply with weight normalization or the recurrent
   weight cache, and layers configured with ``contiguous_blocks``,
   ``scatter_dense`` or a Triton backend run that backend. Tiling does not
   change any output value.

Index dtype and checkpoint size
   ``index_dtype: auto`` or ``int32`` stores four bytes per contact instead
   of eight. ``persistent_indices: false`` removes the indices from
   checkpoints (:ref:`indexed-topology`).

Recurrent weight cache
   ``cache_transformed_weights: true`` keeps one transformed copy of the
   weights for a whole recurrent unroll. It has no effect on a layer that
   scales its gradients per forward pass
   (``sparsity.gradient_scaling: conductance_dynamic``).

Input-gradient setup cache
   The deterministic input gradient of ``triton_transposed`` and
   ``triton_transposed_bf16_compat`` precomputes a sorted (CSR) form of each
   layer's indices and caches it. The environment
   variables ``DENDRITIC_GX_SETUP_CACHE_MAX`` (entries, default 64) and
   ``DENDRITIC_GX_SETUP_CACHE_MAX_BYTES`` (default 8 GiB) bound the cache;
   the least recently used entries are evicted first, and one entry larger
   than the byte bound is still kept. Both must be positive integers, are
   read when the kernel module is imported, change memory use only and are
   recorded in each run's ``runtime_environment.json``.

Inference weight folding
   ``freeze_for_inference()`` on an ``IndexedSparseLinear`` or ``BlockLinear``
   module computes its transformed weights once, stops their gradients and
   switches the module to evaluation mode; later evaluation calls reuse the
   folded weights and give the same results. ``train()`` drops the folded copy
   and restores the gradients. ``indexed_release`` layers raise.

.. _memory-efficient-reactivation:

Memory-efficient reactivation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``model.core.reactivation.memory_efficient: true`` (flat field
``reactivation_memory_efficient``, default ``false``) runs the ``param_tanh``
and ``param_tanh_only_m`` gates through a custom autograd function that keeps
only the gate output for the backward pass. In shunting layers the division
and the gate are fused and keep two full-size tensors. Forward values and
gradients follow the same equation as the default path.

The fused shunting path is skipped, with the same forward result, when the
layer stores diagnostics or analysis currents, rounds signals
(``signal_rounding``), runs under a branch-rule override, or has hooks on its
gate, such as the reactivation ``gradient_scaling`` hook or the hooks of gate
calibration. The other reactivation types, including ``endpoint_tanh``, have
no memory-efficient path: they accept ``memory_efficient: true`` and ignore
it. :ref:`activation-types` lists the reactivation types.

.. _indexed-topology:

Compact topology and checkpoints
--------------------------------

Sampling
   A layer samples its contacts once, at construction, from its seed, its
   shape and its connection constraints (structured masks and excluded
   inputs). Execution settings such as ``output_chunk_size`` and
   ``projection_backend`` do not affect the sampled graph, so the same seed,
   shape and constraints always give the same contacts.

Indices outside checkpoints
   With ``persistent_indices: false`` a checkpoint stores a
   ``topology_fingerprint`` (a hash of the index tensor) instead of the
   indices. When the checkpoint loads, the layer regenerates its indices from
   the seed and compares the fingerprint; a different topology (another seed,
   or a library version that samples differently) raises, and a checkpoint
   without a fingerprint loads with a warning that its topology cannot be
   checked. Load such a checkpoint with the library version that wrote it, or
   keep ``persistent_indices: true`` for checkpoints that must outlive library
   updates. Building such a layer without a seed raises.
   ``dendritic-train`` fills unset seeds from the run's topology seed and
   records them in the run's ``config.json``, so rebuild the model from that
   file. Among the synapse types, only ``indexed`` and ``indexed_dynamic``
   accept ``persistent_indices: false``; ``indexed_rewire`` and
   ``indexed_release`` raise, because their topology can change during
   training.

Distributed initialization
   With ``init_mode: rank0_broadcast`` and an initialized process group, only
   rank 0 samples the topology; the other ranks hold placeholder indices
   until FSDP broadcasts the module state, which
   ``distributed.fsdp.sync_module_states: true`` (the default) does. A forward
   pass before that broadcast raises.
   :ref:`distributed-training` describes wrapping each indexed synapse module
   as its own FSDP unit.

Resource accounting (the run's ``model_resources.json`` and the
compact-checkpoint report) counts the contacts of indexed layers from their
stored indices, without building a dense matrix. After training,
``dendritic-compact-checkpoint`` converts Top-K, dense-to-sparse,
``indexed_dynamic`` and ``indexed_rewire`` layers into fixed indexed layers
(:ref:`compact-checkpoints`).

.. _indexed-scope:

Models and synapse types
------------------------

``projection_backend`` belongs to the shared indexed-synapse implementation,
so it applies wherever an indexed layer runs:

- structured E/I feedforward and recurrent cores and the spatial dendritic
  cores (``tensor_map``, ``hierarchical_dendritic_conv`` and their aliases);
- population networks and ``unified_ei`` cores;
- the degree-grouped indexed projections of the heterogeneous-leak CTRNN and
  Legendre-memory cores;
- the ``fixed`` mode of the affine MLP control, and the parameter-budget
  models and indexed readouts of :ref:`baseline-models` (their
  ``projection_backend`` argument).

``indexed_dynamic`` layers run the selected backend over their whole
candidate pool, with zero weight on unselected candidates.

The dense-candidate Top-K types (``standard``, ``stochastic``, ``variance``,
``dense_to_sparse``) and the DeepST types (``deepst``, ``credit_deepst``) do
not use the selector. They keep a dense parameter bank and multiply a masked
weight matrix. Replacing them with fixed indexed contacts changes the model and
its learning dynamics; it is not an execution setting.

Recurrent cores call the projection at every time step. With small batches,
the launch cost of a native kernel at each step can exceed its savings, so
measure ``recompute`` against the Triton backends at ordinary input widths.
Feedforward and spatial models flatten the batch (and image positions) into
one large projection and are better suited to the Triton backends.

Compilation
-----------

``model.core.implementation.compile_forward: true`` compiles a feedforward
structured E/I network with ``torch.compile`` (dynamic shapes) when it is
built. Recurrent structured E/I cores reject it.
``model.core.population_network.compile_step: true`` compiles the step
function of every population-network layer. Recurrent networks call that
function once per time step, so compilation can fuse its many small
operations; results can differ from the uncompiled model by floating-point
reassociation. In both cases copies and pickles of the model run without
compilation. Both default to ``false``; measure graph breaks, compilation
time and complete updates before enabling them for a workload.
``compile_step`` rejects shunting populations with ``conductance_dynamic``
gradient scaling, whose scales the compiled steps do not apply, and
``compile_forward`` rejects shunting cores with ``conductance_dynamic``
BlockLinear gradient scaling, on whose hooks compilation fails; the Top-K
scaling of shunting cores and the BlockLinear rule of additive layers compile
in both.

The indexed Triton kernels fuse the gather, multiply and sum of one
projection. The dendritic tree, the gates, the optimizer update and sparse
maintenance run as separate operations. The ``dendritic_modeling.kernels``
package also provides fused E/I shunting and gated (GLU) kernels; no model
option selects them.

.. _indexed-benchmarks:

Benchmarking
------------

Kernel timings leave out contact selection, gates, autograd, gradient
clipping, optimizer updates and sparse maintenance. Benchmark the configured
model at its batch size before choosing a backend. The library provides four
tools; each prints its results and can write them as JSON.

``benchmark_configured_step``
   Times complete training updates of a configured model on one synthetic
   batch that stays on the device:

   .. code-block:: bash

      python -m dendritic_modeling.scripts.benchmarks.benchmark_configured_step \
        config.yaml --input-dim 784 --classes 10 --steps-per-epoch 128 \
        --warmup 10 --steps 200 --profile-steps 4 \
        --trace step.trace.json --output step.json

   It builds the model through the ordinary configuration path and runs the
   configured loss, gradient clipping, optimizer parameter groups, sparse
   weight maintenance and impact rewiring; ``--steps-per-epoch`` sets the
   epoch length that schedules rewiring. The configuration needs an
   ``identity`` encoder, a classification task, the ``standard`` or
   ``presentation_stream`` strategy, full precision (``use_amp: false``), no
   NaN checking, no additional regularization, backpropagated reactivation
   updates and no ``initial_checkpoint``. The output records the resolved
   configuration, library commit, device, precision flags, the number of
   layers on each configured and each resolved backend, synchronized step
   times (median, mean, maximum and each step), peak allocated memory and, with
   ``--profile-steps``, an operator profile taken in separate steps
   (``--trace`` exports it as a Chrome trace). ``--device`` defaults to
   ``cuda``. Learning rates stay fixed; data loading, validation and
   checkpointing are not included, and accuracy is not measured. Run long
   enough to include scheduled rewiring events and report their effect on the
   mean, not only the median. To profile the data path, train with
   ``training.main.common.profile_dataloader: true``.

``benchmark_training_steps``
   Runs a bounded number of optimizer steps of a configured model on its
   dataset, including data loading, and reports compute throughput, data-wait
   time and CUDA memory, with optional held-out batches (``--eval-steps``).
   It uses a cross-entropy loss with value clipping, the configured optimizer
   and parameter groups and, when configured, AMP; it runs no sparse weight
   maintenance. It needs a CUDA GPU, records the Git commit of the working
   directory and writes ``batch_<N>.json`` to ``--output-dir``.

``profile_component_matrix``
   Builds small built-in models of each component family (feedforward flat
   and branched trees, Top-K and indexed synapses, a spatial convolution core,
   a recurrent ``unified_ei`` core and a population network) and times their
   forward and forward-plus-backward passes. ``--only`` selects cases and
   ``--device`` the device. With ``DENDRITIC_NVTX=1`` each phase is an NVTX
   range for a Nsight Systems timeline.

``benchmark_indexed_kernels``
   Needs CUDA and Triton. At three built-in layer shapes it times the eager
   indexed projection, the fused ELL kernel with row-major and feature-major
   inputs, the fused E/I shunting kernel and a dense matrix multiply of the
   same layer shape.

All four run as ``python -m dendritic_modeling.scripts.benchmarks.<name>``;
``--help`` lists their options.

Compare backends on identical initial weights, connectivity, inputs and random
streams. Check outputs, input and parameter gradients and several optimizer
updates before comparing speed. Backends that compute the same equation still
sum in different orders, so training trajectories can diverge over many
updates. The elapsed time of a CUDA event pair does not measure GPU occupancy:
a profiler timeline separates active kernels, launch gaps, host
synchronization and copies. Small networks can spend more time launching
operations than computing. Independent seeds can run as separate processes on
one GPU; measure aggregate samples per second and peak memory before adding
processes. Do not change the batch size, the sparse selection rule or the
numerical precision of an experiment only to raise GPU utilization.
