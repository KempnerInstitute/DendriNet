Indexed Projection and Reactivation Backends
============================================

Indexed synapses expose one execution selector. The default is the standard
PyTorch implementation, so importing and using the package does not require
Triton, a CUDA compiler, or a GPU.

.. code-block:: yaml

   model:
     core:
       sparsity:
         type: indexed
         indexed:
           projection_backend: recompute  # eager | recompute | triton | cuda | auto
           index_dtype: int32
           output_chunk_size: 65536
           workspace_mb: 128

Population-network entries use the equivalent flat fields:
``indexed_projection_backend``, ``indexed_index_dtype``,
``indexed_output_chunk_size``, and ``indexed_workspace_mb``.

Backend behavior
----------------

``eager``
   Executes the chunked PyTorch gather, multiply, and reduction. This is the
   compatibility default.

``recompute``
   Uses portable custom autograd. Forward saves the input, weights, and compact
   index buffer instead of retaining gathered inputs. Backward gathers one
   bounded output chunk at a time. Float16 and bfloat16 input gradients
   accumulate in float32 before conversion to the input dtype.

``triton``
   Uses optional Triton forward and backward kernels. It requires Triton and a
   CUDA device. An explicit request raises an error when the backend is not
   available.

``cuda``
   Uses an optional JIT-compiled CUDA extension. The extension is compiled on
   first use and cached by GPU compute capability. Its forward selector uses
   staged, unstaged, or hybrid execution for supported GPU architectures.

``auto``
   Selects CUDA, then Triton, on a CUDA device and otherwise selects
   ``recompute``. Each indexed layer caches the resolved choice for its current
   device. Use an explicit backend when an experiment requires one fixed
   implementation.

The legacy ``recompute_backward: true`` and
``indexed_recompute_backward: true`` fields remain accepted and map to
``recompute``. They cannot be combined with an explicit CUDA or Triton choice.

Supported models
----------------

The selector belongs to the shared indexed-synapse implementation, not to one
network architecture. It is available in:

* feedforward DendriNet and E/I networks;
* fixed, dynamically selected, and rewiring indexed synapses;
* degree-grouped indexed projections;
* recurrent E/I, heterogeneous-leak, and Legendre-memory networks;
* transformer FFN replacements; and
* configured AlexNet and other vision-backbone replacements whose dendritic
  core uses ``sparsity.type: indexed`` or ``indexed_rewire``.

Native CUDA and Triton execution is therefore not feedforward-only. Recurrent
models call the same projection at every timestep and produce the same model
equation and gradients. For small recurrent batches, ``recompute`` is the
recommended default because it avoids native-kernel setup and launch overhead
at every timestep. Feedforward and spatial-convolution replacements often
flatten many samples or image locations into one projection and are better
candidates for ``auto`` or an explicit native backend.

For AlexNet replacement YAML files, place the selector under the configured
core:

.. code-block:: yaml

   model:
     vision_replacement:
       enabled: true
       backbone: alexnet
     core:
       type: einet  # also hierarchical_dendritic_tensor_map or hierarchical_dendritic_conv
       sparsity:
         type: indexed
         indexed:
           projection_backend: auto
           index_dtype: int32

The selector does not apply when ``vision_replacement.core_source`` is
``pretrained_span`` because that mode retains the original backbone modules
instead of constructing indexed dendritic synapses. It also has no effect on a
core using a non-indexed sparsity type.

Complete configuration examples
-------------------------------

``configs/examples/indexed_backends/`` contains large feedforward, recurrent,
and AlexNet-fc6 replacement recipes. Each recipe expands through the standard
named-variant mechanism into ``eager``, ``recompute``, ``triton``, ``cuda``,
and ``auto`` training configs. The copy/paste reference for both projection and
reactivation settings is
``configs/templates/indexed_runtime_template.yaml``.

Dtypes and numerical behavior
-----------------------------

The native backends accept float16, bfloat16, and float32 inputs and weights.
Kernel accumulation is float32. Input and weight dtypes may differ; consistently
with the portable indexed projection, the output uses the input dtype and each
gradient is returned in its parameter's original dtype. Stored connection
indices may be int32 or int64 and are retained as compact int32 values inside
native autograd.

Explicit CUDA or Triton selection fails clearly for unsupported devices,
dependencies, shapes, or dtypes rather than silently changing algorithms.
``auto`` is the only selector that permits backend fallback.

Seeded topology compatibility
-----------------------------

Masked topology generation uses a seeded per-row permutation. This sampling
order is part of the reproducibility contract for checkpoints configured with
``persistent_indices: false``, because those checkpoints regenerate their
fixed graph from the seed. Alternative sampling algorithms must use an explicit
topology version if they consume the random stream differently.

Memory-efficient positive tanh
------------------------------

``param_tanh`` and ``param_tanh_only_m`` can opt into custom autograd that
retains fewer full-size tensors. In shunting layers, division and positive tanh
are fused when no diagnostic path requires the intermediate voltage:

.. code-block:: yaml

   model:
     core:
       reactivation:
         enabled: true
         type: param_tanh
         memory_efficient: true

Population-network entries use ``reactivation_memory_efficient: true``. The
default is ``false``. Forward values, parameter gradients, input gradients,
mixed reduced-precision inputs, and saturated activations follow the eager
equation. Diagnostic and analysis-current capture automatically uses the eager
path so the pre-gate voltage remains available.
