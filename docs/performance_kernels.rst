Indexed Projection and Reactivation Backends
============================================

Indexed synapses have one maintained execution selector. The default remains
the original eager PyTorch implementation, so installing or importing the
package never requires Triton, a CUDA compiler, or a GPU.

.. code-block:: yaml

   sparsity:
     type: indexed
     indexed:
       projection_backend: auto   # eager | recompute | triton | cuda | auto
       index_dtype: int32
       output_chunk_size: 65536
       workspace_mb: 128

Population-network configs use the equivalent flat names
``indexed_projection_backend``, ``indexed_index_dtype``, and
``indexed_output_chunk_size``.

Backend behavior
----------------

``eager``
   The standard chunked PyTorch path. This is the compatibility default.

``recompute``
   A portable custom-autograd path that saves the input, weights, and compact
   index buffer, then re-gathers one bounded chunk during backward. Reduced
   precision input gradients accumulate in fp32 before being cast back.

``triton``
   An optional fused forward/backward implementation. It requires Triton and a
   CUDA device. Selecting it explicitly raises a clear error when unavailable;
   it never silently changes the requested algorithm.

``cuda``
   An optional JIT-compiled CUDA extension with staged, unstaged, and hybrid
   forward variants plus a split backward. The extension builds only on first
   use and is cached per GPU compute capability. It supports float16,
   bfloat16, and float32 inputs with fp32 kernel accumulation.

``auto``
   Chooses CUDA, then Triton, when usable on a CUDA device; otherwise it chooses
   ``recompute``. Resolution is cached by each layer for its current device.
   ``auto`` means capability selection, not a promise that a native kernel wins
   at every shape.

The legacy flat option ``indexed_recompute_backward: true`` remains accepted
and maps to ``recompute``. It cannot be combined with an explicit CUDA or
Triton backend.

Measured scope and safety boundary
----------------------------------

The CUDA picker records Ryan's final measured crossover bands for A100
(``sm80``), H100 (``sm90``), and Blackwell (``sm120``). H200 shares the
``sm90`` route but was not independently measured in the finalized matrix.
Unknown architectures keep the conservative staged implementation. Unit tests
replay the full recorded decision grid without requiring a GPU. GPU-marked
tests compare native forward and backward results with portable recompute.

The native kernels were measured for large feedforward indexed projections.
Ryan's batch-one recurrent measurements found the portable recompute path
faster than CUDA and Triton. Recurrent production configs should therefore
continue to select ``recompute`` unless a representative GPU benchmark shows a
different winner. This branch does not enable a native backend in an existing
experiment config.

This checkout had no CUDA device or compiler during integration. CPU tests,
picker-table tests, import checks, and portable numerical equivalence were run;
the CUDA/Triton runtime checks remain hardware-gated and must pass on each
supported cluster image before enabling a native path in a canonical config.

Memory-efficient positive tanh
------------------------------

``param_tanh`` and ``param_tanh_only_m`` can opt into custom autograd that
retains fewer full-size tensors. In shunting layers the divide and positive
tanh are fused as well:

.. code-block:: yaml

   reactivation:
     enabled: true
     type: param_tanh
     memory_efficient: true

Population-network configs use ``reactivation_memory_efficient: true``. The
default is ``false``. The fused shunt is bypassed automatically when diagnostic
or analysis-current capture needs the pre-gate voltage. CPU float32/float64
tests cover forward values and all input and parameter gradients, including
saturated activations.

Integration boundary
--------------------

Only production kernels, backend selection, configuration wiring, tests, and
maintained documentation were integrated from ``origin/ryanconfig``. Profiling
harnesses, NVTX instrumentation, benchmark runners, cluster setup scripts,
generated logs, and the snapshot ``finalized/`` tree were intentionally not
copied. Those tools belong in the separate ProfilePilot codebase; the model
repository keeps only the resulting runtime implementations and reproducible
correctness contracts.
