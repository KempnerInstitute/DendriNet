``iss373_hs`` Integration Notes
===============================

This page records the feature and performance work assembled on ``iss373_hs``
and the review boundary used when comparing it with ``origin/ryanconfig``.
It is intended to make the eventual pull-request review reproducible, not to
replace the user guides for the individual features.

Vision replacement benchmarks
-----------------------------

The branch adds reproducible AlexNet replacement experiments for ``fc6``,
``fc7``, and spatial ``conv5`` features.  The integration includes:

* independently seeded random spatial windows and active-matched spatial
  controls;
* explicit pretrained feature-span selection;
* learned top-k, retrained-span, and replacement-benchmark configurations;
* deterministic seed propagation through generated sweep configurations;
* separate run directories, manifest-aware execution, ``torchrun`` launch
  support, and resource accounting; and
* configuration, factory, sweep, and model-contract tests for the new paths.

See :doc:`configuration_guide` and :doc:`network_architectures` for the public
configuration fields and architecture names.

Safe performance changes from ``ryanconfig``
---------------------------------------------

``origin/ryanconfig`` was reviewed at commit ``acf6a3b6`` on 2026-08-08.  The
following behavior-preserving changes were selected for this branch:

* loop-invariant index and seed values are reused during indexed connectivity
  sampling and rewiring;
* DeepST reuses its previously computed allowed-connection count;
* ``DendriNet.forward`` binds each branch layer once per iteration;
* ``BlockLinear`` centralizes repeated block-index construction;
* a redundant local import was removed from truncated-softplus expectation
  calculation; and
* FSDP can initialize contact-heavy models from CPU and wrap indexed synapse
  units before moving their states to the GPU.
* indexed projections expose one validated ``eager``/``recompute``/``triton``/
  ``cuda``/``auto`` backend selector across fixed, dynamic, degree-grouped,
  recurrent, and transformer construction paths;
* the portable recompute backward accumulates reduced-precision input
  gradients in fp32 and retains compact int32 indices; and
* positive-tanh reactivation has an opt-in memory-saving autograd path, with an
  opt-in fused shunt divide that preserves diagnostic voltage capture.

The FSDP behavior is opt-in and preserves the historical initialization path
by default:

.. code-block:: yaml

   distributed:
     mode: fsdp
     fsdp:
       auto_wrap_policy: indexed_synapses
       cpu_init: true

``cpu_init`` is most useful with ``auto_wrap_policy: indexed_synapses``.  The
large-scale one-million-neuron frozen-readout configuration enables this pair.
Private strategy options are copied and removed before calling PyTorch FSDP,
so a shared configuration can safely be reused when more than one model
component is wrapped.

Equivalent work already present
--------------------------------

Several finalized optimizations already had maintained equivalents on
``iss373_hs`` and were not copied again.  Indexed synapses retain compact index
dtypes, recurrent fixed-weight transforms can be cached, and resource reports
use compact connectivity counts instead of constructing dense connectivity
records.  Keeping the current implementations avoids duplicating dispatch and
configuration paths.

See :doc:`performance_kernels` for backend semantics, supported dtypes,
measured hardware scope, config examples, and hardware-gated validation.

Excluded profiling work
-----------------------

The ``finalized/`` directory on ``ryanconfig`` is a curated snapshot of
hardware-specific experiments rather than a maintained package. Its profiling
harnesses, NVTX changes, benchmark runners, generated logs, cluster jobs, setup
scripts, and duplicate ``finalized/`` copies were not imported. They encode
cluster-specific assumptions and belong in ProfilePilot. The production kernel
and activation solutions were instead placed under the existing
``synapse/kernels`` and ``activations`` packages and wired through the shared
config translators.

Review decisions not merged
---------------------------

The review also rejected or replaced several production-adjacent diffs:

* Ryan's three independent gather booleans and environment-controlled
  activation switches were replaced by typed, explicit config fields. This
  prevents contradictory selections and silent fallback.
* Masked rejection sampling was not copied because it changes the exact
  topology produced by an existing seed. The current per-row masked sampler
  preserves seeded experiment reproducibility; a future faster sampler needs
  an explicit topology-version contract.
* The ``weight_transforms_optimized-rw.py`` scratch copy was not imported. The
  finalized profiling notes refuted the suspected transform materialization as
  a major memory source, and the maintained recurrent cache already removes the
  repeated deterministic transform from an unroll.
* Native CUDA/Triton selection is not enabled in existing recurrent configs.
  The recorded batch-one RNN result favored portable recompute, while the
  architecture-aware native picker was fitted to large feedforward shapes.
* Unrelated feature merges carried through Ryan's branch were not treated as
  profiling results. Equivalent Bayesian-timing wiring already exists on this
  branch; unrelated experimental feature additions remain outside this merge.

Validation expectations
-----------------------

Changes on this branch are checked with formatting and lint hooks, YAML and
typed-config loading, focused model and sweep tests, FSDP CPU-side unit tests,
and the broader test suite. The CUDA picker table is checked on CPU, while
native CUDA/Triton numerical execution and GPU-only FSDP initialization remain
hardware-gated. CPU validation does not substitute for those runtime checks.
