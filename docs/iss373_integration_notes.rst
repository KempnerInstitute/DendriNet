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

Deferred experimental work
--------------------------

The ``finalized/`` directory on ``ryanconfig`` is a curated snapshot of
hardware-specific experiments rather than a self-contained package.  Its
custom CUDA and Triton gathers, fused activation autograd functions, and
architecture-specific kernel picker were therefore not imported in this safe
batch.  They require equivalence, gradient, memory, and performance checks on
each supported GPU architecture before becoming maintained runtime paths.

The profiling material was reviewed but also remains deferred.  On the review
baseline, short-capture control is retained by the recurrent trainer while the
standard trainer has moved to a different NVTX-only annotation path.  The
associated jobs and setup instructions also encode cluster-specific CUDA and
module assumptions.  A later profiling integration should first unify the
trainer behavior, keep all annotations opt-in, expose one documented CLI, and
add CPU no-op plus GPU trace smoke tests.

Validation expectations
-----------------------

Changes on this branch are checked with formatting and lint hooks, YAML and
typed-config loading, focused model and sweep tests, FSDP CPU-side unit tests,
and the broader test suite.  GPU-only FSDP initialization and native-kernel
performance still require a supported multi-GPU allocation; CPU validation
does not substitute for that final runtime check.
