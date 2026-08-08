Integrated Runtime Features
===========================

Vision backbone replacement
---------------------------

Vision replacement can split a supported pretrained backbone into an encoder
prefix, a configured core, and a decoder suffix. AlexNet configurations cover
flat ``fc6``/``fc7`` replacement, tensor-map dendritic cores, and spatial
``conv5`` replacement. The configured core uses the same architecture,
connectivity, sparsity, reactivation, and initialization fields as a standalone
model.

``vision_replacement.core_source: configured`` constructs the requested core.
``core_source: pretrained_span`` retains the original modules in the selected
span. Target and omitted module names are validated against the backbone split,
and the replacement output contract is checked against the suffix input.

See :doc:`configuration_guide` and :doc:`network_architectures` for supported
core types and spatial contracts.

Compact indexed connectivity
----------------------------

Indexed synapses store only active presynaptic indices and weights. They support
compact int32 topology buffers, bounded gather workspaces, deterministic
topology regeneration, recurrent transformed-weight caching, and compact
resource accounting. The same fields are translated through structured E/I,
population-network, recurrent, transformer-replacement, and vision-replacement
configuration paths.

See :doc:`performance_kernels` for execution backends, dtype behavior,
recurrent guidance, activation memory controls, and YAML examples.

Distributed construction
------------------------

FSDP can construct contact-heavy models on CPU and wrap indexed synapse units
before synchronizing and moving their states to the GPU:

.. code-block:: yaml

   distributed:
     mode: fsdp
     fsdp:
       auto_wrap_policy: indexed_synapses
       cpu_init: true

``cpu_init`` is opt-in. It is intended for models whose compact connectivity is
still too large to materialize independently on every GPU before wrapping.
With ``indexed_persistent_indices: false``, an explicit topology seed and
synchronized initialization are required so every rank reconstructs the same
fixed graph.
