.. _reference-config:

Annotated reference configurations
==================================

Two complete configurations in ``configs/`` carry a comment on every option:
its type, the values it accepts and what it does. The values in the files are
a working example, not the library defaults. Where a value differs from its
default and the difference matters, the comment gives the default; an option
that a file leaves out takes its default. :doc:`configuration_guide` explains
how configs are loaded and checked and links the pages that document each
option in full.

Single-process configuration
----------------------------

``configs/config_exp.yaml`` trains a one-layer population network on CIFAR-10
in one process: an inhibitory population feeds an excitatory population whose
neurons have a three-level dendritic tree. Besides the options this run uses,
the file lists the options of the other training strategies, the local
learning rules and most analyzers, so it can serve as a starting point. Check
it without training:

.. code-block:: bash

   dendritic-train configs/config_exp.yaml --validate-only

.. literalinclude:: ../configs/config_exp.yaml
   :language: yaml
   :linenos:

FSDP configuration
------------------

``configs/config_exp_fsdp.yaml`` trains a population network with indexed
synapses with FullyShardedDataParallel (``distributed.mode: fsdp``). Its
sections are in alphabetical order, as in a saved ``config.json``, and its
comments explain the settings that matter under FSDP. Launch one process per
GPU, here on a node with four GPUs:

.. code-block:: bash

   torchrun --nproc_per_node=4 \
       -m dendritic_modeling.scripts.training.train_experiments_fsdp \
       configs/config_exp_fsdp.yaml

:ref:`distributed-training` describes multi-node launches and the FSDP
options.

.. literalinclude:: ../configs/config_exp_fsdp.yaml
   :language: yaml
   :linenos:

Sweep template
--------------

``configs/sweep_config.yaml`` is the annotated template of a parameter sweep:
a ``base_config`` and the options to vary (see :ref:`sweeps`). Generate its
configs without submitting jobs:

.. code-block:: bash

   dendritic-sweep --config configs/sweep_config.yaml --generate-only --output-dir sweeps

.. literalinclude:: ../configs/sweep_config.yaml
   :language: yaml
   :linenos:

Other templates
---------------

- ``configs/templates/analysis_full_template.yaml`` lists the analyzer
  sections with their options, all disabled. Copy sections into an experiment
  config, or pass a copy of the file to ``dendritic-analyze-checkpoint
  --analysis-config`` (see :ref:`analyzer-catalogue`).
- ``configs/templates/indexed_runtime_template.yaml`` shows the
  indexed-synapse runtime settings of a structured E/I core and their names in
  population networks (see :ref:`indexed-backends`).
- :ref:`examples-map` lists the focused example configs in
  ``configs/examples/``.
