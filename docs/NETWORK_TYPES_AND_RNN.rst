Network Types and RNN Configuration
===================================

This page documents the current core-network options and recurrent configuration
used by ``dendritic_modeling``.

Supported Core Types
--------------------

Set ``model.core.type`` to one of:

- Feedforward E/I: ``dendritic_shunting``, ``dendritic_additive``, ``flat_shunting``, ``flat_additive``, ``dendritic_mlp``
- Feedforward baseline: ``mlp`` (or ``point_mlp``)
- Unified E/I recurrent-capable core: ``ei_unified`` (aliases: ``unified_ei``, ``ei_net``, ``unified_einet``)
- Baseline recurrent cores: ``gru``, ``lstm``, ``vanilla_rnn`` (alias: ``rnn``)

Feedforward Preservation
------------------------

Feedforward behavior is unchanged:

- If ``model.core.type`` is feedforward, recurrent modules are not instantiated.
- Data and forward passes stay on the standard non-recurrent path.

Unified E/I Core (Feedforward or Recurrent)
-------------------------------------------

Use ``ei_unified`` with ``model.core.unified_ei``:

.. code-block:: yaml

   model:
     task: classification
     core:
       type: ei_unified
       unified_ei:
         output_mode: last        # last | all | mean
         store_routing: false
         layers:
           - recurrent: true
             dt: 1.0
             excitatory:
               n_neurons: 64
               branch_factors: [3, 3]
               ff_excitatory_synapses: 12
               ff_inhibitory_synapses: 6
               rec_excitatory_synapses: 8
               rec_inhibitory_synapses: 4
               tau_base: 50.0
               tau_ratio: 3.0
             inhibitory:
               n_neurons: 16
               branch_factors: [1]
               ff_excitatory_synapses: 8
               ff_inhibitory_synapses: 4
               rec_excitatory_synapses: 4
               rec_inhibitory_synapses: 4

Notes:

- Set ``recurrent: false`` per layer for feedforward unified E/I layers.
- If ``level_taus`` is empty, per-level time constants are generated from
  ``tau_base`` and ``tau_ratio``.

Baseline RNN Cores
------------------

Baseline recurrent cores use the same ``core.type`` switch with optional
``baseline_rnn`` parameters:

.. code-block:: yaml

   model:
     task: classification
     core:
       type: gru                  # or lstm / vanilla_rnn
       baseline_rnn:
         hidden_dim: 128
         num_layers: 1
         dropout: 0.0
         bidirectional: false
         output_mode: last        # last | all | mean

Recurrent Training Strategy
---------------------------

For recurrent runs, set:

.. code-block:: yaml

   training:
     main:
       strategy: recurrent

Sequence Datasets
-----------------

The unified dataset loader supports sequence datasets:

- ``sequential_mnist``
- ``copy_task``
- ``adding_problem``
- ``variable_delay_dms``
- ``multi_frequency``
- ``lorenz``
- ``stringer_v1``

Output-Mode Conventions
-----------------------

- ``output_mode: last`` -> output shape ``[B, C]``
- ``output_mode: mean`` -> output shape ``[B, C]`` (temporal average)
- ``output_mode: all`` -> output shape ``[B, T, C]``

For ``output_mode: all``, targets must include the temporal axis.

Routing Analysis
----------------

Routing analysis can be enabled through analysis configuration:

.. code-block:: yaml

   analysis:
     routing_analysis:
       enabled: true
       training: false

Run Training
------------

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH \
   python -m dendritic_modeling.scripts.training.train_experiments \
     path/to/config.yaml --seed 0 --output_dir results/my_run
