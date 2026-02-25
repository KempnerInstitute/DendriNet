Network Types and RNN Configuration
===================================

This page documents how to configure all supported core network families:

- Feedforward dendritic EI models (existing path)
- Feedforward MLP baselines (existing path)
- Recurrent dendritic models (new path)
- Recurrent baselines (vanilla/GRU/LSTM)

Non-RNN Preservation and Overhead
---------------------------------

Non-recurrent behavior is preserved:

- If ``model.core.type`` is a feedforward type (for example ``dendritic_shunting``), the recurrent factory path is never used.
- ``RecurrentConfig.enabled`` defaults to ``false`` and is inert unless a recurrent core type is selected.
- ``BaseModel`` caches whether the core is recurrent at initialization, so non-RNN forwards stay on the original
  ``encoder -> core -> decoder`` path without recurrent-specific tensor reshaping logic.

In short: feedforward models keep the same functional path and no recurrent unrolling cost.

Core Type Matrix
----------------

Use ``model.core.type`` to select the core architecture:

- ``dendritic_shunting``: feedforward EI network with shunting integration
- ``dendritic_additive``: feedforward EI network with additive integration
- ``flat_shunting``: flattened dendrites with shunting
- ``flat_additive``: flattened dendrites with additive integration
- ``dendritic_mlp``: signed-weight EI MLP variant
- ``mlp`` (or ``point_mlp``): point-neuron baseline
- ``dendritic_rnn`` (aliases: ``dendrnn``, ``vanilla_rnn``): recurrent family

Recurrent Core Configuration
----------------------------

Recurrent models use:

.. code-block:: yaml

   model:
     core:
       type: "dendritic_rnn"
       recurrent:
         enabled: true
         cell_type: "dendritic"   # "dendritic" | "gru" | "lstm"
         n_excitatory: 64
         n_inhibitory: 16
         num_layers: 1
         branches:
           - tau: 10.0
             ff_synapses: 12
             rec_synapses: 0
             ie_synapses: 6
             compartment: "topk"  # "topk" | "dendritic_branch" | "dendrinet" | "linear"
             activation: "relu"
           - tau: 50.0
             ff_synapses: 6
             rec_synapses: 6
             ie_synapses: 4
             compartment: "topk"
           - tau: 200.0
             ff_synapses: 0
             rec_synapses: 10
             ie_synapses: 4
             compartment: "topk"
         soma_mode: "shunting"     # "shunting" | "additive" | "sum"
         output_mode: "last"       # "last" | "all" | "mean"
         dt: 1.0
         g_leak: 1.0
         input_projection_layers: [64]
         dropout: 0.0
         learn_taus: false
         store_routing: false

Training strategy:

.. code-block:: yaml

   training:
     main:
       strategy: "recurrent"

Feedforward EI Example (Unchanged Path)
---------------------------------------

.. code-block:: yaml

   model:
     core:
       type: "dendritic_shunting"
       # existing architecture/connectivity/morphology sections unchanged

   training:
     main:
       strategy: "standard"

Vanilla RNN Emulation
---------------------

Single branch + linear compartment + no inhibitory population:

.. code-block:: yaml

   model:
     core:
       type: "dendritic_rnn"
       recurrent:
         enabled: true
         cell_type: "dendritic"
         n_excitatory: 80
         n_inhibitory: 0
         branches:
           - tau: 50.0
             ff_synapses: 80
             rec_synapses: 80
             ie_synapses: 0
             compartment: "linear"
             activation: "tanh"
         soma_mode: "sum"
         output_mode: "last"

GRU / LSTM Baselines
--------------------

.. code-block:: yaml

   model:
     core:
       type: "dendritic_rnn"
       recurrent:
         enabled: true
         cell_type: "gru"   # or "lstm"
         n_excitatory: 128

When ``cell_type`` is ``gru`` or ``lstm``, the branch system is bypassed.

DendriNet Compartment in Recurrent Branches
-------------------------------------------

You can reuse existing ``DendriNet`` modules inside recurrent branches:

.. code-block:: yaml

   model:
     core:
       type: "dendritic_rnn"
       recurrent:
         enabled: true
         branches:
           - tau: 20.0
             ff_synapses: 8
             rec_synapses: 8
             compartment: "dendrinet"
             dendrinet_branch_factors: [2, 2]

DendriticBranch Compartment (Per-Branch Shunting)
-------------------------------------------------

``compartment: "dendritic_branch"`` reuses ``DendriticBranchLayer`` inside each
recurrent branch, so branch-level shunting stays consistent with feedforward
dendritic computation.

.. code-block:: yaml

   model:
     core:
       type: "dendritic_rnn"
       recurrent:
         enabled: true
         n_excitatory: 64
         n_inhibitory: 16
         branches:
           - tau: 50.0
             compartment: "dendritic_branch"
             ff_synapses: 8
             rec_synapses: 8
             ie_synapses: 4
             rec_i_synapses: 2         # recurrent inhibitory synapses
             use_shunting: true
             reactivate: false
             reactivation_type: "param_tanh"

Notes:

- ``rec_i_synapses`` is only valid for ``compartment: "dendritic_branch"``.
- If ``n_inhibitory: 0``, both ``ie_synapses`` and ``rec_i_synapses`` are forced to 0 at runtime.
- Feedforward models are unaffected because these fields are only read by recurrent cores.

Compatibility Guarantees
------------------------

Old feedforward configs and non-RNN runs remain valid and unchanged:

- Feedforward cores (for example ``dendritic_shunting``) do not instantiate
  recurrent modules.
- Recurrent-only config fields are ignored unless ``model.core.type`` is
  ``dendritic_rnn``/``dendrnn``/``vanilla_rnn`` with ``recurrent.enabled: true``.
- ``DendriticBranchLayer`` adds no recurrent synapse layers unless recurrent
  dimensions/synapse counts are explicitly provided.

Sequence Datasets
-----------------

The unified dataset loader supports sequence names:

- ``sequential_mnist``
- ``copy_task``
- ``adding_problem``
- ``variable_delay_dms``
- ``multi_frequency``
- ``lorenz``

``output_mode`` and target shape expectations:

- ``output_mode: "last"`` -> model output ``[B, C]`` (many-to-one)
- ``output_mode: "all"`` -> model output ``[B, T, C]`` (many-to-many targets)
- ``output_mode: "mean"`` -> temporal mean reduced to ``[B, C]``

Routing Analysis (alpha)
------------------------

Enable routing analysis for recurrent runs:

.. code-block:: yaml

   analysis:
     routing:
       enabled: true
       training: false

At analysis time, routing storage is enabled temporarily and restored afterward.

Training and Analysis Tools
---------------------------

Train a recurrent model from a YAML config:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH \
   python -m dendritic_modeling.scripts.training.train_experiments \
     path/to/config.yaml --seed 0 --output_dir results/my_run

Post-hoc analysis tools for trained models:

.. code-block:: bash

   # Branch routing analysis (alpha conditioned on task variable)
   python -m dendritic_modeling.scripts.analysis.analyze_routing_by_condition \
     --model_path <model.pt> --config_path <config.yaml> --output_dir <out>

   # Branch lesion analysis (causal role of branches)
   python -m dendritic_modeling.scripts.analysis.lesion_analysis \
     --model_path <model.pt> --config_path <config.yaml> --output_dir <out>

   # Out-of-distribution delay evaluation
   python -m dendritic_modeling.scripts.analysis.evaluate_ood_delay \
     --model_path <model.pt> --config_path <config.yaml> --output_dir <out>

   # Jacobian eigenvalue and gradient stability analysis
   python -m dendritic_modeling.scripts.analysis.stability_analysis \
     --results_dir <results_dir> --output_dir <out>
