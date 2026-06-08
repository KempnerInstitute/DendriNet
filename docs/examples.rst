Runnable Config Examples
========================

These YAML files are copy-paste starting points for local runs. The first four
are the primary vision examples; the recurrent examples exercise the newer
cell-type, synapse-type, structured-connectivity, and spiking components.

Feature Map
-----------

Use this table when choosing the smallest example that exercises a newer
implementation path:

.. list-table::
   :header-rows: 1
   :widths: 22 31 31 16

   * - Feature
     - Main config keys
     - Example config
     - Runner
   * - Named recurrent cell types
     - ``model.core.type: population_graph``,
       ``population_graph.cell_types``,
       ``population_graph.connections``
     - ``configs/examples/recurrent_population_graph_pyr_pv_sst.yaml``
     - ``train_experiments.py``
   * - Multi-population feedforward E/I layer
     - ``model.core.type: multi_population_ei``,
       ``multi_population.layers[].populations``,
       ``multi_population.layers[].connections``
     - ``configs/examples/multi_population_ei_feedforward.yaml``
     - ``train_experiments.py``
   * - Multi-population independent E/I transfer streams
     - ``multi_population.use_transfer: true``,
       ``transfer_params.independent_pathways: true``,
       ``connections[].source: input_e``/``input_i``
     - ``configs/examples/multi_population_ei_independent_transfer.yaml``
     - ``train_experiments.py``
   * - Multi-population two-layer E/I stack
     - ``multi_population.layers`` with two E/I population layers,
       per-layer ``connections``
     - ``configs/examples/multi_population_ei_two_layer.yaml``
     - ``train_experiments.py``
   * - Legacy feedforward configs in population format
     - ``model.core.type: multi_population_ei`` plus
       ``use_transfer: true`` and old E/I sizes/synapse counts
     - ``configs/examples/standard_cifar10_multi_population_ei.yaml``
     - ``train_experiments.py``
   * - Dense-to-sparse in population format
     - ``topk_type: dense_to_sparse`` and
       ``dense_to_sparse_*`` population fields
     - ``configs/examples/standard_cifar10_dense_to_sparse_multi_population_ei.yaml``
     - ``train_experiments.py``
   * - AMPA/NMDA/GABA-style synapse types
     - ``synapse_types.enabled``, ``excitatory``, ``inhibitory``,
       ``tau_rise``, ``tau_decay``, ``reversal_potential``
     - ``configs/examples/recurrent_synapse_types_ampa_nmda_gaba.yaml``
     - ``train_experiments.py``
   * - Structured TopK/DeepST masks
     - ``model.core.connectivity.structured.enabled``,
       ``pathways.ee``/``ie``/``rec_ee``/``rec_ie``
     - ``configs/examples/recurrent_structured_connectivity.yaml``
     - ``train_experiments.py``
   * - Spiking soma
     - ``population.dynamics_mode: spike`` or
       ``model.core.dynamics.mode: spike``
     - ``configs/examples/recurrent_spiking_population_graph.yaml``
     - ``train_experiments.py``
   * - Soma-to-dendrite feedback
     - ``soma_feedback_enabled`` or ``model.core.soma_feedback.enabled``
     - ``configs/examples/recurrent_soma_feedback.yaml``
     - ``train_experiments.py``
   * - Dendritic transformer FFN
     - ``model.transformer_replacement``,
       ``model.core.type: dendritic_ffn``,
       ``replacement_kwargs.kind: dendritic_ffn``
     - ``configs/examples/transformer_replacement_synthetic_distillation.yaml``
     - ``train_transformer_replacement``
   * - Transformer EI stack replacement
     - ``replacement_kwargs.kind: ei_stack`` plus EI stack sizing fields such
       as ``excitatory_cells`` and ``inhibitory_cells``
     - ``configs/benchmarks/transformer_replacement/generated/olmo2_1b_ei_stack_all_mlp/olmo2_1b_ei_stack_e200_i50.yaml``
     - ``train_transformer_replacement``

All scripts under ``src/dendritic_modeling/scripts`` are package-owned entry
points and should be tracked. Local scratch scripts in the repository root
``scripts/`` directory are intentionally ignored.

MNIST Single Layer
------------------

Original single-layer MNIST feedforward example:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_mnist.yaml

Multi-Population E/I Feedforward
--------------------------------

The same feedforward E/I computation can be expressed with named populations:
``input -> i``, ``input -> e``, and same-step ``i -> e`` inhibition. This is
the smallest example of the new layer schema; adding PV/SST/VIP or distinct
morphologies is done by adding population blocks and directional edges.

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/multi_population_ei_feedforward.yaml

Multi-Population E/I with Independent Transfer
----------------------------------------------

To reproduce the legacy independent-pathway ``TransferLayer`` split, use
``input_e`` and ``input_i`` as explicit first-layer sources:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/multi_population_ei_independent_transfer.yaml

Multi-Population Two-Layer E/I
------------------------------

The old two-layer E/I shape, ``128E/32I -> 64E/16I``, can be written as two
multi-population layers. Layer 0 uses ``input -> i0`` and ``input -> e0``.
Layer 1 then addresses the earlier populations explicitly, for example
``layer0.e0 -> i1``, ``layer0.e0 -> e1``, and ``layer0.i0 -> e1``, before
local same-step ``i1 -> e1`` inhibition:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/multi_population_ei_two_layer.yaml

Old-Format Population Mirrors
-----------------------------

The following examples mirror the older ``model.core.type: einet`` configs
with the newer named-population schema. They are useful for migration and
regression checks because the old morphology, transfer mode, inhibitory-input
convention, synapse counts, decoder, and training settings are kept aligned:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_mnist_multi_population_ei.yaml

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_cifar10_multi_population_ei.yaml

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/cifar10_128_333_40e10i_multi_population_ei.yaml

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/cifar10_two_layer_multi_population_ei.yaml

CIFAR-10 Single Layer
---------------------

CIFAR-10 with one 128E/32I dendritic layer, ``[3, 3, 3]`` morphology, and
40 excitatory / 10 inhibitory synapses per excitatory branch:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/cifar10_128_333_40e10i.yaml

CIFAR-10 Dense-to-Sparse TopK
-----------------------------

CIFAR-10 with the standard single-layer DendriNet, but
``model.core.sparsity.type: dense_to_sparse`` so allowed synapses receive
gradient early and are annealed down to the configured TopK budget:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_cifar10_dense_to_sparse.yaml

The same dense-to-sparse selector can be used from the population schema:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_cifar10_dense_to_sparse_multi_population_ei.yaml

CIFAR-10 Two Layers
-------------------

CIFAR-10 with two dendritic EI layers, ``128E -> 64E`` and ``32I -> 16I``:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/cifar10_two_layer.yaml

AlexNet 500E Replacement
------------------------

AlexNet ``fc7`` replacement with a 500E-only dendritic tensor-map core. For
real training, set ``data.base_dir`` or ``IMAGENET_LOCAL_DIR`` to a local
ImageNet root; without ImageNet, the loader falls back to small fake data for
smoke testing only.

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/alexnet_fc7_dendritic_500.yaml

Recurrent Cell Types
--------------------

Named pyramidal, PV, and SST populations on a compact synthetic
context-dependent decision task:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/recurrent_population_graph_pyr_pv_sst.yaml

Synapse Types
-------------

Population graph with AMPA/NMDA and GABA-A/GABA-B conductance traces, including
rise/decay kinetics, NMDA magnesium block, and an aggregate short-term
plasticity example:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/recurrent_synapse_types_ampa_nmda_gaba.yaml

Structured Connectivity
-----------------------

Recurrent EINet with fixed-indegree and Bernoulli topology masks on selected
feedforward and recurrent pathways:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/recurrent_structured_connectivity.yaml

Spiking Soma
------------

Population graph with opt-in LIF soma dynamics, a rate-decoded spike trace, and
``analysis.spike_train`` summaries for raster, firing-rate, Fano, and ISI
diagnostics:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/recurrent_spiking_population_graph.yaml

DeepSeek-V3 MLP Replacement Template
------------------------------------

Transformer replacement uses the same target-plus-core structure as the
AlexNet examples, but is applied to a Hugging Face model object rather than the
vision training script. The template below replaces decoder-block MLPs
``0, 1, 2`` with a token-wise dendritic FFN using ``indexed_dynamic`` sparse
candidate pools:

.. code-block:: text

   configs/examples/deepseek_v3_mlp_dendritic_replacement.yaml

Transformer Replacement Distillation Smoke Test
-----------------------------------------------

This example trains a small dendritic transformer-MLP replacement against a
synthetic teacher MLP.  It is the fastest check that the replacement and the
new ``training.transformer_replacement`` block are wired correctly before
loading a large Hugging Face model:

.. code-block:: text

   configs/examples/transformer_replacement_synthetic_distillation.yaml

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python -m \
     dendritic_modeling.scripts.training.train_transformer_replacement \
     configs/examples/transformer_replacement_synthetic_distillation.yaml

Transformer Replacement Benchmarks
----------------------------------

Benchmark configs live under
``configs/benchmarks/transformer_replacement/``.  Each job first trains the
dendritic replacement and then benchmarks the original teacher block against
the trained replacement on held-out hidden states:

.. code-block:: text

   configs/benchmarks/transformer_replacement/synthetic_linear_projection.yaml
   configs/benchmarks/transformer_replacement/synthetic_gelu_mlp.yaml
   configs/benchmarks/transformer_replacement/synthetic_gated_silu_mlp.yaml

Slurm launch scripts are in
``jobs/transformer_replacement_benchmarks/``.  The aggregate output is written
to ``results/transformer_replacement_benchmarks/summary.md`` and
``summary.csv``.

Component Coverage
------------------

.. list-table::
   :header-rows: 1
   :widths: 38 12 12 12 12 14

   * - Config
     - Cell types
     - Synapse types
     - Structured masks
     - Spikes
     - Soma feedback
   * - ``recurrent_population_graph_pyr_pv_sst.yaml``
     - yes
     - no
     - source-target
     - no
     - no
   * - ``multi_population_ei_feedforward.yaml``
     - yes
     - no
     - directional same-step
     - no
     - no
   * - ``multi_population_ei_independent_transfer.yaml``
     - yes
     - no
     - transfer split + directional same-step
     - no
     - no
   * - ``multi_population_ei_two_layer.yaml``
     - yes
     - no
     - two-layer directional same-step
     - no
     - no
   * - ``standard_cifar10_multi_population_ei.yaml``
     - yes
     - no
     - old E/I mirror
     - no
     - no
   * - ``standard_cifar10_dense_to_sparse_multi_population_ei.yaml``
     - yes
     - no
     - old E/I mirror + dense-to-sparse
     - no
     - no
   * - ``recurrent_synapse_types_ampa_nmda_gaba.yaml``
     - yes
     - yes
     - source-target
     - no
     - no
   * - ``recurrent_structured_connectivity.yaml``
     - legacy E/I
     - no
     - yes
     - no
     - no
   * - ``recurrent_spiking_population_graph.yaml``
     - yes
     - no
     - source-target
     - yes
     - no
   * - ``recurrent_soma_feedback.yaml``
     - yes
     - no
     - source-target
     - no
     - yes
