Runnable Config Examples
========================

These YAML files are copy-paste starting points for local runs. The first four
are the primary vision examples; the recurrent examples exercise the newer
population, synapse-type, structured-connectivity, and spiking components.

Recommended Example Ladder
--------------------------

The examples are organized so a user can move from the original E/I model to
the fully compositional population-network notation without changing more
than one idea at a time.

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Step
     - Config
     - What it teaches
   * - 1. Original feedforward E/I
     - ``configs/examples/standard_mnist.yaml``
     - Legacy-compatible shunting MNIST with the current recommended
       ``occupancy_quantile`` gate initialization.
   * - 2. Same model, canonical notation
     - ``configs/examples/standard_mnist_population_network.yaml``
     - The same E/I computation written as named populations and explicit
       directed connections.
   * - 3. Multiple populations in one feedforward layer
     - ``configs/examples/population_network_heterogeneous_feedforward.yaml``
     - PV/SST-like inhibitory populations and multiple excitatory morphologies
       in one layer.
   * - 4. Explicit multilayer routing
     - ``configs/examples/population_network_two_layer.yaml``
     - Cross-layer sources such as ``layer0.e0`` and local same-step
       inhibition in each layer.
   * - 5. Recurrent microcircuit
     - ``configs/examples/population_network_recurrent_pyr_pv_sst.yaml``
     - Delayed recurrent edges, connection probabilities, and recurrent
       training.
   * - 6. Biophysics extensions
     - ``population_network_synapse_types_ampa_nmda_gaba.yaml``,
       ``population_network_spiking.yaml``, ``population_network_soma_feedback.yaml``
     - Typed conductance traces, LIF soma output, and previous-timestep
       soma-to-dendrite feedback.
   * - 7. Developmental sparsity
     - ``configs/examples/standard_cifar10_dense_to_sparse_population_network.yaml``
     - Dense-to-sparse TopK warmup and pruning in the population schema.

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
   * - Canonical recurrent population network
     - ``model.core.type: population_network``,
       ``population_network.layers[].populations``,
       ``population_network.layers[].connections``
     - ``configs/examples/population_network_recurrent_pyr_pv_sst.yaml``
     - ``train_experiments.py``
   * - Canonical feedforward population network
     - ``model.core.type: population_network``,
       ``population_network.layers[].populations``,
       ``population_network.layers[].connections``
     - ``configs/examples/population_network_feedforward.yaml``
     - ``train_experiments.py``
   * - Heterogeneous feedforward populations
     - ``population_network.layers[].populations[].population`` with
       per-population morphology and dynamics overrides
     - ``configs/examples/population_network_heterogeneous_feedforward.yaml``
     - ``train_experiments.py``
   * - Independent E/I transfer streams
     - ``population_network.use_transfer: true``,
       ``transfer_params.independent_pathways: true``,
       ``connections[].source: input_e``/``input_i``
     - ``configs/examples/population_network_independent_transfer.yaml``
     - ``train_experiments.py``
   * - Two-layer population network
     - ``population_network.layers`` with two E/I population layers,
       per-layer ``connections``
     - ``configs/examples/population_network_two_layer.yaml``
     - ``train_experiments.py``
   * - Legacy feedforward configs in population format
     - ``model.core.type: population_network`` plus
       ``use_transfer: true`` and old E/I sizes/synapse counts
     - ``configs/examples/standard_cifar10_population_network.yaml``
     - ``train_experiments.py``
   * - Dense-to-sparse in population format
     - ``topk_type: dense_to_sparse`` and
       ``dense_to_sparse_*`` population fields
     - ``configs/examples/standard_cifar10_dense_to_sparse_population_network.yaml``
     - ``train_experiments.py``
   * - Normalized additive in population format
     - ``use_shunting: false`` plus
       ``use_additive_normalization: true`` in ``population_defaults``
     - ``configs/examples/standard_mnist_normalized_additive_population_network.yaml``
     - ``train_experiments.py``
   * - AMPA/NMDA/GABA-style synapse types
     - ``population_network.layers[].population_defaults.synapse_types`` with
       ``excitatory``/``inhibitory`` receptor lists
     - ``configs/examples/population_network_synapse_types_ampa_nmda_gaba.yaml``
     - ``train_experiments.py``
   * - Taus and delayed recurrent edges
     - ``layers[].dt``, ``population_defaults.tau_base``,
       ``population_defaults.tau_ratio``, ``connections[].timing: delayed``
     - ``configs/examples/population_network_taus_and_delays.yaml``
     - ``train_experiments.py``
   * - Multilayer recurrent population network
     - Cross-layer sources such as ``layer0.e0`` plus local and delayed edges
     - ``configs/examples/population_network_multilayer_recurrent.yaml``
     - ``train_experiments.py``
   * - Structured TopK/DeepST masks
     - ``model.core.connectivity.structured.enabled``,
       ``pathways.ee``/``ie``/``rec_ee``/``rec_ie``
     - ``configs/examples/recurrent_structured_connectivity.yaml``
     - ``train_experiments.py``
   * - Spiking soma
     - ``population_network.layers[].population_defaults.dynamics_mode: spike``
     - ``configs/examples/population_network_spiking.yaml``
     - ``train_experiments.py``
   * - Spiking multilayer recurrent network
     - ``dynamics_mode: spike`` on recurrent population-network layers
     - ``configs/examples/population_network_spiking_multilayer_recurrent.yaml``
     - ``train_experiments.py``
   * - Soma-to-dendrite feedback
     - ``population_network.layers[].population_defaults.soma_feedback_enabled``
     - ``configs/examples/population_network_soma_feedback.yaml``
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
     - ``jobs/transformer_replacement_benchmarks/configs/generated/olmo2_1b_ei_stack_all_mlp/olmo2_1b_ei_stack_e200_i50.yaml``
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

Population Network Feedforward
------------------------------

The same feedforward E/I computation can be expressed with named populations:
``input -> i``, ``input -> e``, and same-step ``i -> e`` inhibition. This is
the smallest example of the canonical layer schema; adding PV/SST/VIP or distinct
morphologies is done by adding population blocks and directional edges.

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_feedforward.yaml

Population Network Heterogeneous Cell Types
-------------------------------------------

Multiple E and I populations can live in the same feedforward layer. The
example below combines PV-like and SST-like inhibitory populations with deep
and flat excitatory populations, each with its own morphology and
``PopulationConfig`` overrides:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_heterogeneous_feedforward.yaml

Population Network Recurrent
----------------------------

The same schema becomes recurrent by adding ``timing: delayed`` connections on
a recurrent layer. Same-step input edges drive the current timestep; delayed
population edges read previous-timestep outputs.

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_recurrent_pyr_pv_sst.yaml

Population Network with Independent Transfer
----------------------------------------------

To reproduce the legacy independent-pathway ``TransferLayer`` split, use
``input_e`` and ``input_i`` as explicit first-layer sources:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_independent_transfer.yaml

Population Network Two-Layer E/I
--------------------------------

The old two-layer E/I shape, ``128E/32I -> 64E/16I``, can be written as two
population-network layers. Layer 0 uses ``input -> i0`` and ``input -> e0``.
Layer 1 then addresses the earlier populations explicitly, for example
``layer0.e0 -> i1``, ``layer0.e0 -> e1``, and ``layer0.i0 -> e1``, before
local same-step ``i1 -> e1`` inhibition:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_two_layer.yaml

Old-Format Population Mirrors
-----------------------------

The following examples mirror the older ``model.core.type: einet`` configs
with the newer named-population schema. They are useful for migration and
regression checks because the old morphology, transfer mode, inhibitory-input
convention, synapse counts, decoder, and training settings are kept aligned:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_mnist_population_network.yaml

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_cifar10_population_network.yaml

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/cifar10_128_333_40e10i_population_network.yaml

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/cifar10_two_layer_population_network.yaml

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
     configs/examples/standard_cifar10_dense_to_sparse_population_network.yaml

Normalized Additive Population Network
--------------------------------------

The normalized additive MNIST recipe can also be written directly as a
population network. This is the population-format additive counterpart to the
standard shunting mirrors:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_mnist_normalized_additive_population_network.yaml

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
ImageNet root. The example uses pretrained ImageNet preprocessing and disables
fake-data fallback; for loader smoke tests without the real dataset, set
``data.dataset_params.imagenet.allow_fake_data: true`` explicitly.

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/alexnet_fc7_dendritic_500.yaml

Recurrent Population Networks
-----------------------------

Use ``population_network_recurrent_pyr_pv_sst.yaml`` for named recurrent
E/I cell populations. It uses the same ``population_network.layers[]`` and
``connections`` notation as feedforward population examples:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_recurrent_pyr_pv_sst.yaml

Taus and Delayed Edges
----------------------

Temporal constants are explicit. ``layers[].dt`` is the update step,
``tau_base``/``tau_ratio`` or ``level_taus`` define dendritic trace decays, and
``timing: delayed`` connections read the previous timestep. Receptor-style
``synapse_types`` add their own ``tau_rise``/``tau_decay`` traces:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_taus_and_delays.yaml

Multilayer Recurrent Population Network
---------------------------------------

Multiple recurrent population layers can be stacked. Later layers can consume
any earlier population by qualified source name, for example ``layer0.e0`` or
``layer0.pv0``. Local same-step edges remain directional, while recurrent
feedback uses ``timing: delayed``:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_multilayer_recurrent.yaml

Synapse Types
-------------

Population network with AMPA/NMDA and GABA-A/GABA-B conductance traces,
including rise/decay kinetics, NMDA magnesium block, and an aggregate
short-term plasticity example:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_synapse_types_ampa_nmda_gaba.yaml

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

Population network with opt-in LIF soma dynamics, a rate-decoded spike trace, and
``analysis.spike_train`` summaries for raster, firing-rate, Fano, and ISI
diagnostics:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_spiking.yaml

The same population-network wiring also works with spiking soma dynamics in a
multilayer recurrent network:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_spiking_multilayer_recurrent.yaml

Soma Feedback
-------------

Population network with delayed soma-to-dendrite feedback:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_soma_feedback.yaml

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
``jobs/transformer_replacement_benchmarks/configs/``.  Each job first trains the
dendritic replacement and then benchmarks the original teacher block against
the trained replacement on held-out hidden states:

.. code-block:: text

   jobs/transformer_replacement_benchmarks/configs/synthetic_linear_projection.yaml
   jobs/transformer_replacement_benchmarks/configs/synthetic_gelu_mlp.yaml
   jobs/transformer_replacement_benchmarks/configs/synthetic_gated_silu_mlp.yaml

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
     - Populations
     - Synapse types
     - Structured masks
     - Spikes
     - Soma feedback
   * - ``population_network_feedforward.yaml``
     - yes
     - no
     - directional same-step
     - no
     - no
   * - ``population_network_heterogeneous_feedforward.yaml``
     - yes
     - no
     - heterogeneous same-step DAG
     - no
     - no
   * - ``population_network_recurrent_pyr_pv_sst.yaml``
     - yes
     - no
     - same-step + delayed recurrent
     - no
     - no
   * - ``population_network_independent_transfer.yaml``
     - yes
     - no
     - transfer split + directional same-step
     - no
     - no
   * - ``population_network_two_layer.yaml``
     - yes
     - no
     - two-layer directional same-step
     - no
     - no
   * - ``standard_cifar10_population_network.yaml``
     - yes
     - no
     - old E/I mirror
     - no
     - no
   * - ``standard_cifar10_dense_to_sparse_population_network.yaml``
     - yes
     - no
     - old E/I mirror + dense-to-sparse
     - no
     - no
   * - ``standard_mnist_normalized_additive_population_network.yaml``
     - yes
     - no
     - old E/I mirror + normalized additive
     - no
     - no
   * - ``population_network_synapse_types_ampa_nmda_gaba.yaml``
     - yes
     - yes
     - same-step + delayed recurrent
     - no
     - no
   * - ``population_network_taus_and_delays.yaml``
     - yes
     - yes
     - delayed recurrent + tau controls
     - no
     - no
   * - ``population_network_multilayer_recurrent.yaml``
     - yes
     - no
     - multilayer same-step + delayed
     - no
     - no
   * - ``recurrent_structured_connectivity.yaml``
     - legacy E/I
     - no
     - yes
     - no
     - no
   * - ``population_network_spiking.yaml``
     - yes
     - no
     - same-step + delayed recurrent
     - yes
     - no
   * - ``population_network_spiking_multilayer_recurrent.yaml``
     - yes
     - no
     - multilayer same-step + delayed
     - yes
     - no
   * - ``population_network_soma_feedback.yaml``
     - yes
     - no
     - same-step + delayed recurrent
     - no
     - yes
