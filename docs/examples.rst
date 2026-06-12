Runnable Config Examples
========================

``configs/examples/`` is the curated public example set. New structured
network examples use one convention:

* ``model.core.type: population_network``
* ``model.core.population_network.layers[].populations``
* ``model.core.population_network.layers[].connections``

Older E/I aliases remain supported by the code for reproducibility, but they
are not duplicated in the public examples. Specialized ImageNet/AlexNet
configs live under ``configs/vision/``; transformer benchmark configs live
with their Slurm jobs under ``jobs/transformer_replacement_benchmarks/``.

Recommended Ladder
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Step
     - Config
     - What it teaches
   * - 1. Feedforward E/I
     - ``configs/examples/population_network_feedforward.yaml``
     - One explicit layer with ``input -> i``, ``input -> e``, and
       same-step ``i -> e`` inhibition.
   * - 2. Independent transfer streams
     - ``configs/examples/population_network_independent_transfer.yaml``
     - Legacy-style split input streams addressed as ``input_e`` and
       ``input_i`` while still using named populations.
   * - 3. Legacy input-mode equivalence
     - ``configs/examples/population_network_input_mode0_one_layer.yaml`` and
       ``configs/examples/population_network_input_mode1_one_layer.yaml``
     - Original ``input_mode=0`` inhibitory-cell construction versus
       ``input_mode=1`` direct inhibitory streams in the new schema.
   * - 4. CIFAR-10 training recipe
     - ``configs/examples/population_network_cifar10.yaml``
     - 128E/32I, ``[3, 3, 3]`` morphology, ``40E/10I`` synapses,
       ``occupancy_quantile`` gates, weight decay, and weight boosting.
   * - 5. Dense-to-sparse pruning
     - ``configs/examples/population_network_dense_to_sparse_cifar10.yaml``
     - Starts from dense allowed synapses and anneals to the TopK budget.
   * - 6. Indexed rewiring
     - ``configs/examples/population_network_indexed_rewire_mnist.yaml``
     - Memory-efficient sparse storage plus adaptive-quantile replacement of
       weak same-dendrite synapses.
   * - 7. Normalized additive control
     - ``configs/examples/population_network_normalized_additive_mnist.yaml``
     - Explicit normalized-additive model; raw additive uses
       ``use_additive_normalization: false``.
   * - 8. Heterogeneous cell types
     - ``configs/examples/population_network_heterogeneous_feedforward.yaml``
     - Multiple E and I populations in one layer with per-population
       morphology and parameter overrides.
   * - 9. Two-layer routing
     - ``configs/examples/population_network_two_layer.yaml``
     - Cross-layer sources such as ``layer0.e0`` plus local same-step
       inhibition in each layer.
   * - 10. Recurrent microcircuit
     - ``configs/examples/population_network_recurrent_pyr_pv_sst.yaml``
     - Pyr/PV/SST populations, probabilistic edges, and delayed recurrent
       timing.
   * - 11. Synapse types
     - ``configs/examples/population_network_synapse_types_ampa_nmda_gaba.yaml``
     - AMPA/NMDA/GABA mixtures, receptor time constants, NMDA Mg block, and
       short-term plasticity fields.
   * - 12. Spiking and feedback
     - ``configs/examples/population_network_spiking.yaml`` and
       ``configs/examples/population_network_soma_feedback.yaml``
     - LIF soma dynamics, spike readout traces, and delayed soma-to-dendrite
       feedback.

Feedforward Population Network
------------------------------

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_feedforward.yaml

The layer is a directed graph of named populations:

.. code-block:: yaml

   model:
     core:
       type: population_network
       population_network:
         layers:
           - name: layer0
             readout_population: e
             populations:
               - {name: i, polarity: inhibitory, n_neurons: 32, branch_factors: [1]}
               - {name: e, polarity: excitatory, n_neurons: 128, branch_factors: [4, 4]}
             connections:
               - {source: input, target: i}
               - {source: input, target: e}
               - {source: i, target: e}

CIFAR-10 and Developmental Sparsity
-----------------------------------

Use ``population_network_cifar10.yaml`` for the maintained CIFAR-10 E/I
training recipe:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_cifar10.yaml

Use ``population_network_dense_to_sparse_cifar10.yaml`` when testing learned
pruning from dense allowed synapses down to the final TopK budget:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_dense_to_sparse_cifar10.yaml

Use ``population_network_indexed_rewire_mnist.yaml`` for memory-efficient
indexed synapses with adaptive-quantile same-dendrite turnover. It stores only
``out_features x K`` synaptic weights and periodically replaces the weakest
configured quantile with newly sampled allowed inputs:

.. code-block:: yaml

   population_defaults:
     topk_type: indexed_rewire
     indexed_seed: 42
     indexed_rewire_frequency: 100
     indexed_rewire_quantile: 0.05
     indexed_rewire_until_step: null

Heterogeneous and Multilayer Networks
-------------------------------------

``population_network_heterogeneous_feedforward.yaml`` shows multiple E/I cell
types in one layer. Each population can override morphology, shunting/additive
mode, synapse budgets, receptor mixtures, spiking settings, and feedback
settings through its nested ``population:`` block.

``population_network_two_layer.yaml`` shows how later layers consume earlier
population outputs by qualified names such as ``layer0.e0`` and ``layer0.i0``.

Legacy Input-Mode Equivalents
-----------------------------

These examples show the original E/I feedforward conventions written in the
new population-network notation:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Config
     - Meaning
   * - ``population_network_input_mode0_one_layer.yaml``
     - One-layer ``input_mode=0``: explicit inhibitory cells
       ``input_e -> i0 -> e0``.
   * - ``population_network_input_mode1_one_layer.yaml``
     - One-layer ``input_mode=1``: no explicit I cells; ``input_i`` drives the
       E population's inhibitory dendritic pathway.
   * - ``population_network_input_mode0_two_layer.yaml``
     - Two-layer version with an inhibitory population in each layer.
   * - ``population_network_input_mode1_two_layer.yaml``
     - Two-layer version with direct inhibitory pathways into each E layer.

Recurrent, Taus, Spiking, and Feedback
--------------------------------------

Use these examples for temporal models:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Config
     - Feature
   * - ``population_network_recurrent_pyr_pv_sst.yaml``
     - Single recurrent Pyr/PV/SST microcircuit with delayed edges.
   * - ``population_network_multilayer_recurrent.yaml``
     - Multiple recurrent population layers with cross-layer routing.
   * - ``population_network_taus_and_delays.yaml``
     - ``layers[].dt``, dendritic ``tau_*`` controls, and delayed edges.
   * - ``population_network_spiking.yaml``
     - LIF soma dynamics and spike-trace readout.
   * - ``population_network_spiking_multilayer_recurrent.yaml``
     - Spiking dynamics in a stacked recurrent population network.
   * - ``population_network_soma_feedback.yaml``
     - Previous-timestep soma-to-dendrite feedback.

Local Learning and Multi-stage Training
---------------------------------------

Training strategies use the same population-network model schema:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_local_learning_mnist.yaml

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/population_network_multistage_cifar10.yaml

Transformer Replacement
-----------------------

The small synthetic transformer replacement smoke test remains in
``configs/examples`` because it is fast and self-contained:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python -m \
     dendritic_modeling.scripts.training.train_transformer_replacement \
     configs/examples/transformer_replacement_synthetic_distillation.yaml

Large Hugging Face replacement configs are operational benchmark configs and
live under ``jobs/transformer_replacement_benchmarks/configs/``.

Vision Backbone Replacement
---------------------------

Vision replacement configs live under ``configs/vision/`` because they are
larger operational examples rather than small public smoke tests. They use
``model.pretrained_replacement`` to split a torchvision backbone into a frozen
or trainable prefix, a dendritic replacement core, and a suffix.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Config
     - What it demonstrates
   * - ``configs/vision/alexnet_fc7_dendritic_500.yaml``
     - AlexNet ``fc7`` replacement with a 500-unit dendritic tensor-map core.
   * - ``configs/vision/alexnet_k1_einet_single.yaml``
     - K=1 classifier-layer replacement for single-GPU runs.
   * - ``configs/vision/alexnet_k1_einet_ddp.yaml``
     - Same K=1 replacement configured for DDP.
   * - ``configs/vision/pretrained_alexnet_fc.yaml``
     - Fully connected pretrained replacement with
       ``hierarchical_dendritic_tensor_map``.
   * - ``configs/vision/pretrained_alexnet_conv.yaml``
     - Feature-map replacement with ``hierarchical_dendritic_conv``.

Other Public Config Areas
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Location
     - Purpose
   * - ``configs/config_exp.yaml``
     - Single-GPU reference config using ``population_network``.
   * - ``configs/config_exp_fsdp.yaml``
     - FSDP reference config using ``population_network``.
   * - ``configs/sweep_config.yaml``
     - Sweep template using population-network dot paths.
   * - ``configs/vision/``
     - AlexNet / pretrained-backbone replacement examples.
   * - ``configs/distributed/``
     - Distributed launcher examples.
