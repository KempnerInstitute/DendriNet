Example configurations
======================

The repository ships complete configurations under ``configs/``: reference
configurations, training examples for the main model families and features,
sweep templates, analysis profiles and smaller building blocks. This page
lists every one of them and the command that uses it. Run the commands from
the root of the repository checkout.

Using the examples
------------------

A training example trains with ``dendritic-train``; check an edited copy
with ``--validate-only`` before a long run, and shorten a first run with
``--epochs``:

.. code-block:: bash

   dendritic-train configs/examples/population_network_two_layer.yaml --validate-only
   dendritic-train configs/examples/population_network_two_layer.yaml --epochs 1

Each example names its own ``outputs.results_dir`` and ``outputs.run_name``
(see :ref:`run-outputs`). The vision examples download MNIST or CIFAR-10 on
first use, and the sequence examples generate their data (see
:doc:`datasets`).

A sweep template expands into training configurations, which you then train
one by one or submit as Slurm jobs:

.. code-block:: bash

   dendritic-sweep --config configs/examples/indexed_backends/rnn_large.yaml \
     --generate-only --output-dir outputs/indexed_backends

An analysis profile holds only an ``analysis:`` section. Run it on a trained
checkpoint with ``dendritic-analyze-checkpoint``, giving the run's
``config.json`` (see :ref:`checkpoint-analysis`). For a checkpoint of the
MNIST example of :doc:`quickstart`:

.. code-block:: bash

   dendritic-analyze-checkpoint \
     --config <run directory>/config.json \
     --checkpoint <run directory>/final_model.pt \
     --analysis-config configs/examples/synaptic_source_support_analysis.yaml \
     --output-dir outputs/synaptic_source_support

Some analysis profiles name a module path, a number of somas or an image
size of the model they were written for; adapt those fields to your model
first.

.. _examples-map:

Example map
-----------

.. list-table::
   :header-rows: 1
   :widths: 38 46 16

   * - Configuration
     - What it shows
     - Used with
   * - ``configs/config_exp.yaml``
     - Annotated reference configuration with a comment for every
       population-network option: one CIFAR-10 layer of 32 inhibitory cells
       and 128 excitatory cells with ``[3, 3, 3]`` branch factors. See
       :ref:`reference-config`.
     - ``dendritic-train``
   * - ``configs/config_exp_fsdp.yaml``
     - The same model with indexed synapses for fully sharded data-parallel
       training (``distributed.mode: fsdp``). See :ref:`distributed-training`.
     - ``torchrun`` with ``train_experiments_fsdp``
   * - ``configs/sweep_config.yaml``
     - Annotated sweep template: synapse counts and excitatory branch factors
       of an MNIST population network, five seeds per combination (40
       configurations). See :ref:`sweeps`.
     - ``dendritic-sweep``
   * - ``configs/examples/population_network_feedforward.yaml``
     - One MNIST layer: an inhibitory population ``i`` (32 cells) and an
       excitatory population ``e`` (128 cells, ``[4, 4]``), with
       ``input -> i``, ``input -> e`` and same-step ``i -> e`` inhibition. Used
       in :doc:`quickstart`.
     - ``dendritic-train``
   * - ``configs/examples/population_network_cifar10.yaml``
     - CIFAR-10 training setup: 128 excitatory cells with ``[3, 3, 3]``
       branch factors and 32 inhibitory cells, 40 excitatory and 10
       inhibitory synapses per branch, ``occupancy_quantile`` gates, synapse
       weight decay and weight boosting.
     - ``dendritic-train``
   * - ``configs/examples/population_network_spatial_morphology_cifar10.yaml``
     - Every branch of 64 excitatory CIFAR-10 cells (``[2, 2, 2, 2]``) receives
       a fixed number of pixels from its own nested image region
       (``structured_connectivity`` with ``method: spatial_morphology``,
       ``topk_type: indexed``).
     - ``dendritic-train``
   * - ``configs/examples/population_network_dense_to_sparse_cifar10.yaml``
     - Learned pruning from dense allowed synapses down to the final synapse
       count during training (``topk_type: dense_to_sparse``).
     - ``dendritic-train``
   * - ``configs/examples/population_network_indexed_rewire_mnist.yaml``
     - Indexed synapses that store only the active contacts of each branch
       and periodically replace the weakest ones with newly sampled inputs
       (``topk_type: indexed_rewire``).
     - ``dendritic-train``
   * - ``configs/examples/population_network_normalized_additive_mnist.yaml``
     - Additive integration with voltage normalization
       (``use_shunting: false``, ``use_additive_normalization: true``) on
       MNIST.
     - ``dendritic-train``
   * - ``configs/examples/population_network_heterogeneous_feedforward.yaml``
     - Four populations in one MNIST layer (inhibitory ``pv`` and ``sst``,
       excitatory ``pyr_flat`` and ``pyr_deep``) with their own morphologies;
       ``population:`` blocks override options per population.
     - ``dendritic-train``
   * - ``configs/examples/population_network_two_layer.yaml``
     - Two CIFAR-10 layers (128 excitatory and 32 inhibitory cells, then 64
       and 16). The second layer reads ``layer0.e0`` and ``layer0.i0`` by
       qualified source names.
     - ``dendritic-train``
   * - ``configs/examples/population_network_local_learning_mnist.yaml``
     - Local credit assignment (``training.main.strategy: local_ca``) on
       MNIST, with the information and synapse-turnover analyzers. See
       :ref:`local-credit-assignment`.
     - ``dendritic-train``
   * - ``configs/examples/population_network_recurrent_pyr_pv_sst.yaml``
     - A recurrent microcircuit (``pyr``, ``pv``, ``sst``) on the
       ``context_dependent_decision`` sequence task: same-step input edges
       and delayed, probabilistic recurrent edges, trained with
       ``training.main.strategy: recurrent``.
     - ``dendritic-train``
   * - ``configs/examples/population_network_taus_and_delays.yaml``
     - Where the time constants live: the layer ``dt``, the dendritic time
       constants (``tau_base``, ``tau_ratio``, ``level_taus``), receptor time
       constants and delayed edges. See :ref:`temporal-constants`.
     - ``dendritic-train``
   * - ``configs/examples/population_network_synapse_types_ampa_nmda_gaba.yaml``
     - AMPA, NMDA, GABA-A and GABA-B receptor mixtures (``synapse_types``).
       See :ref:`synapse-types`.
     - ``dendritic-train``
   * - ``configs/examples/population_network_spiking.yaml``
     - Leaky integrate-and-fire somas (``dynamics_mode: spike``) with a spike
       readout and the spike-train analyzer. See :ref:`spiking-soma`.
     - ``dendritic-train``
   * - ``configs/examples/population_network_soma_feedback.yaml``
     - Feedback from the previous time step's soma output into the dendritic
       branches (``soma_feedback_enabled``). See :ref:`soma-feedback`.
     - ``dendritic-train``
   * - ``configs/examples/population_network_contextual_remapping.yaml``
     - A two-layer network for the context-cued EMNIST task
       (``data.dataset_name: contextual_remapping``) trained on a stream of
       freshly sampled presentations (``presentation_stream``), with
       stochastic-release synapses (``topk_type: indexed_release``) and
       ``endpoint_tanh`` gates. It needs the EMNIST download and a CUDA GPU
       for data generation.
     - ``dendritic-train``
   * - ``configs/examples/population_network_target_levels.yaml``
     - Two inhibitory classes that inhibit different dendritic levels of the
       excitatory cells (connection ``target_levels``), with variance Top-K
       synapses (``topk_variance_score: relative``), on the same task. See
       :ref:`population-connections`.
     - ``dendritic-train``
   * - ``configs/examples/contextual_remapping_affine_mlp.yaml``
     - A dense ``affine_mlp`` baseline (816-246-127-47 units) for the same
       task. See :ref:`affine-mlp`.
     - ``dendritic-train``
   * - ``configs/examples/indexed_backends/feedforward_large.yaml``
     - Sweep template for a three-layer feedforward ``einet`` on CIFAR-10
       (8,192 excitatory and 1,024 inhibitory cells per layer) with one
       variant per indexed projection backend (``eager``, ``recompute``,
       ``triton_transposed``, ``auto``). See :ref:`indexed-backends`.
     - ``dendritic-sweep``
   * - ``configs/examples/indexed_backends/rnn_large.yaml``
     - The same backend variants for a recurrent ``unified_ei`` network on a
       sequence task.
     - ``dendritic-sweep``
   * - ``configs/examples/branch_local_sensitivity_analysis.yaml``
     - Analysis profile: local sensitivity of one dendritic tree, including
       the integration rule and the gate derivatives at held-out operating
       points (``branch_local_sensitivity``).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/input_region_intervention_analysis.yaml``
     - Analysis profile: paired interventions on image regions of a
       feedforward classifier, with the layer-by-layer E/I responses
       (``input_region_intervention``).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/local_inhibitory_population_intervention_analysis.yaml``
     - Analysis profile: interventions on whole local inhibitory populations
       of a feedforward population network
       (``local_inhibitory_population_intervention``).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/path_matched_information_analysis.yaml``
     - Analysis profile: information carried by dendritic paths on fixed,
       label-independent supports (``path_matched_information``).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/path_matched_intervention_analysis.yaml``
     - Analysis profile: interventions on a fixed number of paths of one
       dendritic tree (``path_matched_intervention``).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/representation_accessibility_analysis.yaml``
     - Analysis profile: linear and nonlinear probes and class information of
       the soma populations of a population network
       (``representation_accessibility``).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/source_tuning_support_analysis.yaml``
     - Analysis profile: class tuning of the input coordinates each synapse
       contacts, for MNIST images (``source_tuning_support``).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/conductance_source_tuning_analysis.yaml``
     - The same analyzer for inputs that are not images (pixels mixed with
       context cues).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/synaptic_source_support_analysis.yaml``
     - Analysis profile: how often each input coordinate is contacted, and
       with which effective conductance, per dendritic depth (``weight``
       analyzer).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/synaptic_source_region_alignment_analysis.yaml``
     - Analysis profile: overlap of the contacted image region with the
       pixels of each soma's class on MNIST (``weight`` analyzer).
     - ``dendritic-analyze-checkpoint``
   * - ``configs/examples/operator_presets/00_mlp_flat.yaml`` to
       ``04_shunting_tree.yaml``
     - Five ``model.core`` bodies for the ``einet`` core, in steps from a
       signed flat layer to a shunting E/I tree; see the ``README.md`` in
       that directory.
     - ``model.core`` with ``type: einet``
   * - ``configs/examples/deployment/compact_inference.yaml``
     - Deployment policy for exporting a trained model with frozen sparse
       topology. See :ref:`compact-checkpoints`.
     - ``dendritic-compact-checkpoint --compression-config``
   * - ``configs/examples/validation_epoch_selection.yaml``
     - Selection of the best epoch of each run from validation metrics
       collected by ``dendritic-collect-sweep-metrics``.
     - ``dendritic-select-validation-epochs --config``
   * - ``configs/templates/analysis_full_template.yaml``
     - Every analyzer section with comments, to copy into an ``analysis:``
       section. See :ref:`analyzer-catalogue`.
     - Copy into a configuration
   * - ``configs/templates/indexed_runtime_template.yaml``
     - Recommended indexed-synapse runtime settings (backend, index type,
       workspace, memory-efficient gates) for structured cores and population
       blocks.
     - Copy into a configuration

Most population-network examples use indexed synapses
(``topk_type: indexed_rewire``, or ``indexed`` in the spatial-morphology
example) with ``indexed_index_dtype: auto``, the ``recompute`` projection
backend and memory-efficient gates (``reactivation_memory_efficient: true``).
The two context-cued EMNIST examples use the ``eager`` backend with 32-bit
indices. :ref:`indexed-backends` explains the choice.
