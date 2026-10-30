.. _python-api:

Python API
==========

Most work with DendriNet goes through configuration files and the
command-line programs (:doc:`cli_reference`). This page documents the
functions and classes for doing the same from Python: loading a
configuration, building a model, adding components to the registries,
running analyses and saving training state. The descriptions are taken from
the docstrings in the source.

Configuration
-------------

.. autofunction:: dendritic_modeling.config.load_config

.. autofunction:: dendritic_modeling.config.config_from_mapping

.. autofunction:: dendritic_modeling.config.load_analysis_config

Building models
---------------

.. autofunction:: dendritic_modeling.networks.architectures.factory.get_architecture

.. autofunction:: dendritic_modeling.networks.architectures.biological_constraints.biological_violations

.. autofunction:: dendritic_modeling.analysis.utils.model_resources.topk_parameter_inventory

Extension registries
--------------------

Each function adds a named component that configuration files can then
select; :ref:`extension-registries` describes what each registry covers.

.. autofunction:: dendritic_modeling.networks.architectures.registry.register_architecture

.. autofunction:: dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory.register_sparse_layer

.. autofunction:: dendritic_modeling.networks.activations.factory.register_activation_builder

.. autofunction:: dendritic_modeling.datasets.registry.register_dataset

.. autofunction:: dendritic_modeling.datasets.sequence.builders.registry.register_sequence_dataset_builder

.. autofunction:: dendritic_modeling.training.factory.register_trainer_strategy

.. autofunction:: dendritic_modeling.training.loss.functions.register_loss_function

.. autofunction:: dendritic_modeling.training.optimizers.factory.register_optimizer

.. autofunction:: dendritic_modeling.analysis.registry.register_analyzer

Plugins
-------

Installed plugins and the registries that only plugins use;
:doc:`extending` describes them.

.. autofunction:: dendritic_modeling.plugins.load_plugins

.. autofunction:: dendritic_modeling.plugins.activate_plugin

.. autofunction:: dendritic_modeling.plugins.loaded_plugins

.. autoclass:: dendritic_modeling.plugins.PluginRecord

.. autoexception:: dendritic_modeling.plugins.PluginLoadError

.. autofunction:: dendritic_modeling.config.extensions.register_config_section

.. autofunction:: dendritic_modeling.config.extensions.register_config_migration

.. autofunction:: dendritic_modeling.models.builders.register_model_builder

.. autoclass:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics.AdditiveRule

.. autofunction:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics.register_additive_rule

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse.IndexedSparseLinear.effective_sparse_weight

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse.IndexedSparseLinear.project_with_weights

Analysis
--------

.. autoclass:: dendritic_modeling.analysis.managers.analysis_manager.AnalysisManager
   :members: run_analysis

.. autofunction:: dendritic_modeling.scripts.analysis.run_checkpoint_analysis.run_checkpoint_analysis

.. autoclass:: dendritic_modeling.analysis.utils.streaming_covariance.StreamingCovariance
   :members:

.. autofunction:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.capture_analysis_currents

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.DendriticBranchLayer.set_analysis_currents

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.DendriticBranchLayer.get_analysis_currents

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.DendriticBranchLayer.clear_analysis_currents

.. autoclass:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.compartment_perturbation.CompartmentPerturbation

.. autofunction:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.perturb_compartments

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.DendriticBranchLayer.set_compartment_perturbation

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer.DendriticBranchLayer.clear_compartment_perturbation

.. autoclass:: dendritic_modeling.analysis.tools.node_perturbation_credit.NodePerturbationCreditAnalyzer
   :members: analyze

.. automethod:: dendritic_modeling.training.strategies.local_learning.LocalCreditAssignment.assign_gradients

Training state
--------------

:ref:`resuming-training` describes saving and restoring the state of a run at
an epoch boundary.

.. autofunction:: dendritic_modeling.training.state.capture_random_state

.. autofunction:: dendritic_modeling.training.state.restore_random_state

Dendritic levels and delivered errors
-------------------------------------

:ref:`dendrinet-level-overrides` describes per-level overrides,
re-initialization recipes and level outputs; :ref:`local-delivery` the
delivered-error building blocks.

.. autoclass:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.level_overrides.DendriticLevelOverride

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet.DendriNet.forward_levels

.. automethod:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet.DendriNet.reinitialize

.. autoclass:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.reinitialization.LogNormalReinitialization

.. autoclass:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.reinitialization.LogOffset

.. autoclass:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.level_interventions.CurrentInjection

.. autofunction:: dendritic_modeling.training.local_delivery.soma_error

.. autofunction:: dendritic_modeling.training.local_delivery.delivered_error_objective

.. autofunction:: dendritic_modeling.training.local_delivery.relative_resistance

.. autofunction:: dendritic_modeling.training.local_delivery.coupling_ratios

.. autofunction:: dendritic_modeling.training.local_delivery.activation_slope

.. autofunction:: dendritic_modeling.training.local_delivery.expand_to_children

:ref:`node-perturbation` describes the node-perturbation estimates.

.. autofunction:: dendritic_modeling.training.node_perturbation.node_perturbation_estimates

.. autoclass:: dendritic_modeling.training.node_perturbation.NodePerturbationEstimate
   :members: evaluations_per_example

.. autofunction:: dendritic_modeling.training.node_perturbation.group_basis

.. autofunction:: dendritic_modeling.training.node_perturbation.check_orthonormal_basis

Comparison models
-----------------

:ref:`baseline-models` describes the parameter-budget model families built by
``dendritic_modeling.networks.benchmarks.parameter_budget.build_model``.

.. autofunction:: dendritic_modeling.networks.benchmarks.parameter_budget.match_parameter_budget

.. autofunction:: dendritic_modeling.networks.benchmarks.parameter_budget.model_report

.. autofunction:: dendritic_modeling.networks.utils.readouts.make_signed_readout

.. autofunction:: dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.passive_equilibrium.passive_star_soma_voltage

Compact checkpoints
-------------------

.. autofunction:: dendritic_modeling.deployment.compression.freeze_sparse_topology_

.. autofunction:: dendritic_modeling.deployment.compression.bitmask_slot_order_keys

.. autofunction:: dendritic_modeling.networks.checkpoints.compact_sparse_bitmask_state_dict

.. autofunction:: dendritic_modeling.networks.checkpoints.decode_sparse_bitmask_state_dict

Project provenance
------------------

:doc:`project_provenance` describes the source contract, export and evidence
formats these functions read and the receipts they write.

.. autofunction:: dendritic_modeling.utils.project_source.read_contract

.. autofunction:: dendritic_modeling.utils.project_source.verify_contract

.. autofunction:: dendritic_modeling.utils.project_source.activate_source

.. autoexception:: dendritic_modeling.utils.project_source.SourceContractError

.. autofunction:: dendritic_modeling.utils.project_export.export_project

.. autofunction:: dendritic_modeling.utils.project_evidence.check_evidence

.. autofunction:: dendritic_modeling.utils.project_evidence.sha256_file

.. autoexception:: dendritic_modeling.utils.project_evidence.EvidenceError

.. autofunction:: dendritic_modeling.utils.exact_file_inventory.build_inventory

.. autofunction:: dendritic_modeling.utils.exact_file_inventory.validate_inventory

.. autofunction:: dendritic_modeling.utils.exact_file_inventory.balanced_assignment

.. autofunction:: dendritic_modeling.utils.exact_file_inventory.verify_shard

.. autofunction:: dendritic_modeling.utils.exact_file_inventory.merge_verified_shards

.. autofunction:: dendritic_modeling.utils.exact_file_inventory.file_snapshot

.. autofunction:: dendritic_modeling.utils.exact_file_inventory.canonical_sha256
