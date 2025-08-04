Connectivity & Synaptic Sparsity
================================

The framework offers several *Top-K* layers that enforce controlled synaptic sparsity.  Each keeps only the **K** most important synapses per neuron, dramatically reducing parameter count and promoting biological realism.

Layer Variants
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Class
     - Selection Strategy
     - Typical Use-case
   * - :pyclass:`dendritic_modeling.networks.architectures.excitation_inhibition.synapse.TopKLinear`
     - Deterministic **K** highest weights per neuron (optionally with additive noise during training)
     - Baseline controlled sparsity
   * - :pyclass:`dendritic_modeling.networks.architectures.excitation_inhibition.synapse.StochasticTopKLinear`
     - Samples a Top-K set from a Boltzmann distribution over weight rank (temperature-controlled)
     - Exploration of multiple subnetworks during training
   * - :pyclass:`dendritic_modeling.networks.architectures.excitation_inhibition.synapse.VarianceTopKLinear`
     - Combines weight magnitude with running activation variance (Hebbian-like importance)
     - Data-driven pruning for sensory models

Configuring K
-------------
Specify **K** per layer (and optional temperature/variance tracking) via the YAML config:

.. code-block:: yaml

   core_network:
     type: EINet
     params:
       layer_sizes: [512, 256, 10]
       synapse:
         kind: stochastic_topk   # {topk, stochastic_topk, variance_topk}
         K:    [50, 30, null]    # null → dense output layer
         temperature: 0.5        # only used by stochastic_topk

Runtime Adaptation
------------------
Call :pymeth:`update_K()` on any Top-K layer to adjust sparsity during training—for example, to implement gradual pruning:

.. code-block:: python

   for new_K in [100, 80, 60, 40]:
       for module in model.modules():
           if hasattr(module, "update_K"):
               module.update_K(new_K)

Internals
~~~~~~~~~
All three variants inherit from ``TopKLinear`` and override ``_select_mask`` to build a binary weight mask.  Gradients flow through the *pre-masked* weight matrix, so pruning is differentiable via straight-through estimation.

Performance Tip
~~~~~~~~~~~~~~~
On CUDA, sparse matrix multiplications are internally dispatched to dense GEMM for small K; use batch size ≥ 64 to saturate the GPU. 