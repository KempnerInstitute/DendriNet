Training Strategies
===================

The framework uses an optional encoder trainer followed by the main trainer for
the core network and decoder. The main trainer strategy is set via
``training.main.strategy`` in the YAML config.

.. contents:: On this page
   :local:
   :depth: 2

Available Strategies
--------------------

.. figure:: _static/architecture/learning-rules.svg
   :alt: Credit-assignment comparison — backprop (exact gradient), feedback alignment/DFA (fixed random feedback), and local credit assignment (Hebbian/STDP + error broadcast)
   :width: 100%

   How the weight update reaches each layer across the available strategies:
   **backprop** (``standard``) flows the exact gradient back through every layer;
   **feedback alignment** (``fa``/``dfa``, plus shunting-aware ``shunting_fa``/
   ``shunting_dfa``) uses fixed random feedback weights with no weight transport;
   **local credit assignment** (``local_ca``) updates each layer from local
   pre/post activity (Hebbian/STDP) plus an error-broadcast/HSIC signal, with no
   global backward pass.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Description
   * - ``standard``
     - Backpropagation with optional regularization and pruning
   * - ``fa``
     - Feedback Alignment with fixed random feedback matrices
   * - ``dfa``
     - Direct Feedback Alignment; output error projected to each layer
   * - ``shunting_fa``
     - FA with conductance-based modulation
   * - ``shunting_dfa``
     - DFA with conductance-based modulation
   * - ``local_ca``
     - Local credit assignment (3/4/5-factor Hebbian rules and trace STDP)
   * - ``freeze_layers``
     - Progressive layer unfreezing
   * - ``freeze_branches``
     - Progressive branch unfreezing
   * - ``freeze_branch_kl``
     - Branch freezing with KL divergence loss
   * - ``voltage_stabilization``
     - Pretrain to stabilize dendritic voltages
   * - ``homeostatic_control``
     - Homeostatic regulation of soma activity
   * - ``recurrent``
     - Sequence training for RNN architectures
   * - ``multi_stage``
     - Chain multiple strategies in sequence

Distributed launch mode is configured separately via ``distributed.mode``:
``none``, ``ddp``, or ``fsdp``.

Transformer Replacement Distillation
------------------------------------

Transformer MLP/FFN replacement is trained with the dedicated
``training.transformer_replacement`` block rather than the image-classification
``training.main.strategy`` loop.  The target model and layers are defined in
``model.transformer_replacement``; the dendritic architecture is defined in
``model.core``; and this block defines how teacher data are produced.

The most controlled mode is ``layerwise_distillation``:

.. math::

   h_{\ell} \mapsto \mathrm{MLP}_{\ell}(h_{\ell})
   \quad\text{is approximated by}\quad
   h_{\ell} \mapsto D_{\theta,\ell}(h_{\ell}).

Only the dendritic replacement is trained in this mode
(``train_target: replacement_only``).  This keeps the experiment interpretable:
each loss value measures how well the dendritic module matches the original
target module before any full-model language-model fine-tuning.

``joint_lm_distillation`` patches the selected modules inside a loaded causal
LM and trains through the language-modeling loss on token windows from
``teacher_source: hf_text``.  With ``train_target: replacement_only`` the
backbone remains frozen and only the dendritic modules update.  With
``train_target: full_model`` the entire patched student is trainable, which is
the fine-tuning path after a replacement has already passed the layerwise
smoke test.

.. code-block:: yaml

   training:
     main:
       optimizer:
         name: adamw
         lr: 3.0e-4
         weight_decay: 0.01

     transformer_replacement:
       enabled: true
       mode: layerwise_distillation
       teacher_source: hf_text          # synthetic_mlp | hidden_cache | hf_text
       train_target: replacement_only
       layers: [0, 1, 2]
       device: cuda
       dtype: bfloat16
       sequence_length: 128
       batch_size: 1
       max_steps: 1000
       eval_every: 100
       loss: mse
       text:
         text_path: /path/to/text_corpus.txt
         tokenizer_name: deepseek-ai/DeepSeek-V3

For joint training, switch the mode and choose the trainable scope:

.. code-block:: yaml

   training:
     transformer_replacement:
       enabled: true
       mode: joint_lm_distillation
       teacher_source: hf_text
       train_target: full_model         # or replacement_only
       lm_loss_weight: 1.0
       kl_loss_weight: 0.0
       hidden_loss_weight: 0.0

Run it with:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python -m \
     dendritic_modeling.scripts.training.train_transformer_replacement \
     configs/examples/transformer_replacement_synthetic_distillation.yaml

For DeepSeek-scale runs, the recommended workflow is to start with
``synthetic_mlp`` for a learning smoke test, then use ``hidden_cache`` if you
can precompute teacher activations, and only then use ``hf_text`` for direct
teacher-hook training.

Standard Training
-----------------

Basic backpropagation. Supports gradient clipping, early stopping, mixed
precision, NaN detection, and adaptive clipping.

.. code-block:: yaml

   training:
     main:
       strategy: standard
       common:
         epochs: 100
         batch_size: 256
         loss_function: ce
         grad_clip_value: 5.0
         early_stopping: true
         patience: 20

Key ``common`` parameters:

``epochs`` (int, default: 1000), ``batch_size`` (int, default: 512),
``loss_function`` (str, default: ``"ce"``), ``grad_clip_value`` (float,
default: 2.0), ``early_stopping`` (bool, default: True), ``patience``
(int, default: 500), ``use_amp`` (bool, default: False),
``enable_nan_checking`` (bool, default: False),
``enable_adaptive_clipping`` (bool, default: False).

Feedback Alignment
------------------

Implements FA (Lillicrap et al., 2016) and DFA (Nokland, 2016). Uses fixed
random feedback matrices instead of weight transposes for error propagation.

- **FA** (``fa``): errors backpropagate layer-by-layer through random B
  matrices.
- **DFA** (``dfa``): each hidden layer receives the output error directly.

``shunting_fa`` and ``shunting_dfa`` extend these with conductance-based
modulation: the error at each layer is modulated by the local total
conductance and driving force, ``error * g_tot * (E_rev - V)``.

.. code-block:: yaml

   training:
     main:
       strategy: dfa

Local Credit Assignment
-----------------------

Biologically-plausible local learning using Hebbian-style updates with
locally available signals (pre/post-synaptic activity) and a broadcast soma
error. The current config path is ``training.main.learning_strategy_config``.

Rule Variants
~~~~~~~~~~~~~

``rule_variant: "3f"`` -- three-factor rule
   ``pre * post * error``. The post-synaptic factor depends on the dynamics
   mode: in conductance mode it uses driving force ``(E_rev - V)`` scaled by
   ``1/g_tot``; in additive mode it uses ``(V - theta)`` with optional gain
   control.

``rule_variant: "4f"`` -- four-factor rule
   Adds a morphology correlation factor ``rho`` that captures the correlation
   between branch-level and soma-level activity. Estimated via EMA
   (``rho_estimator: "ema"``) or data augmentation (``"augment"``).

``rule_variant: "5f"`` -- five-factor rule
   Adds an information proxy ``phi`` that estimates class-conditional
   information content. Modes: ``"conditional"`` (conditional variance proxy)
   or ``"variance"`` (raw variance).

``rule_variant: "stdp"`` -- trace-based STDP rule
   Uses local pre/post activity traces at TopK synapses. This is a separate
   local rule: no task-error term is required unless ``stdp.use_error_modulation``
   is enabled. It can also be added as an auxiliary rule to ``3f``/``4f``/``5f``
   by setting ``stdp.enabled: true``.

STDP Rule
~~~~~~~~~

The STDP implementation keeps one pre-synaptic and one post-synaptic trace per
recorded TopK synapse module. At each local update it applies:

.. code-block:: text

   Delta w_ij = A_plus * post_i(t) * pre_trace_j(t-1)
              - A_minus * post_trace_i(t-1) * pre_j(t)

Positive ``Delta w`` is converted to a negative optimizer gradient so standard
PyTorch optimizers potentiate the underlying raw synapse parameter.

.. code-block:: yaml

   training:
     main:
       strategy: local_ca
       learning_strategy_config:
         rule_variant: stdp
         stdp:
           apply_to: [exc, rec_exc]
           activity_mode: relu
           tau_pre: 20.0
           tau_post: 20.0
           a_plus: 1.0
           a_minus: 0.5
           learning_rate_scale: 1.0

Key STDP options:

``apply_to``
   Synaptic pathways to update. Supported values are ``exc``, ``inh``,
   ``rec_exc``, ``rec_inh``, or ``all``. The default is ``[exc, rec_exc]``.

``activity_mode``
   How local tensors are converted to activities: ``relu`` (default),
   ``identity``, ``binary``, or ``abs``.

``use_error_modulation``
   If true, scales the trace update by the broadcast soma error. This gives a
   reward/error-modulated STDP variant while keeping the base rule local.

``inhibitory_update_sign``
   Sign multiplier for ``inh`` and ``rec_inh`` pathways. The default ``-1``
   makes inhibitory STDP anti-Hebbian; set ``1`` for Hebbian inhibitory updates.

Error Broadcast
~~~~~~~~~~~~~~~

``error_broadcast_mode``
   ``"scalar"`` -- single scalar error for the whole network.
   ``"per_soma"`` -- per-neuron error at the soma level.
   ``"local_mismatch"`` -- locally computed mismatch signal.

``broadcast_bandwidth``
   ``"full"`` -- transmit full error values.
   ``"sign_only"`` -- transmit only the sign of the error.
   ``"quantized"`` -- quantize to ``broadcast_bits`` bits.
   ``"sparse_topk"`` -- keep only top ``broadcast_topk_fraction`` of values.

HSIC Auxiliary Loss
~~~~~~~~~~~~~~~~~~~

Optional Hilbert-Schmidt Independence Criterion loss to encourage decorrelated
representations. Enabled via ``hsic.enabled: true``. Kernel options:
``"linear"``, ``"rbf"``, ``"polynomial"``.

.. code-block:: yaml

   training:
     main:
       strategy: local_ca
       learning_strategy_config:
         rule_variant: "4f"
         error_broadcast_mode: per_soma
         three_factor:
           dynamics_mode: auto
           use_conductance_scaling: true
           use_driving_force: true
         four_factor:
           rho_mode: pearson
           rho_estimator: ema
           ema_alpha: 0.05
         hsic:
           enabled: true
           weight: 0.01

Component Compatibility
~~~~~~~~~~~~~~~~~~~~~~~

The local learning recorder captures all TopK pathways on a
``DendriticBranchLayer``:

.. code-block:: text

   external E input  -> branch_excitation      -> exc
   external I input  -> branch_inhibition      -> inh
   recurrent E input -> branch_recurrent       -> rec_exc
   recurrent I input -> branch_rec_inhibition  -> rec_inh

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Component
     - Local-learning behavior
   * - Population network
     - Branch-local rules operate on each population's dendritic branch layers.
       Named populations are compatible as long as they use the shared
       ``DendriticBranchLayer`` / ``TopKLinear`` modules.
   * - General synapse types
     - Local rules update the shared TopK synaptic weights. Receptor fractions,
       reversal potentials, time constants, and short-term plasticity parameters
       remain configured model dynamics, not learned by LocalCA/STDP.
   * - Structured connectivity
     - Connection masks are always respected. ``update_inactive_weights`` can
       allow non-active TopK weights to move, but forbidden masked connections
       stay frozen. Fixed ``indexed`` synapses compute 3F/4F/5F updates directly
       on their stored contacts, without a dense output-by-input gradient.
   * - Recurrent pathways
     - ``3f``/``4f``/``5f`` and STDP update ``rec_exc`` and ``rec_inh``
       pathways when those modules are present and enabled by config.
   * - Spiking soma
     - STDP can use spike-like or rate-like pre/post tensors depending on the
       model readout and ``activity_mode``. Exact event timing is approximated
       by the trainer's update steps; for finer time-bin STDP, use sequence
       batches with recurrent outputs exposed at each step.

Under DDP, LocalCA averages its manually assigned local gradients across ranks
before clipping and stepping the optimizer. This is required because these
gradients are not produced by an ordinary ``loss.backward()`` call and
therefore do not use DDP's autograd reduction hooks.

Morphology-Aware Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extensions that use dendritic tree topology explicitly:

``use_path_propagation``
   Approximate path-integrated attenuation along dendritic branches.

``use_dendritic_normalization``
   Normalize update by total conductance at each branch.

``use_branch_type_rules``
   Differentiate apical vs basal branches with separate scaling
   (``apical_branch_scale``, ``basal_branch_scale``).

Recurrent Training
------------------

For sequence data (``[B, T, D]`` inputs). Handles temporal loss computation
for both ``output_mode: last`` and ``output_mode: all``.

``grad_clip`` (float, default: 1.0), ``lr_schedule`` (``"none"`` or
``"cosine"``), ``lr_warmup_epochs`` (int, default: 0).

.. code-block:: yaml

   training:
     main:
       strategy: recurrent
       common:
         epochs: 100
         batch_size: 64
         loss_function: ce

Multi-Stage Training
--------------------

Chains multiple strategies in sequence. Each stage can have its own strategy,
epoch count, optimizer state, and config overrides.

.. code-block:: yaml

   training:
     main:
       strategy: multi_stage
       multi_stage_config:
         analyze_between_stages: true
         use_best_from_each_stage: true
         stages:
           - learning_strategy: standard
             epochs: 50
             trainer_config:
               loss_function: ce
               early_stopping: true
           - learning_strategy: local_ca
             epochs: 50
             local_rule_config:
               rule_variant: "4f"

FSDP Distributed Training
--------------------------

For large models across multiple GPUs or nodes. See
:doc:`fsdp_troubleshooting` for setup and runtime issues.

.. code-block:: yaml

   distributed:
     mode: "fsdp"
     fsdp:
       sharding_strategy: "FULL_SHARD"
       mixed_precision: true
       cpu_offload: false
       gradient_checkpointing: false

Parameter Groups
----------------

Per-component learning rates, set via ``training.main.common.param_groups``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``lr``
     - 0.01
     - Base learning rate
   * - ``split_params``
     - True
     - Use separate LRs per component type
   * - ``topk_lr``
     - 0.001
     - Sparse synaptic layer LR
   * - ``blocklinear_lr``
     - 0.0001
     - BlockLinear (branch aggregation) LR
   * - ``reactivation_lr``
     - 0.0001
     - Reactivation nonlinearity LR
   * - ``decoder_lr``
     - 0.001
     - Decoder network LR

Optimizer
---------

Set via ``training.main.optimizer``: ``"adam"`` (default), ``"adamw"``,
``"sgd"``, ``"rmsprop"``. Standard parameters: ``lr``, ``weight_decay``,
``momentum`` (SGD), ``betas`` (Adam).

Loss Functions
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Name
     - Alias
     - Description
   * - ``ce``
     - ``cross_entropy``
     - Cross-entropy (classification)
   * - ``mse``
     - ``mean_squared_error``
     - Mean squared error (regression)
   * - ``log_mse``
     - ``log_mean_squared_error``
     - Log of MSE
   * - ``bce``
     - ``binary_cross_entropy``
     - Binary cross-entropy
   * - ``cat_nll``
     - ``categorical_negative_log_likelihood``
     - Categorical NLL
   * - ``vae_elbo``
     - ``variational_autoencoder_elbo``
     - VAE ELBO loss
   * - ``neg_cos_sim``
     - ``negative_cosine_similarity``
     - Negative cosine similarity
   * - ``log_adj_neg_cos_sim``
     -
     - Log-adjusted negative cosine similarity

Regularization and Pruning
--------------------------

Regularization (``training.main.regularization``):

- ``l1_weight``, ``l2_weight``: L1/L2 penalty weights
- ``selective_layers``, ``selective_branches``, ``selective_weight_types``:
  restrict to specific components
- ``ei_ratio_weight``: enforce E/I weight ratio constraint

Pruning (``training.main.pruning``):

- ``enabled``: bool
- ``threshold``: magnitude threshold for pruning
- ``frequency``: prune every N epochs
- ``selective_layers``, ``selective_branches``: restrict scope
