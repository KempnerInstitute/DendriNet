Datasets
========

All datasets are configured via ``data.dataset_name`` in the YAML config.
Processing options live under ``data.processing``; dataset-specific parameters
under ``data.dataset_params``.

.. contents:: On this page
   :local:
   :depth: 2

Standard Datasets
-----------------

``mnist``
   MNIST handwritten digits. 28x28 grayscale, 10 classes.

``fashion_mnist``
   Fashion MNIST. 28x28 grayscale, 10 classes.

``cifar10``
   CIFAR-10. 32x32 color, 10 classes.

``imagenet``
   ImageNet classification. By default, preprocessing uses the
   ``legacy_resize`` resize preset. Pretrained
   backbone runs should set ``transform_preset: "pretrained"`` for
   RandomResizedCrop/flip training transforms and Resize/CenterCrop eval
   transforms. Missing ImageNet roots raise an error. Flattening is rejected
   unless ``allow_flatten_materialize: true`` is set because full ImageNet
   tensors can materialize tens of GB. Parameters: ``transform_preset``,
   ``allow_flatten_materialize``, ``train_samples_per_class``,
   ``val_samples_per_class``.

``mnist_modulo10``
   Two random MNIST images concatenated; label = ``(label_a + label_b) % 10``.
   Parameter: ``shuffle_iterations`` (default: 2).

``cifar10_modulo10``
   CIFAR-10 analogue of ``mnist_modulo10``. Two random CIFAR examples are
   paired and the target is the modulo-10 sum of the source labels.
   Parameters: ``shuffle_iterations``, ``pair_seed``.

``double_cifar10_contextual_mod10``, ``double_mnist_contextual_mod10``
   Contextual paired-digit / paired-image tasks. Two source examples are
   combined and a context signal selects which arithmetic mapping to apply.
   Parameters: ``shuffle_iterations``, ``pair_seed``, ``context_seed``.

Synthetic Datasets
------------------

``correlated_gaussian``
   Gaussian-distributed inputs with controlled correlation structure between
   excitatory and inhibitory dimensions. Classes differ by E vs I activation
   patterns. Inputs are clamped to [0, 1].

   Parameters: ``n_samples``, ``input_dim``, ``n_classes``, ``rho_ee``
   (within-E correlation), ``rho_ii`` (within-I), ``rho_ei`` (cross E-I),
   ``ne``, ``ni`` (E/I dimension split), ``class_separation``, ``noise_std``.

``linear_separable``
   Classes separated by a hyperplane. Same correlation structure as
   ``correlated_gaussian``.

``nonlinear_separable``
   Nonlinear decision boundaries. ``nonlinearity_type``: ``"quadratic"``,
   ``"xor"``, ``"multiplicative"``, or ``"radial"``.

``nonlinear_interaction``
   Simple XOR-like interaction task. Class depends on the product of input
   dimension pairs.

``block_correlated``
   Synthetic classification task with block-structured feature covariance.

``balanced_information``
   Synthetic task designed to balance information across excitatory and
   inhibitory feature subsets.

``contextual_stream_gain_shift``
   Synthetic contextual stream task with train/test gain-shift control for
   alignment and robustness analysis.

``multixor``
   Multi-XOR synthetic classification task used for structured nonlinearity
   benchmarks.

``orientation_bars``
   Synthetic vision task built from oriented bar patterns.

``orthonet``
   Synthetic structured-vision dataset with orthogonal feature families.

.. code-block:: yaml

   data:
     dataset_name: correlated_gaussian
     dataset_params:
       correlated_gaussian:
         n_samples: 1000
         input_dim: 100
         n_classes: 2
         rho_ee: 0.3
         rho_ii: 0.3
         rho_ei: 0.0

Sequence Datasets
-----------------

For use with recurrent architectures (``ei_unified``, ``gru``, ``lstm``,
``vanilla_rnn``). Implementations live under
``dendritic_modeling.datasets.sequence`` and are grouped by task family:
``memory_tasks.py``, ``neuro_tasks.py``, ``signal_tasks.py``, and
``decision_tasks.py``. ``sequence_datasets.py`` remains the public
compatibility import path.

``sequential_mnist``
   MNIST presented as a sequence. Mode: ``"pixel"`` (seq_len=784, dim=1) or
   ``"row"`` (seq_len=28, dim=28).

``copy_task``
   Memorize and reproduce a symbol sequence after a delay. Parameters:
   ``seq_len``, ``delay``, ``n_symbols``.

``adding_problem``
   Sum two marked entries in a long sequence. Input shape: ``[seq_len, 2]``
   (values channel + mask channel). Target: scalar sum.

``variable_delay_dms``
   Delayed match-to-sample. A stimulus is presented, followed by a variable
   delay, then a probe. Binary classification (match vs non-match). Parameters:
   ``stimulus_dim``, ``stimulus_duration``, ``min_delay``, ``max_delay``,
   ``probe_duration``.

``multi_frequency``
   Multi-frequency signal classification. Target: dominant frequency class.
   Parameters: ``seq_len``, ``n_frequencies``, ``base_freq``, ``noise_std``.

``context_dependent_decision``
   Two sensory channels with a context cue indicating which channel is
   task-relevant (Mante et al., 2013 style). Binary classification on
   integrated evidence. Parameters: ``signal_dim``, ``context_duration``,
   ``integration_duration``, ``coherence_levels``, ``noise_std``,
   ``variable_delay``, ``conflict_fraction``.

``context_dependent_decision_hard``
   Harder variant of the context-dependent decision task with stronger
   distractors / conflict structure.

``context_dependent_decision_positive``, ``context_dependent_decision_hard_positive``
   Positive-drive encodings of the Mante-style tasks for shunting / positive
   conductance experiments.

``switching_context_decision``
   Evidence integration under slowly switching context. Fixed-length
   sequences partitioned into context blocks. Parameters: ``signal_dim``,
   ``n_contexts``, ``block_duration``, ``n_blocks``.

``switching_context_decision_positive``
   Positive-drive encoding of ``switching_context_decision``.

``timescale_generalization``
   Delayed match-to-sample with variable delays. Train on one delay range,
   test on another to evaluate temporal generalization. Parameters:
   ``stimulus_dim``, ``stimulus_duration``, ``min_delay``, ``max_delay``,
   ``probe_duration``.

``timescale_generalization_positive``
   Positive-drive version of ``timescale_generalization``.

``odr_generalization``
   Oculomotor delayed-response style task with train/test delay shifts.

``odr_generalization_positive``
   Positive-drive version of ``odr_generalization``.

``romo_delay_comparison``
   Romo-style delayed frequency-comparison task.

``romo_delay_comparison_positive``
   Positive-drive / positive-code version of the Romo task.

``gain_modulated_contextual_comparison``
   Gain- and context-modulated comparison task with pathway-specific nuisance
   variation.

``gain_modulated_contextual_comparison_positive``
   Positive-drive version of the gain/context comparison task.

``switching_lds``
   Switching linear dynamical system. Latent dynamics follow mode-dependent
   linear transitions with partial observation. Parameters: ``latent_dim``,
   ``obs_dim``, ``n_modes``, ``seq_len``.

``lorenz``
   Lorenz attractor dynamics prediction (regression). Input: ``[seq_len, 3]``.
   Target: next-step state. Parameters: ``seq_len``, ``dt``, ``sigma``,
   ``rho``, ``beta``.

``lorenz_sequence``
   Alias of ``lorenz`` retained for compatibility with older sequence configs.

``multi_sine_forecast``
   Multi-sine forecasting task with several concurrent oscillatory components.

``stringer_v1``
   Real V1 calcium imaging data (Stringer et al., 2019). Drifting grating
   responses. Parameters: ``n_neurons``, ``n_classes`` (orientation bins),
   ``train_fraction``, ``normalize``.

Poisson Generator
-----------------

Wrapper that adds Poisson noise to any base dataset. Parameters:
``multiplicative_gain``, ``fixed_gain_factor``, ``max_gain_factor``,
``poisson_sampling``, ``stimulus_duration``.

This wrapper is selected via ``data.dataset_name: poisson_generator`` and
delegates to a base dataset such as ``mnist`` or ``cifar10`` through
``data.dataset_params.poisson_generator.base_dataset``.

Data Processing
---------------

Set under ``data.processing``:

``flatten`` (bool, default: True)
   Flatten images to vectors.

``normalize`` (bool, default: False)
   Z-score normalize inputs.

``label_noise_rate`` (float, default: 0.0)
   Fraction of labels to randomize.

.. code-block:: yaml

   data:
     dataset_name: mnist
     processing:
       flatten: true
       normalize: false
       label_noise_rate: 0.0

ImageNet / Pretrained Backbones
-------------------------------

Use the pretrained preset whenever ImageNet enters a torchvision pretrained
backbone such as AlexNet:

.. code-block:: yaml

   data:
     dataset_name: imagenet
     base_dir: "/path/to/imagenet"
     processing:
       flatten: false
       normalize: true
     dataset_params:
       imagenet:
         transform_preset: "pretrained"
         loader_backend: "auto"
         allow_flatten_materialize: false

``loader_backend: "auto"`` preserves the default behavior: official
torchvision ImageNet roots are loaded through ``torchvision.datasets.ImageNet``,
while plain ``train/<class>`` and ``val/<class>`` trees use
``ImageFolder``.  On clusters where ImageNet is already staged as class
directories on local storage, set ``loader_backend: "imagefolder"`` to avoid
the torchvision ImageNet metadata path.  Missing ImageNet roots fail loudly;
there is no synthetic-data fallback.
