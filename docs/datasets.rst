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

``mnist_modulo10``
   Two random MNIST images concatenated; label = ``(label_a + label_b) % 10``.
   Parameter: ``shuffle_iterations`` (default: 2).

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
``vanilla_rnn``).

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

``lorenz_sequence``
   Lorenz attractor dynamics prediction (regression). Input: ``[seq_len, 3]``.
   Target: next-step state. Parameters: ``seq_len``, ``dt``, ``sigma``,
   ``rho``, ``beta``.

``stringer_v1``
   Real V1 calcium imaging data (Stringer et al., 2019). Drifting grating
   responses. Parameters: ``n_neurons``, ``n_classes`` (orientation bins),
   ``train_fraction``, ``normalize``.

Poisson Generator
-----------------

Wrapper that adds Poisson noise to any base dataset. Parameters:
``multiplicative_gain``, ``fixed_gain_factor``, ``max_gain_factor``,
``poisson_sampling``, ``stimulus_duration``.

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
