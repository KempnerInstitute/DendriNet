Datasets
========

``data.dataset_name`` selects the dataset of a run (default ``cifar10``).
``data.processing`` holds the input options shared by all datasets, and
``data.dataset_params.<dataset_name>`` holds the options of the selected
dataset. ``experiment.train_valid_split`` and the ``experiment`` test-split
options decide how the data are divided.

.. code-block:: yaml

   experiment:
     train_valid_split: 0.8
   data:
     dataset_name: mnist
     base_dir: ""            # empty: DENDRITIC_DATA_DIR or the user cache
     processing:
       flatten: true
       normalize: false
     dataset_params: {}

.. contents:: On this page
   :local:
   :depth: 2

Selecting a dataset
-------------------

An unknown ``data.dataset_name`` raises when the datasets are built, with the
closest known name and the full list of names. The
:ref:`dataset-catalogue` below lists every name the loaders accept.

``data.dataset_params`` maps dataset names to option blocks. Only the block
named by ``data.dataset_name`` is read; blocks for other datasets are ignored,
so one file can keep the options of several datasets. A key in the selected
block that its loader does not read raises, with a close match when there is
one. These keys are checked when the dataset is built (at the start of
training, or with ``dendritic-train config.yaml --validate-only``), not when
the configuration file loads. Annotations for people or other tools belong in
the top-level ``metadata:`` section.

A ``flatten`` or ``normalize`` key in the selected ``dataset_params`` block
takes precedence over ``data.processing``. Generated datasets draw their
samples from ``experiment.dataset_seed`` unless the block sets ``seed``, and
random train/validation partitions use ``experiment.split_seed``; both fall
back to ``experiment.seed`` (see :ref:`random-seeds`).

.. _dataset-catalogue:

Dataset catalogue
-----------------

Every value of ``data.dataset_name`` that the loaders accept. A name ending in
``_positive`` is the positive-input variant of the task before it (see
:ref:`dataset-sequence`).

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - ``data.dataset_name``
     - Family
     - Content
   * - ``mnist``
     - :ref:`Vision <dataset-vision>`
     - MNIST digits, 28x28 grayscale, 10 classes.
   * - ``fashion_mnist``
     - :ref:`Vision <dataset-vision>`
     - Fashion-MNIST, 28x28 grayscale, 10 classes.
   * - ``cifar10``
     - :ref:`Vision <dataset-vision>`
     - CIFAR-10, 32x32 color, 10 classes.
   * - ``cifar100``
     - :ref:`Vision <dataset-vision>`
     - CIFAR-100, 32x32 color, 100 classes.
   * - ``imagenet``
     - :ref:`ImageNet <dataset-imagenet>`
     - ILSVRC-2012 classification from a local copy, 1000 classes.
   * - ``mnist_modulo10``
     - :ref:`Paired arithmetic <dataset-paired>`
     - Two MNIST digits; label ``(a + b) mod 10``.
   * - ``cifar10_modulo10``
     - :ref:`Paired arithmetic <dataset-paired>`
     - Two CIFAR-10 images; label ``(a + b) mod 10``.
   * - ``double_mnist_contextual_mod10``
     - :ref:`Paired arithmetic <dataset-paired>`
     - Two MNIST digits and a context cue; label ``(a + b) mod 10`` or
       ``(a - b) mod 10``.
   * - ``double_cifar10_contextual_mod10``
     - :ref:`Paired arithmetic <dataset-paired>`
     - Two CIFAR-10 images and a context cue; label ``(a + b) mod 10`` or
       ``(a - b) mod 10``.
   * - ``poisson_generator``
     - :ref:`Poisson generator <dataset-poisson>`
     - MNIST or CIFAR-10 intensities as Poisson rates, with optional
       multiplicative gain.
   * - ``correlated_gaussian``
     - :ref:`Synthetic <dataset-synthetic>`
     - Gaussian E and I features with set within- and cross-group
       correlations; classes differ in E versus I activity.
   * - ``linear_separable``
     - :ref:`Synthetic <dataset-synthetic>`
     - Two classes separated by a hyperplane, correlated E/I features.
   * - ``nonlinear_separable``
     - :ref:`Synthetic <dataset-synthetic>`
     - Two classes with a quadratic, XOR, multiplicative or radial boundary.
   * - ``nonlinear_interaction``
     - :ref:`Synthetic <dataset-synthetic>`
     - XOR-like task where the class depends on products of input pairs.
   * - ``block_correlated``
     - :ref:`Synthetic <dataset-synthetic>`
     - Feature blocks with different correlation structures.
   * - ``balanced_information``
     - :ref:`Synthetic <dataset-synthetic>`
     - Class information spread evenly, concentrated, or graded across
       features.
   * - ``heterogeneous_information``
     - :ref:`Synthetic <dataset-synthetic>`
     - A few highly informative features among weakly informative ones.
   * - ``nonlinear_transformation``
     - :ref:`Synthetic <dataset-synthetic>`
     - Low-dimensional XOR, quadratic or multiplicative structure.
   * - ``asymmetric_noise``
     - :ref:`Synthetic <dataset-synthetic>`
     - Different noise levels on the E and I features.
   * - ``population_code``
     - :ref:`Synthetic <dataset-synthetic>`
     - Mixed-selectivity or ring population code projected onto E and I
       inputs.
   * - ``contextual_stream_gain_shift``
     - :ref:`Synthetic <dataset-synthetic>`
     - Two input streams and a context cue that selects the relevant stream;
       stream gains shift between training and test.
   * - ``branch_local_gain_load``
     - :ref:`Synthetic <dataset-synthetic>`
     - Positive E and I streams under group-wise multiplicative gain and
       optional inhibitory load.
   * - ``hierarchical_gain_load``
     - :ref:`Synthetic <dataset-synthetic>`
     - Positive E and I streams under nested multiscale gain fields.
   * - ``sequential_mnist``
     - :ref:`Sequence: memory <dataset-sequence-memory>`
     - MNIST read pixel by pixel or row by row.
   * - ``copy_task``
     - :ref:`Sequence: memory <dataset-sequence-memory>`
     - Reproduce a symbol sequence after a delay.
   * - ``adding_problem``
     - :ref:`Sequence: memory <dataset-sequence-memory>`
     - Sum the two marked values of a long sequence (regression).
   * - ``variable_delay_dms``
     - :ref:`Sequence: memory <dataset-sequence-memory>`
     - Delayed match-to-sample with a variable delay.
   * - ``speeded_distractor_dms``, ``speeded_distractor_dms_positive``
     - :ref:`Sequence: memory <dataset-sequence-memory>`
     - Delayed match-to-sample with interleaved distractors and a fast
       secondary task, in counterfactual pairs.
   * - ``hierarchical_temporal``
     - :ref:`Sequence: decision <dataset-sequence-decision>`
     - A context cue decides which block type to count over a sequence of
       noisy blocks.
   * - ``context_dependent_decision``, ``context_dependent_decision_positive``
     - :ref:`Sequence: decision <dataset-sequence-decision>`
     - Integrate the evidence of the channel a context cue selects.
   * - ``context_dependent_decision_hard``,
       ``context_dependent_decision_hard_positive``
     - :ref:`Sequence: decision <dataset-sequence-decision>`
     - Another name for the task above, with the same defaults.
   * - ``switching_context_decision``, ``switching_context_decision_positive``
     - :ref:`Sequence: decision <dataset-sequence-decision>`
     - Per-step decisions while the relevant channel switches in long blocks.
   * - ``timescale_generalization``, ``timescale_generalization_positive``
     - :ref:`Sequence: decision <dataset-sequence-decision>`
     - Delayed match-to-sample whose delay range can differ between splits.
   * - ``ready_set_go``, ``ready_set_go_positive``
     - :ref:`Sequence: timing <dataset-sequence-timing>`
     - Reproduce the Ready-Set interval with a Go time.
   * - ``contextual_ready_set_go``, ``contextual_ready_set_go_positive``
     - :ref:`Sequence: timing <dataset-sequence-timing>`
     - Two competing Ready-Set streams; a context cue selects the one to
       reproduce.
   * - ``bayesian_ready_set_go``, ``bayesian_ready_set_go_positive``
     - :ref:`Sequence: timing <dataset-sequence-timing>`
     - Interval reproduction with a prior cue; targets are posterior-
       predictive Go-time distributions.
   * - ``factual_bayesian_ready_set_go``,
       ``factual_bayesian_ready_set_go_positive``
     - :ref:`Sequence: timing <dataset-sequence-timing>`
     - Interval reproduction with a prior cue and one sampled Go step per
       trial.
   * - ``odr_generalization``, ``odr_generalization_positive``
     - :ref:`Sequence: neuroscience tasks <dataset-sequence-neuro>`
     - Oculomotor delayed response: report a cued angle after a delay.
   * - ``romo_delay_comparison``, ``romo_delay_comparison_positive``
     - :ref:`Sequence: neuroscience tasks <dataset-sequence-neuro>`
     - Compare two vibrotactile-like frequencies separated by a delay.
   * - ``gain_modulated_contextual_comparison``,
       ``gain_modulated_contextual_comparison_positive``
     - :ref:`Sequence: neuroscience tasks <dataset-sequence-neuro>`
     - Cued delayed comparison under global and pathway-specific gain.
   * - ``multi_frequency``
     - :ref:`Sequence: signals <dataset-sequence-signal>`
     - Classify the dominant frequency of a noisy mixture of sinusoids.
   * - ``multi_sine_forecast``
     - :ref:`Sequence: signals <dataset-sequence-signal>`
     - Next-step forecasting of a quasi-periodic signal (regression).
   * - ``lorenz``, ``lorenz_sequence``
     - :ref:`Sequence: signals <dataset-sequence-signal>`
     - Lorenz-attractor next-state prediction (regression); two names for one
       loader.
   * - ``switching_lds``
     - :ref:`Sequence: signals <dataset-sequence-signal>`
     - Partially observed switching linear dynamical system (regression).
   * - ``stringer_v1``
     - :ref:`Neural recordings <dataset-recordings>`
     - Mouse V1 responses to drifting gratings from a local copy of the
       recordings; orientation classification.
   * - ``contextual_remapping``
     - :ref:`Registered <dataset-registered>`
     - EMNIST Balanced with context-shifted labels and per-presentation gain
       and noise corruption; needs ``training.main.strategy:
       presentation_stream``.

.. _dataset-splits:

Training, validation and test splits
------------------------------------

Every loader returns three splits. The test split is the dataset's own test
set: the MNIST and CIFAR test sets, the official ImageNet validation set, or a
separately generated test set.

Training and validation
~~~~~~~~~~~~~~~~~~~~~~~

``experiment.train_valid_split`` (default ``0.8``) is the fraction of the
training data kept for training; the rest forms the validation split, which
selects the best checkpoint (``training.main.common.load_best_state_dict``)
and drives early stopping. The partition is random, seeded by
``experiment.split_seed``. It applies to:

- the vision, paired arithmetic, Poisson and ImageNet datasets (the paired
  datasets split their base images before pairing them);
- sequence datasets that generate one training set and one test set, which is
  every sequence dataset without explicit splits (below). The
  counterfactually paired datasets (``contextual_ready_set_go`` and
  ``bayesian_ready_set_go``) keep both members of a pair in the same split;
- ``stringer_v1``, whose fixed test split is 10% of the trials.

It does not apply to:

- the synthetic datasets. The first ten split one generated set 60/20/20 into
  training, validation and test; ``contextual_stream_gain_shift``,
  ``branch_local_gain_load`` and ``hierarchical_gain_load`` generate each
  split separately (see :ref:`dataset-synthetic`);
- sequence datasets with explicit splits;
- registered datasets such as ``contextual_remapping``, which define their own
  splits.

The value must lie in ``[0, 1]``; other values raise when the configuration
loads, and ``0`` raises when the datasets are built for every dataset except
registered ones. At ``1.0`` the datasets above have an empty validation
split. A run then stops during setup when
``training.main.common.load_best_state_dict`` or ``early_stopping`` is true
(both are by default), because both select on the validation loss; with both
false, the validation loss is recorded as NaN.

Explicit splits for sequence tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several sequence builders accept ``train``, ``valid`` and ``test`` mappings
inside their ``dataset_params`` block. Each mapping overrides task parameters
for one split, and ``experiment.train_valid_split`` is then not used. This is
how a test split can use delays that the training split never sees:

.. code-block:: yaml

   data:
     dataset_name: timescale_generalization
     dataset_params:
       timescale_generalization:
         n_samples: 20000
         train: {min_delay: 10, max_delay: 50}
         valid: {min_delay: 10, max_delay: 50}
         test: {min_delay: 60, max_delay: 100}

A split without a mapping uses the common values. ``n_samples`` in a mapping
sets the size of that split; by default the training split has ``n_samples``
examples and the validation and test splits have
``max(n_samples // 5, 2000)``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Datasets
     - Explicit splits
   * - ``timescale_generalization``, ``odr_generalization``,
       ``romo_delay_comparison``, ``gain_modulated_contextual_comparison``
       (and their ``_positive`` variants)
     - Optional. Any task parameter may differ between splits.
   * - ``ready_set_go``, ``contextual_ready_set_go``,
       ``bayesian_ready_set_go`` (and their ``_positive`` variants)
     - Optional. Each split draws its own seed (``seed``, ``seed + 1``,
       ``seed + 2`` unless a mapping sets ``seed``; the seeds must differ).
       The sequence length, Set-time bounds and, for the contextual and
       Bayesian tasks, the context timing and observer parameters must be the
       same in every split.
   * - ``speeded_distractor_dms``, ``factual_bayesian_ready_set_go``
       (and their ``_positive`` variants)
     - Always. The splits are generated separately, with seeds ``seed``,
       ``seed + 1`` and ``seed + 2``, even without mappings.
       ``speeded_distractor_dms`` mappings may set only ``n_samples`` (or
       ``n_pairs``) and ``seed``; ``factual_bayesian_ready_set_go`` mappings
       may also change the cue gains, input noise, distractors, context gain,
       ``motor_seed`` and ``nested_distractor_seed``.

Sealed test split
~~~~~~~~~~~~~~~~~

These ``experiment`` options protect the test split of a confirmatory run:

``max_test_samples`` (default ``null``)
   Upper bound on the size of the built test split. A larger test split
   raises; nothing is subsampled. It applies whenever it is set.

``sealed_test`` (default ``false``)
   Seals the test split. It requires ``allow_test_data: false`` and
   ``max_test_samples``, and training then refuses to start while any
   analysis is enabled, since analyzers read the test split.

``allow_test_data`` (default ``true``)
   ``false`` is part of the seal and is accepted only with
   ``sealed_test: true``.

``record_dataset_fingerprints`` (default ``false``)
   Writes ``dataset_fingerprints.json`` to the run directory: content hashes
   (SHA-256) of the tensors and stored attributes of the training, validation
   and test datasets, and one combined hash. For file-backed datasets such as
   ImageNet the hash covers the stored file list, not the image bytes.

A sealed run on a generated dataset whose test split has 400 examples:

.. code-block:: yaml

   experiment:
     sealed_test: true
     allow_test_data: false
     max_test_samples: 400
     record_dataset_fingerprints: true
   data:
     dataset_name: correlated_gaussian
     dataset_params:
       correlated_gaussian:
         n_samples: 1000

The check that rejects enabled analyses is part of training:
``dendritic-analyze-checkpoint`` runs the analyses it is given on a saved
checkpoint even when its training configuration seals the test split (see
:ref:`checkpoint-analysis`).

.. _dataset-location:

Where data are stored
---------------------

The data root is ``data.base_dir`` when it is set; otherwise
``$DENDRITIC_DATA_DIR``; otherwise ``$XDG_CACHE_HOME/dendritic_modeling``
(``~/.cache/dendritic_modeling`` when ``XDG_CACHE_HOME`` is unset), so runs
share one copy. ``dendritic-train --data_dir <path>`` overrides
``data.base_dir``. Each run logs the data root and records
``DENDRITIC_DATA_DIR`` in ``runtime_environment.json``.

Torchvision datasets are downloaded on first use into subdirectories of the
root:

- ``<root>/datasets/mnist``: MNIST, also used by ``sequential_mnist``, the
  MNIST paired datasets and the Poisson generator on MNIST;
- ``<root>/datasets/fashion_mnist``: Fashion-MNIST;
- ``<root>/datasets/cifar``: CIFAR-10 and CIFAR-100, also used by the CIFAR-10
  paired datasets and the Poisson generator on CIFAR-10.

Processes that load the same directory at once take turns through a
``.download.lock`` file in it. A missing or corrupted copy is deleted and
downloaded again.

Three datasets read local files and never download by default: ImageNet (see
:ref:`dataset-imagenet`), ``stringer_v1`` (see :ref:`dataset-recordings`)
and ``contextual_remapping``, which downloads EMNIST only with
``download: true`` (see :ref:`dataset-registered`).

.. _dataset-processing:

Input processing
----------------

``data.processing`` has three options. A written ``true`` that the selected
loader cannot apply raises instead of being ignored.

``flatten`` (default: unset, which loaders treat as ``true``)
   Return inputs as vectors. Vision, paired and Poisson datasets flatten when
   it is unset; ImageNet then raises, so ImageNet runs set ``flatten: false``.
   Synthetic datasets are already flat and accept ``true``. Sequence and
   registered datasets keep their own layout and raise on ``true``.

``normalize`` (default: unset, which loaders treat as ``false``)
   Dataset-specific input scaling. For the vision, paired and ImageNet
   datasets, ``true`` subtracts the per-channel mean and divides by the
   standard deviation (torchvision ``Normalize`` with the dataset's
   statistics). ``sequential_mnist`` divides pixel values by 255 and
   ``stringer_v1`` z-scores every neuron, both unless ``normalize: false``.
   The synthetic, Poisson, other sequence and ``contextual_remapping``
   datasets have no input scaling and raise on ``true``.

``label_noise_rate`` (float in ``[0, 1]``, default ``0.0``)
   Fraction of training labels replaced by a class drawn uniformly from all
   classes, including the original one, so about
   ``label_noise_rate * (C - 1) / C`` of the labels change for ``C`` classes.
   Only the training split changes. The draws use ``experiment.dataset_seed``.
   Only ``mnist``, ``fashion_mnist``, ``cifar10`` and ``cifar100`` apply it; a
   positive rate for any other dataset raises.

.. code-block:: yaml

   data:
     dataset_name: cifar10
     processing:
       flatten: true
       normalize: true
       label_noise_rate: 0.2

Shunting models with positive weight transforms expect non-negative input to
their first shunting layer. ``normalize: true`` makes the inputs signed: keep
``normalize: false``, or rectify the transfer layer of a structured E/I core
with ``model.core.transfer.output_activation`` set to ``relu``, ``softplus``
or ``sigmoid``.

.. _dataset-vision:

Vision datasets
---------------

``mnist``, ``fashion_mnist``, ``cifar10`` and ``cifar100`` load the
torchvision datasets into memory as tensors with values in ``[0, 1]``
(before ``normalize``). With the default ``flatten`` the inputs are vectors of
784 (MNIST, Fashion-MNIST) or 3072 (CIFAR) values; with ``flatten: false``
they keep the ``[channels, height, width]`` shape. They read no
``dataset_params`` keys.

.. _dataset-paired:

Paired arithmetic datasets
--------------------------

Each split is built from the matching split of the base dataset: the images
are permuted and the first half is paired with the second half, so one pass
gives ``n // 2`` pairs. ``shuffle_iterations`` repeats the pairing with
further permutations and multiplies the split size accordingly.

``mnist_modulo10``
   Concatenates the two images along the last axis (a 1568-value vector by
   default) and labels the pair ``(a + b) mod 10``. Options:
   ``shuffle_iterations: 2``, ``pair_seed: 0``.

``cifar10_modulo10``
   Labels a pair of CIFAR-10 images ``(a + b) mod 10``. With the default
   ``flatten`` the input is the two flattened images; with ``flatten: false``
   the images sit side by side. Options: ``shuffle_iterations: 1``,
   ``pair_seed: 0``.

``double_mnist_contextual_mod10``, ``double_cifar10_contextual_mod10``
   Draw a random context for every pair (seeded by ``context_seed``). Context
   0 labels the pair ``(a + b) mod 10`` and context 1 ``(a - b) mod 10``. The
   context is appended as a two-value one-hot vector, or as two constant image
   planes with ``flatten: false``. Options: ``shuffle_iterations: 1``,
   ``pair_seed: 0``, ``context_seed: 123``.

.. code-block:: yaml

   data:
     dataset_name: double_mnist_contextual_mod10
     dataset_params:
       double_mnist_contextual_mod10:
         shuffle_iterations: 2
         pair_seed: 0
         context_seed: 123

.. _dataset-poisson:

Poisson generator
-----------------

``poisson_generator`` treats the pixel intensities of a base dataset as rates.
Every access draws fresh noise, so each epoch and each evaluation sees
different samples:

.. math::

   \tilde{x} = \mathrm{Poisson}(T\, g\, x) / T,

where ``T = stimulus_duration`` and ``g`` is the gain. Options of
``data.dataset_params.poisson_generator``:

``base_dataset`` (default ``mnist``)
   ``mnist`` or ``cifar10``. Any other name raises. The base images are not
   normalized.

``poisson_sampling`` (default ``true``)
   ``false`` returns ``g x`` without noise.

``stimulus_duration`` (default ``1.0``)
   ``T``: longer stimuli give more counts and relatively less noise.

``multiplicative_gain`` (default ``true``)
   Apply the gain ``g``. With ``false`` the gain options are not used.

``fixed_gain_factor`` (default ``null``)
   One gain for every input. It takes precedence over the bounds below.

``max_gain_factor`` (default ``null``)
   Draw ``g`` from ``[1, max_gain_factor]``; it must be at least 1.

``max_gain_tau_ratio`` (default ``null``)
   Sets the upper bound to ``max_gain_tau_ratio * stimulus_duration`` when
   ``max_gain_factor`` is not set.

``gain_sampling`` (default ``log_uniform``)
   ``log_uniform`` draws ``log g`` uniformly; ``linear_uniform`` draws ``g``
   uniformly.

``uniform_gain`` (default ``true``)
   ``true`` draws one gain per sample; ``false`` draws one gain per input
   element and needs a fixed gain or an upper bound.

With the defaults no bound is set and the gain is 1, so only Poisson noise is
added. Of the processing options only ``flatten`` applies; ``normalize: true``
and label noise raise.

.. code-block:: yaml

   data:
     dataset_name: poisson_generator
     dataset_params:
       poisson_generator:
         base_dataset: mnist
         max_gain_factor: 10.0
         stimulus_duration: 1.0

.. _dataset-synthetic:

Synthetic datasets
------------------

Synthetic datasets are generated in memory from ``experiment.dataset_seed``
(or the block's ``seed``). Their inputs are flat vectors.

- The first ten datasets below generate ``n_samples`` examples per class,
  with inputs in ``[0, 1]``, shuffle them and split the set 60/20/20 into
  training, validation and test. ``nonlinear_interaction``,
  ``nonlinear_separable`` and ``population_code`` assign classes after
  sampling and drop examples of the larger class to balance them, so their
  total can be smaller than ``n_samples`` times the number of classes. Inputs
  combine ``ne`` excitatory-designated and ``ni`` inhibitory-designated
  features, and ``input_dim`` must equal ``ne + ni`` where both appear.
- ``contextual_stream_gain_shift``, ``branch_local_gain_load`` and
  ``hierarchical_gain_load`` generate the training, validation and test
  splits separately, with about ``n_samples`` examples each. Their test split
  is drawn from a shifted gain distribution, and so is their validation split
  unless ``valid_split_mode: train``: checkpoint selection and early stopping
  then see test-distribution data. Leaving ``valid_split_mode`` unset logs a
  warning; set it explicitly.

The options and their defaults (every block also accepts ``seed``):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Dataset
     - Options (defaults)
   * - ``correlated_gaussian``
     - ``n_samples: 1000``, ``input_dim: 100``, ``n_classes: 2``,
       ``rho_ee: 0.0``, ``rho_ii: 0.0``, ``rho_ei: 0.0``, ``ne: 50``,
       ``ni: 50``, ``class_separation: 0.3``, ``noise_std: 0.2``
   * - ``linear_separable``
     - ``n_samples: 1000``, ``input_dim: 100``, ``ne: 50``, ``ni: 50``,
       ``rho_ee: 0.0``, ``rho_ii: 0.0``, ``rho_ei: 0.0``,
       ``signal_strength: 0.3``, ``noise_std: 0.2``
   * - ``nonlinear_separable``
     - ``n_samples: 1000``, ``input_dim: 100``, ``ne: 50``, ``ni: 50``,
       ``nonlinearity_type: quadratic`` (``quadratic``, ``xor``,
       ``multiplicative``, ``radial``), ``rho_ee: 0.0``, ``rho_ii: 0.0``,
       ``rho_ei: 0.0``, ``noise_std: 0.2``
   * - ``nonlinear_interaction``
     - ``n_samples: 1000``, ``input_dim: 2``, ``n_classes: 2``,
       ``interaction_strength: 1.0``, ``noise_std: 0.1``
   * - ``block_correlated``
     - ``n_samples: 1000``, ``input_dim: 100``, ``ne: 50``, ``ni: 50``,
       ``n_blocks: 5``, ``block_rho_ee: [0.0, 0.2, -0.1]``,
       ``block_rho_ii: [0.0, 0.2, -0.1]``, ``block_rho_ei: [0.0, 0.3, 0.5]``,
       ``class_separation: 0.3``, ``noise_std: 0.2``
   * - ``balanced_information``
     - ``n_samples: 1000``, ``input_dim: 100``, ``ne: 50``, ``ni: 50``,
       ``info_per_dim: balanced`` (``balanced``, ``concentrated``,
       ``gradient``), ``rho_ee: 0.0``, ``rho_ii: 0.0``, ``rho_ei: 0.0``,
       ``noise_std: 0.2``
   * - ``heterogeneous_information``
     - ``n_samples: 2000``, ``input_dim: 100``, ``n_classes: 10``, ``ne: 50``,
       ``ni: 50``, ``n_high_info_dims: 10``, ``high_info_strength: 2.0``,
       ``low_info_strength: 0.1``, ``noise_std: 0.2``, ``rho_ee: 0.0``,
       ``rho_ii: 0.0``, ``rho_ei: 0.0``
   * - ``nonlinear_transformation``
     - ``n_samples: 2000``, ``n_classes: 2``, ``nonlinearity_type: xor``
       (``xor``, ``quadratic``, ``multiplicative``), ``ne: 2``, ``ni: 2``,
       ``noise_std: 0.1``
   * - ``asymmetric_noise``
     - ``n_samples: 2000``, ``input_dim: 100``, ``n_classes: 10``, ``ne: 50``,
       ``ni: 50``, ``sigma_e: 0.2``, ``sigma_i: 0.1``,
       ``class_separation: 0.2``, ``rho_ee: 0.0``, ``rho_ii: 0.0``,
       ``rho_ei: 0.0``
   * - ``population_code``
     - ``n_samples: 2000``, ``pop_dim: 200``, ``ne: 50``, ``ni: 50``,
       ``n_classes: 2``, ``code_type: mixed_selectivity``
       (``mixed_selectivity``, ``ring``), ``noise_std: 0.1``, ``kappa: 2.0``
   * - ``contextual_stream_gain_shift``
     - ``n_samples: 4000``, ``stream_dim: 64``, ``n_classes: 2``,
       ``signal_strength: 0.35``, ``relevant_noise_std: 0.08``,
       ``irrelevant_noise_std: 0.18``, ``train_gain_relevant_min: 0.9``,
       ``train_gain_relevant_max: 1.1``, ``train_gain_irrelevant_min: 0.8``,
       ``train_gain_irrelevant_max: 1.2``, ``test_gain_relevant: 1.0``,
       ``test_gain_irrelevant: 3.0``, ``test_irrelevant_alignment_alpha: 1.0``,
       ``ood_mode: irrelevant`` (``relevant``, ``irrelevant``, ``both``),
       ``valid_split_mode: ood`` (``ood``, ``train``),
       ``context_signal_scale: 1.0``
   * - ``branch_local_gain_load``
     - ``n_samples: 6000``, ``stream_dim: 64``, ``signal_fraction: 0.5``,
       ``n_gain_groups: 8``, ``n_classes: 2``, ``e_baseline: 5.0``,
       ``i_baseline: 2.0``, ``e_signal_delta: 1.0``, ``i_signal_delta: 0.0``,
       ``signal_mode: e_only`` (``e_only``, ``balanced_ratio``),
       ``support_mode: prefix`` (``prefix``, ``sampled``),
       ``train_gain_support_mode: signal``, ``gain_fraction: -1.0``,
       ``load_fraction: 1.0``, ``load_alignment_alpha: 1.0``,
       ``independent_noise_std: 0.05``, ``train_gain_sigma: 0.8``,
       ``test_gain_sigma: 0.8``, ``gain_alignment_alpha: 1.0``,
       ``load_mean: 0.0``, ``load_noise_sigma: 0.0``,
       ``valid_split_mode: test`` (``test``, ``train``)
   * - ``hierarchical_gain_load``
     - ``n_samples: 6000``, ``stream_dim: 64``, ``n_levels: 3``,
       ``hierarchy_branching: 2``, ``n_flat_groups: 8``, ``n_classes: 2``,
       ``e_baseline: 5.0``, ``i_baseline: 2.0``, ``e_signal_delta: 0.35``,
       ``i_signal_delta: 0.35``, ``signal_mode: e_only``,
       ``signal_profile: all``, ``nuisance_layout: factorized_sensors``,
       ``sensor_e_baseline: 0.02``, ``gain_structure: hierarchical``,
       ``gain_scale_decay: 1.0``, ``train_gain_sigma: 0.4``,
       ``test_gain_sigma: 1.4``, ``sensor_alignment_alpha: 1.0``,
       ``sensor_support_mode: matched``, ``private_gain_sigma: 0.0``,
       ``independent_noise_std: 0.08``, ``load_mean: 0.0``,
       ``load_noise_sigma: 0.0``, ``valid_split_mode: test`` (``test``,
       ``train``)

``contextual_stream_gain_shift`` inputs are ``[stream A | stream B | context]``
(``2 * stream_dim + 2`` values in ``[0, 1]``); the gains change contrast around
0.5. The two gain-load datasets produce positive ``[E | I]`` streams
(``2 * stream_dim`` values) for a transfer layer with independent pathways.
In ``branch_local_gain_load`` a mean-one log-normal gain, with log standard
deviation ``train_gain_sigma`` in the training split and ``test_gain_sigma``
in the test split, multiplies groups of neighbouring features of both
streams. With the defaults both are 0.8 on the same features, so there is no
train/test shift; set different values, or ``gain_alignment_alpha`` below 1 to
move part of the test gain off the class-signal features.
``hierarchical_gain_load`` shifts by default (0.4 in training, 1.4 in test).
The generator docstrings in ``dendritic_modeling.datasets.synthetic_datasets``
define every option.

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

.. _dataset-sequence:

Sequence datasets
-----------------

Sequence datasets return inputs of shape ``[time, features]`` for recurrent
cores (a recurrent ``population_network`` or one of the recurrent cores in
:ref:`core-types`). The target is a class label, a per-step label sequence, a
distribution over time steps, or a regression target, as listed per dataset.
Tasks with variable trial lengths also return each trial's length. Regression
tasks (``adding_problem``, ``lorenz``, ``lorenz_sequence``,
``multi_sine_forecast``, ``switching_lds``) need ``model.task: regression``.

Shared conventions:

- **Sizes.** ``n_samples`` is the size of the training data before the
  validation split. The test split has ``max(n_samples // 5, minimum)``
  examples, with a minimum of 1000 for ``copy_task``, ``adding_problem``,
  ``variable_delay_dms`` and the signal tasks and 2000 for the others.
  ``sequential_mnist`` and ``stringer_v1`` use the recorded data instead.
- **Seeds.** ``seed`` defaults to ``experiment.dataset_seed``. Tasks that
  build their test split from a second seed default it to ``seed + 1``
  (``test_seed`` where listed).
- **Positive encoding.** A ``_positive`` name selects inputs that are never
  negative, for shunting models with positive conductances. Setting
  ``positive_input_encoding: true`` under the base name selects the same
  encoding.
- **Processing.** ``flatten`` and ``normalize`` raise when ``true`` (except
  ``normalize`` for ``sequential_mnist`` and ``stringer_v1``), and label noise
  raises.

.. code-block:: yaml

   data:
     dataset_name: context_dependent_decision
     dataset_params:
       context_dependent_decision:
         n_samples: 20000
         signal_dim: 8
         noise_std: 1.0
         variable_integration: true
         min_integration: 20
         max_integration: 80

.. _dataset-sequence-memory:

Memory tasks
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Dataset
     - Input and target
     - Options (defaults)
   * - ``sequential_mnist``
     - ``[784, 1]`` (``mode: pixel``) or ``[28, 28]`` (``mode: row``);
       class of 10
     - ``mode: pixel``, ``normalize: true``. Uses the MNIST training set
       (60000 images) and test set (10000); it has no ``n_samples`` option.
   * - ``copy_task``
     - ``[2 * seq_len + delay, n_symbols + 2]``; per-step symbol labels, with
       ``-100`` (ignored) outside the recall phase
     - ``n_samples: 10000``, ``seq_len: 10``, ``delay: 50``,
       ``n_symbols: 8``
   * - ``adding_problem``
     - ``[seq_len, 2]`` (values and marker); target ``[1]``, the sum of the
       two marked values (regression)
     - ``n_samples: 10000``, ``seq_len: 200``
   * - ``variable_delay_dms``
     - stimulus, delay, probe; binary match label
     - ``n_samples: 10000``, ``stimulus_dim: 16``, ``stimulus_duration: 10``,
       ``min_delay: 10``, ``max_delay: 100``, ``probe_duration: 10``
   * - ``speeded_distractor_dms``
     - ``[sequence_duration, features]``; target ``[time, 4]`` with the
       planes ``fast_label``, ``fast_mask``, ``memory_label``,
       ``memory_mask``, for ``loss_function: masked_dual_task_temporal``
     - ``n_samples: 10000`` (even; or ``n_pairs``),
       ``delay_support: [128, 160, 192, 224]``, ``n_distractors: 4``,
       ``lure_fraction: 0.25``,
       ``distractor_identity_mode: sample_conditioned_lures``,
       ``salience_mu: 0.8``, ``circular_noise_std: 0.04``,
       ``sequence_duration: 360``, ``intervention_seed: null``. Always
       builds explicit splits; validation and test default to
       ``max(n_samples // 5, 2000)`` (rounded up to even).

.. _dataset-sequence-decision:

Decision tasks
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Dataset
     - Input and target
     - Options (defaults)
   * - ``hierarchical_temporal``
     - context cue, then ``n_blocks`` noisy blocks; binary label (does the
       context's target block type occur more often than average)
     - ``n_samples: 20000``, ``n_contexts: 2``, ``n_block_types: 4``,
       ``n_blocks: 8``, ``context_duration: 20``, ``block_duration: 25``,
       ``signal_dim: 16``, ``noise_std: 0.5``
   * - ``context_dependent_decision``
     - context cue and two evidence channels (``2 + 2 * signal_dim``
       features, more with the positive encoding); binary label (sign of the
       relevant channel's evidence)
     - ``n_samples: 20000``, ``signal_dim: 8``, ``context_duration: 10``,
       ``coherence_levels: [0.05, 0.1, 0.2, 0.4]``, ``noise_std: 1.0``,
       ``variable_integration: true``, ``min_integration: 20``,
       ``max_integration: 80``, ``integration_duration: 50``,
       ``delay_duration: 0``, ``min_delay: null``, ``max_delay: null``,
       ``conflict_fraction: 0.0``, ``distractor_burst_prob: 0.0``,
       ``distractor_burst_scale: 2.0``, ``distractor_burst_duration: 5``,
       ``cue_dropout_prob: 0.0``, ``cue_keep_duration: 1``,
       ``late_context_switch_prob: 0.0``, ``context_signal_scale: 1.0``,
       ``return_seq_lengths: false``
   * - ``switching_context_decision``
     - ``seq_len`` steps of context hint and two evidence channels; one
       binary label per step
     - ``n_samples: 16000``, ``signal_dim: 8``, ``seq_len: 360``,
       ``trial_duration: 12``, ``min_context_duration: 90``,
       ``max_context_duration: 150``,
       ``coherence_levels: [0.03, 0.06, 0.12, 0.24]``, ``noise_std: 1.1``,
       ``conflict_fraction: 0.75``, ``distractor_burst_prob: 0.35``,
       ``distractor_burst_scale: 2.0``, ``distractor_burst_duration: 3``,
       ``context_cue_duration: 10``, ``switch_cue_duration: 4``,
       ``context_signal_scale: 1.0``, ``switch_signal_scale: 0.35``,
       ``min_blocks: 2``
   * - ``timescale_generalization``
     - stimulus, delay, probe; binary match label
     - ``n_samples: 20000``, ``stimulus_dim: 16``, ``stimulus_duration: 10``,
       ``min_delay: 10``, ``max_delay: 100``, ``probe_duration: 10``,
       ``noise_std: 0.3``; explicit splits (see :ref:`dataset-splits`)

In ``context_dependent_decision``, ``variable_integration: true`` draws each
trial's integration length from ``[min_integration, max_integration]`` and
ignores ``integration_duration`` (with a warning); ``false`` uses
``integration_duration``. The delay after integration is set by
``delay_duration`` or by ``min_delay``/``max_delay``. The option's former
name ``variable_delay`` still loads, with a warning. For a harder task, raise
``conflict_fraction``, ``noise_std`` or the ``distractor_burst_*`` options;
the ``_hard`` names build the same task with the same defaults.

.. _dataset-sequence-timing:

Interval timing tasks
~~~~~~~~~~~~~~~~~~~~~

In the Ready-Set-Go tasks a Ready pulse and a Set pulse are separated by a
sampled interval, and the model reproduces the interval after Set. The target
is the Go time, for ``training.main.common.loss_function: temporal_event_ce``:
a normalized distribution over the time steps (``ready_set_go``,
``contextual_ready_set_go``, ``bayesian_ready_set_go``) or one Go step
(``factual_bayesian_ready_set_go``). The sequence length is fixed for all
trials; when ``sequence_duration`` is unset it is the shortest length that
fits every trial. ``set_min_step`` and ``set_max_step`` bound the Set time and
default to values derived from the intervals.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Dataset
     - Input and target
     - Options (defaults)
   * - ``ready_set_go``
     - 3 channels (Ready, Set, distractor pulses); Go-time distribution
     - ``n_samples: 10000``,
       ``sample_intervals: [40, 50, 60, 70, 80, 90, 100]``,
       ``production_scale: 1.0``, ``pulse_duration: 2``,
       ``target_sigma: 2.0``, ``post_go_duration: 20``,
       ``cue_gain_min: 0.7``, ``cue_gain_max: 1.3``,
       ``input_noise_std: 0.05``, ``mean_distractors_per_trial: 1.0``,
       ``distractor_amplitude: 1.0``, ``distractor_pulse_duration: 2``,
       ``set_min_step``, ``set_max_step``, ``sequence_duration``
   * - ``contextual_ready_set_go``
     - 7 channels (two Ready-Set streams, two context channels,
       distractors); Go-time distribution; counterfactual pairs
     - the ``ready_set_go`` options and ``context_start_step: 0``,
       ``context_duration: 10``, ``context_gain: 1.0``,
       ``minimum_competing_go_separation: 10``
   * - ``bayesian_ready_set_go``
     - 5 channels (Ready, Set, short- and long-prior cues, distractors);
       posterior-predictive Go-time distribution; counterfactual pairs
       (``n_samples`` even)
     - ``n_samples: 10000``,
       ``prior_supports: [[24, 28, 32, 36, 40], [40, 45, 50, 55, 60]]``,
       ``prior_probabilities: null``, ``weber_fraction: 0.12``,
       ``measurement_noise_floor: 0.0``, ``likelihood_tail_sigma: 4.0``,
       ``target_sigma: 2.0``, ``target_tail_sigma: 4.0``,
       ``observed_interval_min``, ``observed_interval_max`` (derived from
       the priors), ``production_scale``, ``pulse_duration``,
       ``context_start_step``, ``context_duration``, ``context_gain``,
       ``post_go_duration``, ``cue_gain_min``, ``cue_gain_max``,
       ``input_noise_std``, ``mean_distractors_per_trial``,
       ``distractor_amplitude`` and ``distractor_pulse_duration`` (defaults
       as for ``contextual_ready_set_go``), and ``set_min_step``,
       ``set_max_step`` and ``sequence_duration`` (derived when unset)
   * - ``factual_bayesian_ready_set_go``
     - 5 channels as above; one Go step per trial
     - the ``bayesian_ready_set_go`` options without the target widths, and
       ``motor_sigma: 2.0``, ``motor_tail_sigma: 4.0``,
       ``post_go_duration: 0``, ``motor_seed: null``,
       ``set_step_values: null``, ``nested_distractor_pairs_per_trial: 0``,
       ``nested_distractor_interval_values: [3, 4, 5]``,
       ``nested_distractor_clearance: 2``, ``nested_distractor_seed: null``.
       Always builds explicit splits.

.. _dataset-sequence-neuro:

Neuroscience-inspired tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three accept explicit splits (see :ref:`dataset-splits`).

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Dataset
     - Input and target
     - Options (defaults)
   * - ``odr_generalization``
     - fixation, angular cue, delay, go; class of ``n_targets``
     - ``n_samples: 20000``, ``n_targets: 8``, ``input_tuning_dim: 16``,
       ``fixation_duration: 6``, ``cue_duration: 8``, ``min_delay: 10``,
       ``max_delay: 100``, ``go_duration: 8``, ``tuning_kappa: 2.0``,
       ``cue_noise_std: 0.05``
   * - ``romo_delay_comparison``
     - first frequency, delay, second frequency, response; binary label
       (second frequency higher)
     - ``n_samples: 20000``,
       ``frequency_values: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]``,
       ``stimulus_duration: 20``, ``min_delay: 10``, ``max_delay: 100``,
       ``response_duration: 8``, ``noise_std: 0.08``, ``tuning_dim: 8``,
       ``tuning_sigma: null``, ``pair_sampling: independent_unequal_v1``
   * - ``gain_modulated_contextual_comparison``
     - pathway cue, two stimuli on every pathway separated by a delay;
       binary label (the cued pathway's second value is larger)
     - ``n_samples: 24000``, ``n_pathways: 4``, ``pathway_dim: 4``,
       ``cue_duration: 8``, ``stimulus_duration: 16``, ``min_delay: 20``,
       ``max_delay: 80``, ``response_duration: 8``,
       ``amplitude_values: [0.45, 0.6, 0.75, 0.9, 1.05, 1.2]``,
       ``comparison_margin: 0.15``, ``global_gain_min: 0.85``,
       ``global_gain_max: 1.15``, ``pathway_gain_min: 0.8``,
       ``pathway_gain_max: 1.2``, ``gain_drift_std: 0.03``,
       ``noise_std: 0.08``, ``cue_noise_std: 0.02``,
       ``distractor_burst_prob: 0.35``, ``distractor_burst_scale: 1.5``,
       ``distractor_burst_duration: 4``

.. _dataset-sequence-signal:

Signal and dynamics tasks
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Dataset
     - Input and target
     - Options (defaults)
   * - ``multi_frequency``
     - ``[seq_len, 1]``; class of the dominant frequency
     - ``n_samples: 10000``, ``seq_len: 200``, ``n_frequencies: 5``,
       ``base_freq: 0.01``, ``noise_std: 0.1``
   * - ``multi_sine_forecast``
     - ``[seq_len, 1]``; next-step targets ``[seq_len, 1]``
       (``prediction_mode: all``) or the next value (``last``)
     - ``n_samples: 12000``, ``seq_len: 160``, ``rollout_horizon: 160``,
       ``prediction_mode: all``, ``n_components: 3``,
       ``frequency_bank: [1.0, 2.5, 5.5, 9.0, 14.0]``,
       ``amplitude_min: 0.3``, ``amplitude_max: 1.0``,
       ``envelope_scale: 0.15``, ``envelope_freq_min: 0.15``,
       ``envelope_freq_max: 0.6``, ``observation_noise_std: 0.02``,
       ``normalize_signal: true``, ``test_seed``
   * - ``lorenz``, ``lorenz_sequence``
     - ``[seq_len, observation_dim]``; next state ``[3]``
       (``prediction_mode: last``) or per-step next states (``all``)
     - ``n_samples: 10000``, ``seq_len: 50``, ``dt: 0.01``, ``sigma: 10.0``,
       ``rho: 28.0``, ``beta: 2.6666666666666665`` (8/3),
       ``rollout_horizon: 50``, ``prediction_mode: last``,
       ``observation_mode: full``, ``observation_dim: 3``,
       ``observation_seed`` (default ``seed``),
       ``observation_noise_std: 0.0``, ``burn_in_steps: 200``,
       ``integration_method: euler``, ``normalize_states: true``,
       ``normalize_observations: true``, ``normalization_steps: 10000``,
       ``initial_state_scale: 0.1``, ``test_seed``
   * - ``switching_lds``
     - ``[seq_len, observation_dim]``; per-step targets
       ``[seq_len, latent_dim]``
     - ``n_samples: 12000``, ``seq_len: 96``, ``latent_dim: 4``,
       ``observation_dim: 2``, ``n_modes: 3``, ``rollout_horizon: 80``,
       ``prediction_mode: all``, ``observation_noise_std: 0.03``,
       ``transition_noise_std: 0.02``, ``min_mode_duration: 36``,
       ``max_mode_duration: 96``, ``burn_in_steps: 24``,
       ``normalize_states: true``, ``normalize_observations: true``,
       ``normalization_samples: 4096``, ``stats_seed: 1234``,
       ``initial_state_scale: 0.35``, ``test_seed``

The dataset classes in ``dendritic_modeling.datasets.sequence`` document each
option. A dataset whose builder is not listed here, added with
``register_sequence_dataset_builder``, checks its own options.

.. _dataset-recordings:

Neural recordings: ``stringer_v1``
----------------------------------

``stringer_v1`` classifies the orientation of drifting gratings from mouse V1
population responses (Stringer et al., 2019, Science 364). Each example is a
static vector of ``n_neurons`` responses, not a sequence, so it suits
feedforward cores. The data are not downloaded: ``data_dir`` (default: the data
root) must contain the response file ``grating_file`` (default
``gratings_drifting_GT3_2019_04_05_1.npy``) and ``all_running.npy``;
``session_id`` selects another session by name and then also needs
``database.npy``.

``n_neurons`` (default ``500``)
   Number of neurons kept, ranked by response variance.

``n_classes`` (default ``8``)
   Number of orientation bins over ``[0, 2*pi)``.

``normalize`` (default ``true``)
   Z-score every neuron.

``neuron_selection_scope`` (default ``all_trials``)
   ``all_trials`` ranks neurons on all trials (transductive); ``train_split``
   ranks them on the training trials only and reuses that ranking for the test
   trials.

The test split is a fixed 10% of the trials drawn with the dataset seed, and
the validation split comes from the remaining trials through
``experiment.train_valid_split``. The block also accepts running-state
options (``running_state``, ``running_quantile``, ``train_running_state``,
``test_running_state``, ``state_holdout_fraction``,
``running_residualize``, ``running_residualize_clip``), gain augmentation
(``gain_group_count``, ``gain_grouping``, ``global_gain_min``/``max``,
``group_gain_min``/``max``, ``gain_seed``) and Poisson resampling
(``poisson_sampling``, ``poisson_rate_scale``). The gain, Poisson and
residualization options also take ``train_`` and ``test_`` prefixes to differ
between the training and test trials. Setting ``train_running_state`` and
``test_running_state`` to disjoint states (``low`` and ``high``) trains on one
running state and tests on the other; see ``StringerV1Dataset`` in
``dendritic_modeling.datasets.sequence.neuro_tasks``.

.. _dataset-registered:

Registered datasets
-------------------

``contextual_remapping``
~~~~~~~~~~~~~~~~~~~~~~~~

The context-cued EMNIST Balanced task (47 classes). Each presentation shows an
image with a random context ``c`` of ``num_contexts``; the label is
``(label + c) mod 47`` and the context is appended as a one-hot cue scaled by
``context_scale``. The image (not the cue) is corrupted by a log-uniform gain
in ``[gain_min, gain_max]`` and brightness-dependent Gaussian noise whose
divisor is log-uniform in ``[noise_divisor_min, noise_divisor_max]``.

It requires ``training.main.strategy: presentation_stream``: the training
split serves freshly corrupted presentations, and indexing it returns clean
context-0 items that only fix the input shape. It defines its own splits:
``pool_per_class`` training images per class are held out, the first
``validation_per_class`` of them form a fixed corrupted validation set, and
the test split is the corrupted official EMNIST test set. EMNIST is read from
``<root>/raw`` (``root`` defaults to the data root) and downloaded there only
with ``download: true``. Processing options raise.

Options and defaults: ``root: ""``, ``download: false``, ``upright: true``,
``peak_normalize: true``, ``pool_per_class: 200``,
``validation_per_class: 50``, ``partition_seed: 31003``, ``num_contexts: 32``,
``context_scale: 1.0``, ``gain_min: 0.05``, ``gain_max: 20.0``,
``noise_divisor_min: 1.0``, ``noise_divisor_max: 4.0``,
``validation_seed: 78019``, ``test_seed: 99173``, ``generator_device: cpu``
(``cuda`` draws the evaluation corruptions on a GPU). Unknown keys raise.

.. code-block:: yaml

   training:
     main:
       strategy: presentation_stream
   data:
     dataset_name: contextual_remapping
     dataset_params:
       contextual_remapping:
         download: true
         num_contexts: 32

The model configurations that use this task are listed in :ref:`examples-map`,
and :doc:`conductance_options` describes the conductance options they use.

Adding a dataset
~~~~~~~~~~~~~~~~

``dendritic_modeling.datasets.registry.register_dataset(name, builder, ...)``
adds a dataset selectable by ``data.dataset_name``. The builder receives the
task configuration (attributes ``dataset``, ``data_path``,
``train_valid_split`` and ``parameters``, the merged processing options,
dataset parameters and seeds) and returns ``(train, valid, test)``. Keyword
options declare a ``required_strategy`` and the ``processing`` options the
builder applies (others then raise). Registered names are looked up before the
built-in loaders, so register a name that no built-in dataset uses.
``dendritic_modeling.datasets.sequence.register_sequence_dataset_builder``
adds a sequence builder that the shared sequence factory splits. See
:ref:`extension-registries`.

.. _dataset-imagenet:

ImageNet
--------

``imagenet`` loads ILSVRC-2012 from a local copy; it is never downloaded, and
a missing copy raises with the resolved paths (there is no synthetic
fallback). Point ``data.base_dir`` at the ImageNet root:

.. code-block:: yaml

   data:
     dataset_name: imagenet
     base_dir: "/path/to/imagenet"
     processing:
       flatten: false
       normalize: true
     dataset_params:
       imagenet:
         transform_preset: pretrained
         loader_backend: auto

Two layouts are recognized in the data root, and otherwise in
``<root>/datasets/imagenet``:

- a torchvision ImageNet root: the devkit (``meta.bin``,
  ``ILSVRC2012_devkit_t12.tar.gz`` or ``ILSVRC2012_devkit_t12/``) with
  training (``train/`` or ``ILSVRC2012_img_train/``) and validation (``val/``
  or ``ILSVRC2012_img_val/``) images;
- class directories: ``train/<class>/`` (or ``ILSVRC2012_img_train/<class>/``)
  and ``val/<class>/``, loaded with torchvision ``ImageFolder``.

Options of ``data.dataset_params.imagenet``:

``loader_backend`` (default ``auto``)
   ``auto`` uses a torchvision root when one is found and class directories
   otherwise; ``torchvision`` forces ``torchvision.datasets.ImageNet``;
   ``imagefolder`` forces class directories and skips the devkit metadata.

``transform_preset`` (default ``legacy_resize``)
   ``legacy_resize`` resizes every image to 224x224. ``pretrained`` uses
   ``RandomResizedCrop(224)`` and horizontal flips for training and
   ``Resize(256)`` then ``CenterCrop(224)`` for evaluation; use it for
   torchvision pretrained backbones.

``train_samples_per_class``, ``val_samples_per_class`` (default ``null``)
   Keep a seeded random subset of at most this many images per class.
   ``train_samples_per_class`` limits the training images before the
   validation split is drawn from them; ``val_samples_per_class`` limits the
   test split. ``dendritic-train --train_samples_per_class N
   --val_samples_per_class M`` sets them from the command line for short
   test runs.

``allow_flatten_materialize`` (default ``false``)
   ImageNet requires ``processing.flatten: false``; an unset ``flatten``
   counts as ``true`` and raises. ``allow_flatten_materialize: true`` instead
   materializes flattened tensors in memory, keeping at most the first 50,000
   examples of each split. It is meant for debugging only.

The validation split is the random ``1 - train_valid_split`` fraction of the
training images, seeded by ``experiment.split_seed`` and loaded with the
evaluation transform. The test split is the official validation set.
``normalize: true`` applies the ImageNet channel mean and standard deviation.

Two environment variables help with large copies on shared filesystems:

``IMAGENET_LOCAL_DIR``
   When it names an existing directory, ImageNet is loaded from there instead
   of the data root, for example from a copy on local disk.

``DENDRITIC_IMAGEFOLDER_CACHE``
   A directory for a cached ``ImageFolder`` index (class names and file
   list). The first run scans the class directories and writes the index;
   later runs read it instead of scanning about 1.28 million files.

ImageNet protocol loaders
~~~~~~~~~~~~~~~~~~~~~~~~~

``dendritic_modeling.datasets.imagenet_protocol`` builds single ImageNet
splits of a run without loading the others, for example to evaluate another
model on exactly the images a run trained, selected and tested on. Both
functions take the run's configuration (as returned by
``dendritic_modeling.config.load_config``), require
``data.dataset_name: imagenet`` and ``processing.flatten: false``, and read
the ImageNet location, ``loader_backend``, ``transform_preset`` and
``normalize`` from it. They find the data where training does:
``data.base_dir``, else ``$DENDRITIC_DATA_DIR``, else the cache directory.

``build_protocol_train_partition(config)``
   Returns ``(train, valid, info)``: the training subset with the training
   transform and the disjoint validation subset with the evaluation
   transform, partitioned exactly as in training (``train_valid_split`` and
   ``experiment.split_seed``, which falls back to ``experiment.seed``). It
   raises when ``train_samples_per_class`` is set.

``build_protocol_test_dataset(config)``
   Returns ``(test, info)``: the official validation set with the evaluation
   transform. It raises when ``val_samples_per_class`` is set.

``info`` records the ImageNet directory, backend, transform preset and
transforms, ``normalize``, and the split sizes (and, for the training
partition, ``train_valid_split`` and the split seed).

.. _dataset-loading:

Loading during training
-----------------------

The DataLoader settings of training are fields of ``training.main.common``
(``num_workers``, ``pin_memory``, ``persistent_workers``, ``prefetch_factor``,
``in_order``), listed under "Training loop options" in
:doc:`training_strategies`; analyses use ``analysis.runtime`` instead (see
:ref:`analysis-runtime-profiles`). Except under FSDP, training records the
resolved settings under ``dataloader_tuning`` in ``training_summary.json``.
Left unset, they depend on the dataset type:

- In-memory ``TensorDataset`` data (the vision and synthetic datasets, also
  inside ``Subset`` or ``ConcatDataset``) load in the main process.
- Other datasets, including the package's generated sequence datasets, the
  Poisson generator and the paired datasets other than ``mnist_modulo10``,
  use worker processes. Set ``num_workers: 0`` to load them in the main
  process. From the second epoch on, the shuffle order and the
  per-sample random draws differ between main-process, persistent-worker and
  non-persistent-worker loading, so keep these settings fixed when runs must
  match.
- ImageNet and other torchvision image-folder datasets use workers with
  ``prefetch_factor: 4`` and ``in_order: false``.
