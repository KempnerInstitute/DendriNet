.. _synthetic_datasets:

Datasets and Data Utilities
===========================

This document explains the datasets available in the dendritic modeling framework, including both standard datasets and specialized synthetic datasets designed to test specific computational properties of dendritic networks.

Overview
--------

The dendritic modeling framework supports various datasets for training and evaluating neural networks. These include standard benchmark datasets like MNIST and CIFAR-10, as well as a suite of synthetic datasets designed to test specific computational properties that are theoretically associated with dendritic processing.

Dataset Configuration
---------------------

Datasets are configured in the ``task`` section of the configuration file:

.. code-block:: yaml

    task:
      dataset: mnist                    # Dataset name
      data_path:                        # Optional path to dataset
      parameters:                       # Additional dataset parameters
        train_size: 8000                # For synthetic datasets
        valid_size: 1000
        test_size: 1000
        image_size: 16                  # For image-based synthetic datasets
        n_orient_classes: 8             # For orientation_bars dataset

Standard Datasets
-----------------

The framework provides easy access to standard benchmark datasets:

MNIST Dataset
^^^^^^^^^^^^^

The MNIST dataset consists of 28×28 grayscale images of handwritten digits (0-9):

.. code-block:: yaml

    task:
      dataset: mnist
      data_path:                        # Optional, downloads automatically if not provided

The dataset is loaded using the ``load_mnist_as_datasets`` function, which returns training, validation, and test datasets:

.. code-block:: python

    def load_mnist_as_datasets(train_valid_split=0.8, data_path=None):
        """
        Load the MNIST dataset and split into train, validation, and test datasets.
        
        Parameters
        ----------
        train_valid_split : float
            Fraction of training data to use for training (vs. validation)
        data_path : str or None
            Path to dataset, downloads if None
            
        Returns
        -------
        dict
            Dictionary with 'train', 'valid', and 'test' datasets
        """

CIFAR-10 Dataset
^^^^^^^^^^^^^^^^

The CIFAR-10 dataset consists of 32×32 RGB images from 10 classes:

.. code-block:: yaml

    task:
      dataset: cifar10
      data_path:                        # Optional, downloads automatically if not provided

The dataset is loaded using the ``load_cifar10_as_datasets`` function.

Modified MNIST Variants
-----------------------

The framework also provides several modified versions of MNIST for testing specific properties:

MNIST Modulo-10
^^^^^^^^^^^^^^^

This dataset reorganizes MNIST into modular arithmetic classes:

.. code-block:: yaml

    task:
      dataset: mnist_modulo10
      parameters:
        shuffle_iterations: 2           # Number of shuffling iterations

The dataset is loaded using the ``load_MNIST_modulo10`` function.

MNIST Task Switch
^^^^^^^^^^^^^^^^^

This dataset reorganizes MNIST for task switching experiments:

.. code-block:: yaml

    task:
      dataset: mnist_switch2            # Options: mnist_switch2, mnist_switch10
      parameters:
        train_valid_split: 0.8          # Train/validation split ratio

The ``load_MNIST_task_switch`` function loads this dataset.

Synthetic Datasets
------------------

The framework includes several synthetic datasets specifically designed to test dendritic computations:

Line Detection Dataset
^^^^^^^^^^^^^^^^^^^^^^

A simple line detection task:

.. code-block:: yaml

    task:
      dataset: line
      parameters:
        train_size: 8000
        valid_size: 1000
        test_size: 1000
        image_size: 10

The ``LineDataset`` class implements this dataset:

.. code-block:: python

    class LineDataset(Dataset):
        """
        Dataset for line detection. Each sample is a small image that either 
        contains a horizontal or vertical line, or no line.
        """
        def __init__(self, n_samples=10000, image_size=10):
            self.n_samples = n_samples
            self.image_size = image_size
            self._generate_data()

Center-Surround Detection
^^^^^^^^^^^^^^^^^^^^^^^^^

A center-surround pattern detection task:

.. code-block:: yaml

    task:
      dataset: center_surround
      parameters:
        train_size: 8000
        valid_size: 1000
        test_size: 1000
        image_size: 15

The ``CenterSurroundDataset`` class implements this dataset:

.. code-block:: python

    class CenterSurroundDataset(Dataset):
        """
        Dataset for center-surround pattern detection. Each sample is a small image
        that either contains a center-surround pattern or random noise.
        """
        def __init__(self, n_samples=10000, image_size=15):
            self.n_samples = n_samples
            self.image_size = image_size
            self._generate_data()

Orientation Bars Dataset
^^^^^^^^^^^^^^^^^^^^^^^^

A dataset with oriented bars at different angles:

.. code-block:: yaml

    task:
      dataset: orientation_bars
      parameters:
        train_size: 8000
        valid_size: 1000
        test_size: 1000
        image_size: 16
        n_orient_classes: 8             # Number of orientation classes

The ``OrientationBarsDataset`` class implements this dataset:

.. code-block:: python

    class OrientationBarsDataset(Dataset):
        """
        Dataset with oriented bars at different angles. Each sample is a small image
        containing a bar at one of n_orient_classes different orientations.
        """
        def __init__(self, n_samples=10000, image_size=16, n_orient_classes=8):
            self.n_samples = n_samples
            self.image_size = image_size
            self.n_orient_classes = n_orient_classes
            self._generate_data()

Multi-XOR Dataset
^^^^^^^^^^^^^^^^^

A dataset for testing XOR-like logical operations:

.. code-block:: yaml

    task:
      dataset: multi_xor
      parameters:
        train_size: 8000
        valid_size: 1000
        test_size: 1000
        n_bits: 8                      # Number of bits for XOR operation

The ``MultiXORDataset`` class implements this dataset:

.. code-block:: python

    class MultiXORDataset(Dataset):
        """
        Dataset for testing XOR-like logical operations. Each sample is a bit string,
        and the target is the XOR of all bits.
        """
        def __init__(self,n_samples=10000,n_bits=8):
            self.n_samples = n_samples
            self.n_bits = n_bits
            self._generate_data()

Information Shunting Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dataset for testing shunting inhibition:

.. code-block:: yaml

    task:
      dataset: info_shunting
      parameters:
        train_size: 8000
        valid_size: 1000
        test_size: 1000
        input_dim: 10                  # Input dimension

The ``InfoShuntingDataset`` class implements this dataset:

.. code-block:: python

    class InfoShuntingDataset(Dataset):
        """
        Dataset for testing shunting inhibition. Contains samples where 
        some input dimensions should inhibit others.
        """
        def __init__(self,n_samples=20000, input_dim=10):
            self.n_samples = n_samples
            self.input_dim = input_dim
            self._generate_data()

Noise Resilience Testing
^^^^^^^^^^^^^^^^^^^^^^^^

A dataset for testing resilience to noise:

.. code-block:: yaml

    task:
      dataset: noise_resilience
      parameters:
        train_size: 8000
        valid_size: 1000
        test_size: 1000
        image_size: 10
        noise_level: 0.2               # Level of noise to add

The ``NoisyLineDataset`` class implements this dataset.

Feature Selectivity Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Datasets for testing feature selectivity:

1. **Sparse MNIST**: A sparse version of MNIST with thresholding

   .. code-block:: python

       class SparseMNISTDataset(Dataset):
           """
           A sparse version of MNIST created by thresholding.
           Only values above threshold are kept, others set to 0.
           """
           def __init__(self, base_mnist, threshold=0.3):
               self.base_mnist = base_mnist
               self.threshold = threshold

2. **Contextual Figure-Ground MNIST**: MNIST with contextual noise

   .. code-block:: python

       class ContextualFigureGroundMNIST(Dataset):
           """
           MNIST with contextual noise added to test figure-ground separation.
           """
           def __init__(self, base_mnist, noise_level=0.1):
               self.base_mnist = base_mnist
               self.noise_level = noise_level

Hierarchical Processing Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Datasets for testing hierarchical processing:

1. **Multi-Task Line Dataset**: Multiple line detection tasks

   .. code-block:: python

       class MultiTaskLineDataset(Dataset):
           """
           Dataset with multiple line detection tasks of varying difficulty.
           """
           def __init__(self,total_size=12000,image_size=10):
               self.total_size = total_size
               self.image_size = image_size
               self._generate_data()

2. **Compositional MNIST**: Combining multiple MNIST digits

   .. code-block:: python

       class CompositionalMNIST(Dataset):
           """
           Dataset that combines multiple MNIST digits to test compositional processing.
           """
           def __init__(self, base_mnist, n_samples=20000):
               self.base_mnist = base_mnist
               self.n_samples = n_samples
               self._generate_data()

Data Utilities
--------------

The framework provides several utility functions for data handling:

Split MNIST Inputs
^^^^^^^^^^^^^^^^^^

The ``split_MNIST_inputs`` function splits MNIST inputs by digit:

.. code-block:: python

    def split_MNIST_inputs(inputs, labels):
        """
        Split MNIST inputs into a list where each element contains all samples for one digit.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor [N, 784]
        labels : torch.Tensor
            Label tensor [N]
            
        Returns
        -------
        list
            List of tensors, one per digit
        """

Unified Dataset Loading
^^^^^^^^^^^^^^^^^^^^^^^

The ``get_unified_datasets`` function provides a unified interface for loading datasets:

.. code-block:: python

    def get_unified_datasets(task_cfg, train_cfg):
        """
        Load datasets based on configuration.
        
        Parameters
        ----------
        task_cfg : dict
            Task configuration with dataset name and parameters
        train_cfg : dict
            Training configuration with train_valid_split
            
        Returns
        -------
        dict
            Dictionary with 'train', 'valid', and 'test' datasets
        """

This function examines the dataset name in `task_cfg` and calls the appropriate loading function.

Best Practices
--------------

When working with datasets:

1. **Start with standard datasets**: Begin with MNIST or CIFAR-10 for initial testing
2. **Use synthetic datasets for specific properties**: Use synthetic datasets to test specific computational properties
3. **Create custom datasets**: Extend the framework with custom datasets for specific research questions
4. **Data preprocessing**: Consider normalization and augmentation for improved performance
5. **Balanced samples**: Ensure balanced class distribution in validation and test sets

Synthetic datasets are particularly valuable for understanding the specific computational advantages of dendritic networks compared to standard neural networks. 