Runnable Config Examples
========================

These four YAML files are the main copy-paste starting points for local runs.

MNIST Single Layer
------------------

Original single-layer MNIST feedforward example:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/standard_mnist.yaml

CIFAR-10 Single Layer
---------------------

CIFAR-10 with one 128E/32I dendritic layer, ``[3, 3, 3]`` morphology, and
40 excitatory / 10 inhibitory synapses per excitatory branch:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/cifar10_128_333_40e10i.yaml

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
ImageNet root; without ImageNet, the loader falls back to small fake data for
smoke testing only.

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python \
     src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/examples/alexnet_fc7_dendritic_500.yaml
