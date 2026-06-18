Installation
============

Requirements: Conda and a CUDA-capable GPU. The default environment targets
CUDA 12.1+; for a CUDA 12.9 cluster see :ref:`cluster-setup` below.

Get the code
------------

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling

Standard setup
--------------

Create the default Conda environment and install the package in editable mode:

.. code-block:: bash

   conda env create -f setup/environment.yml
   conda activate dendritic_env
   pip install -e .

Optional dependency groups:

.. code-block:: bash

   pip install -e ".[dev]"    # pytest, ruff, sphinx
   pip install -e ".[train]"  # training and analysis extras
   pip install -e ".[all]"    # everything

.. _cluster-setup:

Cluster setup (CUDA 12.9)
-------------------------

On a CUDA 12.9 cluster (e.g. the Kempner cluster), the conda ``environment.yml``
pins an incompatible ``pytorch-cuda=12.1``. Use the helper script instead, which
installs a CUDA-12.8 PyTorch wheel (bundled CUDA runtime; runs natively on a
CUDA 12.9 driver) and then runs ``pip install -e .``:

.. code-block:: bash

   bash setup/setup_cuda129.sh
   conda activate dendritic_env

Configurable via environment variables — e.g. install into scratch space, or add
extras:

.. code-block:: bash

   ENV_PREFIX=/scratch/$USER/envs/dendritic_env bash setup/setup_cuda129.sh
   EXTRAS="[dev]" bash setup/setup_cuda129.sh

See the comments at the top of ``setup/setup_cuda129.sh`` for all options and the
cu128-vs-cu129 compatibility rationale.

Optional: ffmpeg
----------------

``ffmpeg`` is only needed for ablation video export.

.. code-block:: bash

   conda install -c conda-forge ffmpeg

Verify
------

.. code-block:: bash

   python -c "import torch; import dendritic_modeling; print('OK')"
   python src/dendritic_modeling/scripts/training/train_experiments.py --help
   python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help

Notes
-----

- The supported workflow is ``pip install -e .``.
- Set ``PYTHONPATH`` manually only if you are debugging a broken environment.
