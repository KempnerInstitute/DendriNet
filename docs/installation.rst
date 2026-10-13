Installation
============

DendriNet (Python package ``dendritic_modeling``) needs Python 3.10 or newer
and PyTorch 2.8 or newer. The continuous-integration tests run on Linux with
Python 3.10, 3.11 and 3.12 on CPUs. A CUDA GPU is optional; the Triton
kernels of indexed synapses need one.

Install from a checkout
-----------------------

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/DendriNet.git
   cd DendriNet
   conda env create -f setup/environment.yml
   conda activate dendritic_env
   pip install -e ".[test]" -c setup/constraints/core-py310-torch2.9.1.txt

``setup/environment.yml`` creates a conda environment named ``dendritic_env``
with Python 3.10 and pip only; a ``venv`` with Python 3.10 works the same.
pip then installs the package in editable mode with its dependencies,
including the PyTorch wheels.

The constraints file ``setup/constraints/core-py310-torch2.9.1.txt`` pins the
versions the test suite passed with on Python 3.10: PyTorch 2.9.1 with its
CUDA 12.8 wheels from PyPI, NumPy 2.2 and SciPy 1.15, among others. Without
``-c``, pip resolves the version ranges in ``pyproject.toml``; use that on
other Python versions. ``pip install -e .`` installs the package without the
test tools.

Optional dependencies
---------------------

Extras add dependency groups, for example ``pip install -e ".[dev,gpu]"``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Adds
   * - ``test``
     - pytest with pytest-cov, pytest-timeout and pytest-xdist, and coverage.
   * - ``lint``
     - ruff, black, isort, mypy and pre-commit (see :doc:`contributing`).
   * - ``dev``
     - ``test`` and ``lint``, plus build, setuptools, wheel and twine for
       packaging.
   * - ``docs``
     - Sphinx, the Read the Docs theme and further Sphinx extensions, to build
       these pages.
   * - ``train``
     - Jupyter, IPython and ipykernel for notebooks.
   * - ``gpu``
     - Triton, for the Triton kernels of indexed synapses. The Linux CUDA
       wheels of PyTorch already include it.
   * - ``plotting``
     - networkx, for the graph drawings of ``dendritic_modeling.plotting``.
   * - ``transformers``
     - Hugging Face transformers, for
       ``dendritic_modeling.networks.architectures.transformer``, which
       replaces the feed-forward blocks of a pretrained transformer with
       dendritic layers.
   * - ``compression``
     - torchao, for the optional dense-runtime backends of compact
       checkpoints (``torchao_*``; see :ref:`compact-checkpoints`).
   * - ``all``
     - ``dev``, ``train``, ``docs``, ``gpu`` and ``plotting``.

The ablation and compartment-SNR analyses write their animations as MP4
videos when ``ffmpeg`` is installed (``conda install -c conda-forge ffmpeg``)
and as GIF files otherwise.

Check the installation
----------------------

.. code-block:: bash

   python -c "import dendritic_modeling; print(dendritic_modeling.__file__)"
   dendritic-train --help
   python -c "import torch; print(torch.cuda.is_available())"

The first command prints the path of the imported package, the second lists
the training options (:ref:`cli-reference` lists every command), and the
third reports whether PyTorch sees a CUDA GPU. Without one, models train on
the CPU. The Triton backends of indexed synapses (``triton_*``) need a CUDA
GPU; with ``projection_backend: auto`` a model uses a Triton kernel where
Triton runs on a CUDA device and the portable ``recompute`` implementation
elsewhere. See :ref:`indexed-backends`.

Data
----

Datasets are stored under one data root: ``data.base_dir`` when it is set
(``dendritic-train --data_dir`` sets it), else the directory named by the
``DENDRITIC_DATA_DIR`` environment variable, else
``$XDG_CACHE_HOME/dendritic_modeling`` (by default
``~/.cache/dendritic_modeling``). MNIST, Fashion-MNIST, CIFAR-10 and
CIFAR-100 download into it on first use; ImageNet must be provided. See
:doc:`datasets`.

Numerical environments
----------------------

The package accepts NumPy 1.x and 2.x and the other version ranges in
``pyproject.toml``. An allowed range does not promise bitwise agreement across
versions. Every training run records its library commit, package path and
dependency versions in ``runtime_environment.json`` and its seeds and
numerics settings in ``resolved_seeds.json``. To repeat a result, install the
recorded stack, for example from a constraints file that
``setup/freeze_constraints.py`` writes for an environment:

.. code-block:: bash

   python setup/freeze_constraints.py --extras test lint gpu plotting \
     --output setup/constraints/<name>.txt

See :ref:`run-reproducibility`.
