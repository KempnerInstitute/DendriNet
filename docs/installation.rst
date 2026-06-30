Installation
============

Requirements: Python 3.10+ and Conda. The provided conda environment uses
Python 3.11 and CUDA 12.1 for GPU installs; adjust or remove ``pytorch-cuda``
in ``setup/environment.yml`` for CPU-only or site-specific CUDA builds.

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling
   conda env create -f setup/environment.yml
   conda activate dendritic_env
   pip install -e .

Optional dependency groups:

.. code-block:: bash

   pip install -e ".[dev]"    # pytest, ruff, sphinx
   pip install -e ".[train]"  # training and notebook extras
   pip install -e ".[all]"    # everything

Verify:

.. code-block:: bash

   python -c "import torch; import dendritic_modeling; print('OK')"
