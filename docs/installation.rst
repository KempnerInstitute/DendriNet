Installation
============

Requirements: Conda and CUDA 12.1+.

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling
   bash install.sh

Manual alternative:

.. code-block:: bash

   conda env create -f setup/environment.yml
   conda activate dendritic_env
   pip install -e .

Optional dependency groups:

.. code-block:: bash

   pip install -e ".[dev]"    # pytest, ruff, sphinx
   pip install -e ".[train]"  # training and analysis extras
   pip install -e ".[all]"    # everything

Verify:

.. code-block:: bash

   python -c "import torch; import dendritic_modeling; print('OK')"
