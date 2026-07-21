Contributing
============

Development Setup
-----------------

.. code-block:: bash

   conda activate dendritic_env
   pip install -e ".[dev]"

This installs pytest, ruff, sphinx, and pre-commit hooks.

Code Style
----------

The project uses `ruff <https://docs.astral.sh/ruff/>`_ for linting and
formatting. Configuration is in ``ruff.toml``. Run checks with:

.. code-block:: bash

   ruff check src/ tests/
   ruff format src/ tests/

Do not suppress lint warnings with ``# noqa`` comments. Fix the underlying
issue instead (move imports, restructure code, etc.).

Testing
-------

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python -m pytest tests/

Coverage is collected explicitly in the primary Python CI job. To reproduce
that check locally, run:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python -m pytest tests/ \
      --cov=src/dendritic_modeling --cov-branch \
      --cov-report=xml --cov-fail-under=30

Test markers: ``@pytest.mark.unit``, ``@pytest.mark.integration``,
``@pytest.mark.slow``, ``@pytest.mark.requires_gpu``,
``@pytest.mark.requires_data``.

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html

View the generated site at ``docs/_build/html/index.html`` from the repository
root.
