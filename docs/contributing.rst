Contributing
============

Development setup
-----------------

Install the package from a checkout (see :doc:`installation`) with the
``dev`` extra, which adds the test and lint tools, and install the
pre-commit hooks:

.. code-block:: bash

   pip install -e ".[dev]" -c setup/constraints/core-py310-torch2.9.1.txt
   pre-commit install

Several checkouts
~~~~~~~~~~~~~~~~~

An editable install serves the one checkout it was installed from. When you
work with several checkouts or worktrees, run Python with
``PYTHONPATH=<checkout>/src`` so that the code you edit is the code that
runs, for example
``PYTHONPATH=$PWD/src python -m dendritic_modeling.scripts.training.train_experiments <config>``.
Every training run records the imported package path and its commit in
``runtime_environment.json``.

Where code belongs
------------------

The library holds components that serve more than one study: models, layers,
synapse and connectivity policies, training rules, datasets, analysis tools,
configuration schemas and the command-line programs, under
``src/dendritic_modeling``, with tests in ``tests/`` and documentation in
``docs/``. Runners, sweeps, figure scripts and configurations of one study
belong in that study's own repository, which records the library revision it
uses.

A new option or component:

* is a typed configuration field with validation, so the loader accepts and
  checks it (see :doc:`configuration_guide`);
* is off by default: existing configurations keep their exact behaviour, and
  a test shows that the default output does not change;
* registers through the existing registries where one exists (see
  :ref:`extension-registries`);
* is documented on the page of its topic, with an entry in ``CHANGELOG.md``
  for user-visible changes.

A change of what an omitted key means, such as a new default, changes the
configuration format: increment ``CONFIG_VERSION``
(:ref:`config-version`), add a rule to ``dendritic-upgrade-config`` that
writes the earlier meaning into files of earlier versions, and update the
shipped configurations.

Code style
----------

`black <https://black.readthedocs.io/>`_ formats the code and
`isort <https://pycqa.github.io/isort/>`_ orders the imports (both configured
in ``pyproject.toml``); `ruff <https://docs.astral.sh/ruff/>`_ lints it
(configured in ``ruff.toml``). The pre-commit hooks run all three, check
whitespace, YAML syntax, file sizes and merge markers, and run
``tests/test_training_stability_validator.py``. Run them on the files you
changed:

.. code-block:: bash

   pre-commit run --files <changed files>

Fix lint warnings instead of suppressing them with ``# noqa``; the few
existing suppressions are re-exports and benchmark lambdas.

Tests
-----

Run the tests from the repository root with ``PYTHONPATH`` pointing at the
checkout under test:

.. code-block:: bash

   # quick run without the slow tests
   PYTHONPATH=$PWD/src python -m pytest -m "not slow" -n auto

   # full suite, in parallel (pytest-xdist is in the test extra)
   PYTHONPATH=$PWD/src python -m pytest -n auto

   # one file
   PYTHONPATH=$PWD/src python -m pytest tests/test_config.py -q

``pytest.ini`` runs with ``--strict-markers`` and defines four markers:

``slow``
   Long tests, including tests that spawn worker processes or run training
   launchers; deselect them with ``-m "not slow"``.
``multiprocess``
   Tests that spawn worker processes (distributed tests).
``requires_gpu``
   Tests that need CUDA. Tests parametrized with ``device="cuda"`` get the
   marker automatically, and hosts without CUDA skip them. The continuous
   integration has no GPU, so run ``python -m pytest -m requires_gpu`` on a
   GPU machine before merging changes to kernels or device code.
``requires_git_checkout``
   Tests that read the library's own Git checkout (its commit and tracked
   files); they are skipped when the tests run from a source archive or sdist.

The continuous integration also measures coverage; to reproduce that check:

.. code-block:: bash

   PYTHONPATH=$PWD/src python -m pytest tests/ -n 2 \
     --cov=src/dendritic_modeling --cov-branch \
     --cov-report=xml --cov-fail-under=30

``tests/README.md`` describes the default-path golden tests, which fail when
a change alters the default behaviour, and how to regenerate their fixture.

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   python -m sphinx -b html -W --keep-going docs docs/_build/html

``-W`` turns warnings into errors; the pages build without warnings, and the
pull-request checks and the documentation deployment build them with the same
command. Open
``docs/_build/html/index.html`` to read the result. A page is an ``.rst``
file listed in a ``toctree`` of ``docs/index.rst``; Markdown files under
``docs/`` are not built.

Every ``.. code-block:: yaml`` example must load through the strict loader,
which ``tests/test_doc_yaml_snippets.py`` checks. After changing the
arguments of a command-line program, regenerate the command-line reference,
and after changing a configuration field, the options reference, with the
package importable, and commit the page:

.. code-block:: bash

   PYTHONPATH=$PWD/src python docs/scripts/generate_cli_reference.py
   PYTHONPATH=$PWD/src python docs/scripts/generate_options_reference.py

With ``--check``, either script fails when its page is out of date; the
pull-request checks run both that way.
