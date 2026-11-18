Preparing a public release
==========================

DendriNet uses the existing institutional repository and its accepted core
history. Private extensions and local archive branches are not part of a
public release. A release does not require rewriting that history or changing
the versions used by scientific studies.

Package boundary
----------------

The installed package contains ``dendritic_modeling`` and its small tutorial
configurations. ``projects/`` holds study-specific code and ``apps/`` holds
optional applications; neither is included in the core distribution. The
source distribution also contains the reference configs, tests and docs.
``setup/check_distribution.py`` checks the built archives against this boundary.

Release procedure
-----------------

#. Merge reviewed changes through a pull request to ``develop``. Qualify model
   or numerical changes with their own tests; packaging checks alone cannot
   qualify a scientific workflow.
#. Set the release version consistently and tag the reviewed merge commit.
#. Run the package checks: isolated source-to-wheel build, strict Twine metadata
   checks, archive inventory, installed examples and command-line imports.
#. Configure the PyPI and TestPyPI projects and GitHub environments for trusted
   publishing. Protect the publication environments with maintainer review.
#. Dispatch ``Package release`` for the release tag, first choosing ``testpypi``.
   Install that artifact in a clean environment and run the download-free demo.
#. After approval, dispatch the same tag with ``pypi``. The default
   ``build-only`` target never uploads anything. The workflow refuses uploads
   from an unversioned branch, a mismatched tag, or a commit outside develop.

Repository visibility, index ownership and environment protection are
maintainer settings; the workflow does not change them. Before publication,
check which remote branches, tags and artifacts will become visible. Never
push local private refs as part of release automation.

Installed examples
------------------

.. code-block:: bash

   dendritic-example
   dendritic-example synthetic_population --output demo.yaml
   dendritic-train demo.yaml --epochs 1

The exporter refuses to overwrite an existing file. Tutorial settings are
illustrations, not recommended hyperparameters or qualified paper recipes.

Research projects
-----------------

Study protocols, sweep choices, cohort summaries and figure layouts belong in
the study project. General models, training rules and analysis primitives
remain in the library. A study release records its source, dependencies,
resolved configurations, artifact identities and reproduction commands.
See :doc:`project_provenance`; publication does not migrate existing pins.
