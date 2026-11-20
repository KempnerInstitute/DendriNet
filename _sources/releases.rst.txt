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

Release order
-------------

Build and review both packages in one pull request to ``develop``. After
merging, tag the reviewed source, stage the wheels and source archives on
TestPyPI, and promote the tested files to PyPI. Publish core first so a
normal installation of Studio can resolve its dependency. Pull requests and
pushes to ``develop`` build and test the packages without uploading to an
index. **TestPyPI uploads are publicly downloadable too.**

The commands below stage and publish core **0.6.0** and Studio **0.2.0**.
For the install commands readers should use, see :doc:`installation`.
The empirical paper's examples and data remain attached to the
`v0.5.0 release <https://github.com/KempnerInstitute/DendriNet/releases/tag/v0.5.0>`_.
Its supplement records the versions used for training and for the worked examples.

1. Prepare the release pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Set the core version in ``pyproject.toml`` and
  ``src/dendritic_modeling/__init__.py``. Set the Studio version in
  ``apps/studio/pyproject.toml`` and ``apps/studio/src/dendrinet_studio/__init__.py``.
  For this pair, use ``dendritic_modeling>=0.6.0,<0.7`` in Studio's
  dependencies. Include the release notes in ``CHANGELOG.md``.
* Push only the reviewed candidate branch and open its PR against ``develop``.
  Review the package contents and any public study examples. The
  **Package release** check builds both source archives and wheels, checks
  metadata with Twine and inspects the core archive. It also installs both
  wheels outside the checkout, runs the installed examples and sweep checks,
  and exercises the Studio browser tests. Check the documentation build and
  the ordinary core test suite alongside the packaging job.
* Merge after review and passing CI. Record the merge commit from the PR
  and verify the checks on ``develop``. The documentation workflow also
  deploys the Sphinx site to the Pages branch on this push. The hosted page
  is the Studio manual. Readers can run the application locally or in the linked personal Binder session. Under **Settings →
  Pages**, verify the publishing source and access for the intended readers.
  Before announcing public documentation, open the
  `Studio guide <https://kempnerinstitute.github.io/DendriNet/studio.html>`_
  in a signed-out browser and check that the new walkthrough and figures load.

2. Set up publishing access
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both projects already exist on both indices. Reuse their publishing settings.
Review the environment restrictions and credentials before staging.

#. Use maintainer accounts on both `PyPI <https://pypi.org/>`_ and
   `TestPyPI <https://test.pypi.org/>`_. The indices use separate accounts.
#. In the institutional GitHub repository, create the Actions environments
   ``testpypi`` and ``pypi``. If you restrict deployments by ref, allow the
   release **tags** for both packages (``v*`` and ``studio-v*``). Configure a
   production reviewer if your repository plan supports it.
#. For each package on each index, configure a GitHub Trusted Publisher.
   Use owner ``KempnerInstitute``, repository ``DendriNet`` and workflow
   filename ``package.yml``. On TestPyPI use environment ``testpypi``;
   on PyPI use ``pypi``. The two project names are ``dendritic-modeling``
   (Python import ``dendritic_modeling``) and ``dendrinet-studio``.
   For a new project, create a pending publisher in the account's Publishing
   settings. For an existing project, its owner adds the publisher in project
   settings. A pending publisher does not reserve a name. See the
   `PyPI instructions <https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>`_.

The default ``trusted-publisher`` mode requests a short-lived credential
for the selected environment. Each index checks the issuer, repository,
workflow and environment. PyPI's ``unknown trusted publishing issuer``
error means the index did not recognize the identity provider. GitHub
Enterprise can set a custom issuer across an enterprise. Ask the enterprise
administrator which issuer is in use before editing publisher registrations.
See `PyPI troubleshooting <https://docs.pypi.org/trusted-publishers/troubleshooting/>`_
and `GitHub's issuer settings <https://docs.github.com/en/enterprise-cloud@latest/actions/reference/security/oidc#customizing-the-issuer-value-for-an-enterprise>`_.

For an existing project whose issuer the index does not support, the
``project-token`` mode uses the same build and staging checks. An owner
creates an API token scoped to that **one project on that index** and stores
it as a GitHub **environment secret**. Use ``CORE_PROJECT_TOKEN`` for
``dendritic-modeling`` or ``STUDIO_PROJECT_TOKEN`` for ``dendrinet-studio``.
Select the ``testpypi`` environment for a TestPyPI token and ``pypi`` for a
production token. Choose ``authentication=project-token`` in the dispatch
form or append ``-f authentication=project-token`` to the command for that
upload. This mode generates no Trusted Publishing attestations. Revoke a
temporary token on the issuing index and remove its environment secret
after verifying the published files.

3. Tag the reviewed merge
~~~~~~~~~~~~~~~~~~~~~~~~~

After the PR has merged and checks on ``develop`` pass, open its merge
commit in GitHub and copy the **full commit SHA**. In an authenticated
clone, set ``release_commit`` to that SHA. The example below retrieves it
from PR 404 using the official GitHub CLI. The guard stops if the commit
is absent from ``origin/develop``. Both tags point to the same commit even
if another change has since landed on the branch.

.. code-block:: bash

   (
     set -eu
     repo=KempnerInstitute/DendriNet
     release_commit=$(gh pr view 404 --repo "$repo" --json mergeCommit --jq '.mergeCommit.oid // empty')
     test -n "$release_commit"
     git fetch origin develop
     git merge-base --is-ancestor "$release_commit" origin/develop
     git show "${release_commit}:pyproject.toml" | grep -Fx 'version = "0.6.0"'
     git show "${release_commit}:apps/studio/pyproject.toml" | grep -Fx 'version = "0.2.0"'
     git tag -a v0.6.0 "$release_commit" -m "DendriNet 0.6.0"
     git tag -a studio-v0.2.0 "$release_commit" -m "DendriNet Studio 0.2.0"
     git push --atomic origin refs/tags/v0.6.0 refs/tags/studio-v0.2.0
   )

Use a remote configured for authorized pushes. If a push fails after the
local tags are created, verify that both resolve to the reviewed commit
with ``git rev-parse 'v0.6.0^{commit}' 'studio-v0.2.0^{commit}'``. Then
repeat the push to the authorized remote. Pushing tags alone does not
upload packages. After the tags appear on GitHub, open the Studio documentation’s
Binder link and check that the tagged session opens before staging the packages.

For a Studio-only release, bump the version in ``apps/studio/pyproject.toml``
and ``apps/studio/src/dendrinet_studio/__init__.py`` together. Create only
the matching ``studio-vX.Y.Z`` tag on the reviewed merge. Run the staging
and promotion workflows below for ``distribution=studio`` using this tag.
Install and test the staged Studio wheel with the published compatible core.
Users with an earlier Studio install can upgrade it with
``python -m pip install --upgrade dendrinet-studio``.

4. Stage and install both packages from TestPyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the `official GitHub CLI <https://cli.github.com/>`_ on your workstation.
Run ``gh --version`` and ``gh auth status`` first. GitHub registers a
manually dispatched workflow from the repository's default branch, which
must already contain ``.github/workflows/package.yml``. Pass the **tag**
as the workflow ref. See the `GitHub manual dispatch guide
<https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow>`_.

.. code-block:: bash

   repo=KempnerInstitute/DendriNet
   upload_auth=trusted-publisher  # use project-token for environment-scoped tokens
   gh workflow run package.yml --repo "$repo" --ref v0.6.0 \
     -f distribution=core -f target=testpypi -f authentication="$upload_auth"
   gh workflow run package.yml --repo "$repo" --ref studio-v0.2.0 \
     -f distribution=studio -f target=testpypi -f authentication="$upload_auth"

In GitHub Actions, record the numeric **run ID** from each run's URL
(``/actions/runs/<ID>``) with its distribution and tag. Wait for both runs,
including their upload jobs, to succeed. Run the following outside the
source checkout in an empty directory and a fresh environment. Download
only the two candidate wheels from TestPyPI. Pip will resolve their normal
dependencies from the production index at installation.

.. code-block:: bash

   python -m venv .venv
   . .venv/bin/activate
   python -m pip download --index-url https://test.pypi.org/simple/ \
     --only-binary=:all: --no-deps --dest wheels \
     'dendritic-modeling==0.6.0' 'dendrinet-studio==0.2.0'
   python -m pip install --index-url https://pypi.org/simple/ wheels/*.whl
   python -m pip check
   dendritic-example synthetic_population --output demo.yaml
   dendritic-train demo.yaml --epochs 1
   dendrinet-studio --workspace studio-check

Open the printed URL, run **Check model**, and train the synthetic example
on CPU as described in :ref:`studio-first-run`. Keep the results of this
clean-install check and the two staging run IDs. The `packaging guide to
TestPyPI <https://packaging.python.org/en/latest/guides/using-testpypi/>`_
explains why dependencies are downloaded separately.

5. Promote core, then Studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same shell, set ``core_stage_id`` and ``studio_stage_id`` to the
numeric run IDs from step 4. Use the authentication configured for the
production environment. Run the core command first. Wait for the upload
job to succeed and check the core version on PyPI before starting Studio.

.. code-block:: bash

   gh workflow run package.yml --repo "$repo" --ref v0.6.0 \
     -f distribution=core -f target=pypi -f authentication="$upload_auth" \
     -f staging_run_id="$core_stage_id"
   gh workflow run package.yml --repo "$repo" --ref studio-v0.2.0 \
     -f distribution=studio -f target=pypi -f authentication="$upload_auth" \
     -f staging_run_id="$studio_stage_id"

For each distribution, CI verifies the staging run's repository, workflow,
release tag, commit and successful TestPyPI upload job. It downloads the
artifacts and matches the wheel and source
archive names, versions and SHA-256 digests to the release on TestPyPI.
Production receives those exact files. If verification fails, inspect the
run ID, tag and file digests before approving an upload.

To compare the filenames and SHA-256 digests after an upload, run the
current verifier from a checkout at the release tag with its ``origin/develop``
ref. It reads package metadata from that commit. Use Python 3.11 or later
with ``packaging`` installed and the two files downloaded from the build.
From the checked-out release tag, run:

.. code-block:: bash

   python setup/verify_release.py core refs/tags/v0.6.0 --staged dist/core --index testpypi
   python setup/verify_release.py core refs/tags/v0.6.0 --staged dist/core --index pypi
   python setup/verify_release.py studio refs/tags/studio-v0.2.0 --staged dist/studio --index testpypi
   python setup/verify_release.py studio refs/tags/studio-v0.2.0 --staged dist/studio --index pypi

Finally, in a second empty environment, install only
``python -m pip install dendrinet-studio==0.2.0``. Run ``python -m pip check``,
inspect both installed versions with
``python -m pip show dendritic-modeling dendrinet-studio``, and repeat the
local tutorial. The core-only install is ``python -m pip install dendritic-modeling==0.6.0``.
Link the two PyPI pages and the Studio guide in the release notes. Keep
research workflows on their recorded versions until their numerical
comparisons qualify a newer release.

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
