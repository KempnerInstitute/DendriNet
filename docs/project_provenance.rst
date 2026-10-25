.. _project-provenance:

Project provenance
==================

A study that builds on DendriNet keeps its own code (configurations, runners,
analyses, figure scripts) in its own repository and runs it against an exact
library commit. Four standard-library utilities in
``dendritic_modeling/utils`` record and check that relationship:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Utility
     - What it does
   * - ``project_source.py``
     - Verifies a project's source contract against a library checkout and
       runs the project's declared entry points with the verified package,
       writing a receipt (:ref:`project-source-contracts`).
   * - ``project_export.py``
     - Exports an allow-listed set of committed project files to a
       deterministic archive with an export record
       (:ref:`project-exports`).
   * - ``project_evidence.py``
     - Checks that the files a project declares as evidence for its results
       still have their recorded SHA-256 digests (:ref:`project-evidence`).
   * - ``exact_file_inventory.py``
     - Python functions that freeze a list of input files with their digests
       and rehash them, shared across data-parallel ranks, before a run uses
       them (:ref:`exact-file-inventories`).

The first three are programs that run by file path, so that they can check a
checkout before any copy of DendriNet is imported; they never edit a checkout
or change a Git reference. None of them assesses scientific equivalence:
a verified source or a matching digest says which bytes ran, not that two
versions compute the same results.

.. _project-source-contracts:

Source contracts
----------------

A project declares the library it runs on in ``dendrinet.project.json`` at the
root of its repository:

.. code-block:: json

   {
     "schema_version": 1,
     "project": "example-study",
     "library": {
       "line": "core",
       "commit": "0123456789abcdef0123456789abcdef01234567",
       "ref": "refs/tags/example-study/v1",
       "maintained_ref": "refs/remotes/origin/develop"
     },
     "source_paths": ["scripts", "configs"],
     "entry_points": {
       "train": "scripts/train.py",
       "figures": "scripts/figures.py"
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``schema_version``
     - ``1``.
   * - ``project``
     - The project's name.
   * - ``library.line``
     - ``core`` for a maintained line of the library (``llm`` is accepted for
       existing contracts), or ``historical`` for an original source that no
       maintained branch contains.
   * - ``library.commit``
     - The exact 40-character commit the project runs on.
   * - ``library.ref``
     - A tag (``refs/tags/...``) that points at ``library.commit``.
   * - ``library.maintained_ref``
     - Required for ``core`` and ``llm``, not allowed for ``historical``: the
       full branch ref (``refs/heads/...`` or ``refs/remotes/...``) whose
       history must contain ``library.commit``. A contract without it raises;
       there is no default branch.
   * - ``library.release_bridge``
     - Optional, for a release built by squashing (:ref:`release-bridges`).
   * - ``source_paths``
     - The project's code and configuration paths, relative to the
       repository root.
   * - ``entry_points``
     - Names mapped to Python scripts inside ``source_paths``.

Check a contract, or run an entry point, with the utility of the library
checkout the contract pins:

.. code-block:: bash

   python <library>/src/dendritic_modeling/utils/project_source.py check \
       --contract <project>/dendrinet.project.json --library-root <library>

   python <library>/src/dendritic_modeling/utils/project_source.py run \
       --contract <project>/dendrinet.project.json --library-root <library> \
       --entry figures --receipt <runs>/figures_source.json -- --out <runs>/figures

``--library-root`` defaults to ``$DENDRINET_LIBRARY_ROOT``; there is no
fallback to an installed copy. A tool from a newer checkout can verify an
older library commit that predates the utility.

To keep the contract in a subdirectory, pass ``--paper-root <project>`` (or
``paper_root=Path(...)`` to ``verify_contract``). Source paths and entry points
still resolve from that explicit Git root, and the nested contract is checked
for uncommitted changes. Without this option the contract must remain at the
root, preserving existing calls and receipts. The contract cannot be outside
the declared repository, including through a symlink.

``check`` verifies, and fails with ``SourceContractError`` at the first
problem:

* the contract's fields as described above;
* the project directory (the contract's directory) and the library root are
  each the root of a Git repository;
* the library's ``HEAD`` and the commit of ``library.ref`` are both
  ``library.commit``;
* for a maintained line, ``library.maintained_ref`` resolves to a commit and
  ``library.commit`` is that commit or one of its ancestors (or the release
  bridge holds);
* the library has a package at ``src/dendritic_modeling/__init__.py``;
* ``src``, ``configs``, ``pyproject.toml``, ``setup.py`` and
  ``requirements.txt`` of the library, and the ``source_paths`` and contract of
  the project, have no uncommitted or untracked changes, no ignored files with
  a source suffix (``.py``, ``.so``, ``.pyd``, ``.yaml``, ``.yml``, ``.json``,
  ``.jsonl``, ``.toml``, ``.sh``, ``.sbatch``, ``.slurm``, ``.ipynb``) and no
  tracked symbolic links;
* every declared source path exists, and every entry point is a tracked
  Python file inside a declared source path.

It then prints the receipt:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Receipt field
     - Content
   * - ``schema_version``, ``project``
     - ``1`` and the project's name.
   * - ``contract``, ``contract_sha256``
     - Path and SHA-256 digest of the contract.
   * - ``paper_commit``
     - ``HEAD`` of the project repository.
   * - ``library_commit``, ``library_ref``, ``library_line``
     - The verified library commit, its tag and its line.
   * - ``maintained_ref``, ``maintained_commit``
     - The maintained branch and the commit it pointed at (``null`` for a
       historical source).
   * - ``release_bridge``
     - Only when the contract declares a bridge: its ``source_commit``,
       ``release_commit`` and their common ``tree``.
   * - ``library_root``, ``configuration_root``, ``package_file``
     - The library checkout, its ``configs`` directory and its package file.
   * - ``python_executable``, ``python_version``
     - The interpreter.
   * - ``status``, ``scientific_equivalence``
     - ``source_verified`` and ``not assessed by source verification``.

``run`` verifies the contract the same way and then runs the named entry point
as ``__main__``, with the arguments after ``--``, from the project directory.
It imports ``dendritic_modeling`` from ``<library>/src`` and fails when another
copy was already imported; it sets ``DENDRINET_LIBRARY_ROOT`` and
``DENDRINET_CONFIG_ROOT`` (the library's ``configs``) and puts ``<library>/src``
first on ``PYTHONPATH`` for child processes. The receipt must be a new file
outside the project and library repositories: an existing receipt is never
overwritten. It is written when the run starts and again when it ends, with
``entry_point``, ``script``, ``arguments``, ``imported_package_file``,
``status`` (``started``, then ``completed`` or ``failed``) and ``exit_code``,
or ``error_type`` and ``error`` for an exception. ``run`` exits with the entry
point's exit status.

.. _release-bridges:

Releases built by squashing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A public release can be built by squashing the maintained line into a single
commit whose tree equals a commit of that line. The release then has none of
the line's commits as ancestors, so a contract that pins a commit of the line
and names the release branch as ``maintained_ref`` fails the ancestry check
even though the released files are identical. The contract states that
relationship explicitly:

.. code-block:: json

   {
     "library": {
       "line": "core",
       "commit": "0123456789abcdef0123456789abcdef01234567",
       "ref": "refs/tags/example-study/v1",
       "maintained_ref": "refs/remotes/origin/develop",
       "release_bridge": {
         "source_commit": "89abcdef0123456789abcdef0123456789abcdef",
         "release_commit": "fedcba9876543210fedcba9876543210fedcba98"
       }
     }
   }

With a bridge, the ancestry check is replaced by these checks, all of which
must hold:

* ``source_commit`` and ``release_commit`` are exact 40-character commits
  present in the library checkout (fetch both before verification);
* ``library.commit`` is ``source_commit`` or one of its ancestors, so the
  release contains the pinned commit;
* ``release_commit`` is the ``maintained_ref`` commit or one of its ancestors;
* ``source_commit`` and ``release_commit`` have the same tree
  (``git rev-parse <commit>^{tree}``).

The receipt records the bridge and the tree. Without a bridge the ancestry
check is unchanged, and a historical contract cannot declare one. The bridge
establishes identical released files only; it does not approve a release or
assess its results.

.. _project-exports:

Exports
-------

``project_export.py`` writes the files that a manifest at the root of the
project repository lists, as committed at a given commit, to a new ZIP
archive:

.. code-block:: bash

   python <library>/src/dendritic_modeling/utils/project_export.py \
       --manifest <project>/manuscript_export.json \
       --commit <project commit> --output <exports>/manuscript.zip

The manifest is read from that commit, not from the working tree:

An optional ``--paper-root <project>`` (``paper_root=Path(...)`` in the Python
API) allows the manifest itself to live in a subdirectory. Its listed file
paths remain relative to the explicit repository root. Containment, committed
input checks and exclusive creation are unchanged. Existing root-level calls
retain their default behavior. At the same exporter revision, equivalent root
and nested manifests produce byte-identical archives; changing exporter
revisions changes the recorded exporter hash, not the exported file contents.

.. code-block:: json

   {
     "schema_version": 1,
     "kind": "manuscript",
     "files": ["main.tex", "references.bib", "figures/figure1.pdf"]
   }

``kind`` is ``manuscript`` or ``paper-code``. Manuscript files have the
suffixes ``.tex``, ``.bib``, ``.bst``, ``.cls``, ``.sty``, ``.pdf``, ``.eps``,
``.png`` or ``.jpg``. Paper-code files have the suffixes ``.py``, ``.sh``,
``.sbatch``, ``.slurm``, ``.yaml``, ``.yml``, ``.json``, ``.md``, ``.txt``,
``.tsv``, ``.csv``, ``.toml``, ``.ini``, ``.cpp`` or ``.h``; explicitly
listed files named ``LICENSE`` or ``Makefile`` are also allowed. Paper-code paths
cannot contain ``src``, ``dendritic_modeling``, ``runtime``, ``historical_runtimes``,
``source_snapshots``, ``frozen_sources``, ``.git``, ``outputs``, ``results``
or ``logs`` (library source is never exported with a project). A
paper-code manifest also declares ``publication_status: internal_candidate``,
a ``library_dependency`` object and optionally ``reproduction_status``. Every
listed file must be a regular file committed at that commit; absolute paths,
``..`` and backslashes are refused, and so is a list with duplicates.

The archive must be a new file outside the project repository and the library
checkout. It holds each file with its Git mode and a fixed timestamp, so the
same commit and manifest give a byte-identical archive, and
``EXPORT_SOURCE.json`` with ``schema_version``, ``kind``, ``paper_commit``,
``manifest_sha256``, ``exporter_sha256`` and ``files`` (path, SHA-256 digest
and size of each); a paper-code export adds ``publication_status``,
``reproduction_status`` (default ``not validated``), ``library_dependency``
and ``library_source_included: false``. An export does not make a candidate
ready for publication.

.. _project-evidence:

Evidence checks
---------------

``project_evidence.py`` checks a manifest that binds a project's results to
the files that support them:

.. code-block:: bash

   python <library>/src/dendritic_modeling/utils/project_evidence.py \
       --manifest <project>/evidence.json \
       --root paper=<project> --root runs=<results root> \
       --report <reports>/evidence_report.json --require-qualified

.. code-block:: json

   {
     "schema_version": 1,
     "project": "example-study",
     "baseline_paper_commit": "0123456789abcdef0123456789abcdef01234567",
     "files": [
       {"id": "fig1_data", "root": "runs", "path": "figure1/summary.csv",
        "sha256": "<64 hexadecimal digits>"}
     ],
     "results": [
       {"id": "Figure 1", "artifact": "fig1_data", "evidence": [],
        "workflow_kind": "native", "qualification": "pending"}
     ],
     "missing_evidence": []
   }

Each file names one of the ``--root NAME=PATH`` directories and a relative path
inside it, with its expected SHA-256 digest. Each result names its artifact
and evidence files by ``id`` and classifies its computation in
``workflow_kind``; it counts as qualified only with ``qualification:
numerically_validated`` and a non-empty ``validation_evidence`` list of file
``id`` values. Unknown roots, paths that leave their root, duplicate
identifiers and references to undeclared files raise.

The report, a new file outside every root, holds ``integrity`` (``passed``
when every file matches), one check per file (``match``, ``changed`` or
``missing``), ``unqualified_results``, ``unresolved_dependencies`` (the
manifest's ``missing_evidence``), the manifest digest and the baseline commit.
The command exits with status 1 when a file does not match, and with
``--require-qualified`` also when a result is unqualified or a dependency is
unresolved. Matching digests show that the declared bytes are preserved, not
that the workflow that produced them is valid.

.. _exact-file-inventories:

Exact-file inventories
----------------------

``dendritic_modeling.utils.exact_file_inventory`` freezes the input files of a
run after the application has verified them, and rehashes every file before
each execution; no size-or-timestamp shortcut is accepted. An inventory lists
each file's absolute resolved path, size, SHA-256 digest and roles, sorted by
path with task identifiers ``file_00000``, ``file_00001``, ..., and carries its
own digest:

* ``build_inventory`` canonicalizes the application's list of files (merging
  the roles of a repeated path) with the application's generation binding;
* ``validate_inventory`` checks an inventory's schema, order and digest
  without reading the files;
* ``balanced_assignment`` assigns the files to ranks, largest first, each to
  the rank with the fewest bytes so far;
* ``verify_shard`` rehashes one rank's files, comparing a symlink-aware
  ``file_snapshot`` before and after each read, and returns a shard receipt;
* ``merge_verified_shards`` accepts exactly one shard per rank with exactly
  the assigned files, checks that no file changed since it was hashed, and
  returns the verification receipt (with its own digest) and the snapshots.

The functions are listed in :ref:`python-api`.
