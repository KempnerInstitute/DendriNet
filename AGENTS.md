# AGENTS.md

Guidance for coding agents and contributors working in this repository.
Tools that read `CLAUDE.md` can use a local, untracked link to this file
(`ln -s AGENTS.md CLAUDE.md`). Human-oriented detail lives in
[`docs/contributing.rst`](docs/contributing.rst).

## Project

DendriNet (Python package `dendritic_modeling`) is a PyTorch library for
neural networks built from dendritic neurons: branched dendritic trees with
sparse excitatory and inhibitory synapses, shunting or additive integration,
reactivation gates, feedforward and recurrent population networks, local
credit-assignment rules, distributed training, sweeps and analysis tools.
Several research projects depend on it, so changes must stay general,
backward compatible, tested and documented.

## Layout

| Path | Contents |
| --- | --- |
| `src/dendritic_modeling/config/` | Typed configuration dataclasses, the strict YAML loader, retired-key handling and config upgrade rules |
| `src/dendritic_modeling/networks/` | Dendritic neurons, branch layers, synapse and connectivity policies, EINet, recurrent and population networks, spatial and vision cores, baselines |
| `src/dendritic_modeling/training/` | Trainers, training strategies (backpropagation, feedback alignment, local rules), losses, optimizers |
| `src/dendritic_modeling/datasets/` | Dataset registry, loaders and synthetic tasks |
| `src/dendritic_modeling/analysis/` | Analyzer registry and analysis tools |
| `src/dendritic_modeling/deployment/`, `kernels/` | Compact checkpoint export and GPU kernels |
| `src/dendritic_modeling/scripts/` | Command-line programs (`dendritic-train`, `dendritic-sweep`, ...) |
| `configs/` | Example, template and sweep configurations; every YAML is load-tested |
| `tests/` | Pytest suite |
| `docs/` | Sphinx documentation; `docs/scripts/` generates the CLI and options references |

## Setup

```bash
pip install -e ".[dev]" -c setup/constraints/core-py310-torch2.9.1.txt
pre-commit install
```

Python 3.10+ and PyTorch 2.8+ are required. Optional extras: `gpu` (Triton
kernels), `plotting`, `transformers`, `docs`. When several checkouts or
worktrees exist, run Python with `PYTHONPATH=<checkout>/src` so the code you
edit is the code that runs.

## Commands

```bash
python -m pytest -m "not slow" -n auto            # fast suite
python -m pytest -n auto                          # full suite
python -m pytest -m requires_gpu                  # CUDA tests, on a GPU host
pre-commit run --files <changed files>            # ruff, black, isort, checks
python -m sphinx -b html -W --keep-going docs docs/_build/html
python docs/scripts/generate_cli_reference.py     # after changing a CLI
python docs/scripts/generate_options_reference.py # after changing a config field
dendritic-upgrade-config --output-dir <dir> <configs...>  # migrate old YAMLs
```

Use compute nodes, not shared login nodes, for parallel tests, training and
benchmarks.

## Design rules

- **Configuration is typed and strict.** Every option is a dataclass field
  that the loader validates. Unknown keys raise, and so do options the selected
  type would silently ignore. Keys retired from earlier versions load with a
  warning only when the current library reproduces their effect; renamed and
  retired keys also get a rule in `dendritic-upgrade-config`.
- **Extend through registries.** Architectures, sparse layers, datasets,
  losses, optimizers, trainer strategies, analyzers and activations have
  registries; new components register there and get typed configuration at
  every construction entry point. Connectivity and index-selection policies
  also record their provenance in compiled plans.
- **New behaviour is opt-in.** Defaults keep existing outputs, random streams,
  initialization order and checkpoint layout byte-identical; add a test that
  proves it. Promoting an option to a default is an explicit, documented
  decision.
- **Configuration files carry `config_version`.** A change of what an
  omitted key means (a new default, a new reading of an absent key) increments
  `CONFIG_VERSION` in `config/config.py`, adds a `dendritic-upgrade-config`
  rule that writes the earlier meaning into files of earlier versions, and
  updates the shipped configurations.
- **Integration rules and nonlinearities** need configuration-to-constructor
  tests through every supported builder, equation and gradient checks for each
  supported training rule, checkpoint and anchor round trips, and explicit
  errors for unsupported combinations. Keep across-feature additive
  normalization distinct from division by total conductance.
- **Model families are tested separately**: feedforward trees, recurrent and
  stateful cores, population networks, spatial cores. Include branched
  morphologies and realistic sparsity in tests and profiling; an operator
  speedup is not a model-level speedup.
- **Library scope.** Reusable models, layers, kernels, controls, datasets,
  training rules and analysis primitives belong here. Study protocols, sweeps,
  recipes, figure scripts and paper-specific analyses belong in the studies'
  own repositories, which pin a library revision.
- **Reuse before adding.** Inspect existing implementations and their contracts
  before adding a method, control or tool. Prefer composition or a small opt-in
  extension over a parallel implementation. Document the mathematical or API
  gap when a new component is necessary. A better existing method may replace
  paper code only after numerical comparisons; do not silently change the
  paper's equations or historical behavior to fit it.
- **Keep features working.** Do not remove a feature because no current study
  uses it; make it correct and tested. Remove only unreachable code and record
  each removal in `CHANGELOG.md`.
- **No site-specific state.** Take paths from configuration or arguments; never
  hard-code cluster paths, accounts or partitions. Do not add experiment
  trackers (wandb is not used).

## Tests

- Every behaviour change ships with a focused regression test; every new
  option with a default-identity test.
- Tests are deterministic and seeded. Mark CUDA tests `requires_gpu` and long
  tests `slow`; tests that spawn processes are `multiprocess`; tests that read
  the library's own Git checkout are `requires_git_checkout`.
- `tests/test_all_example_configs_smoke.py` loads every shipped YAML and builds
  every example model; `tests/test_doc_yaml_snippets.py` loads every YAML block
  in the documentation. Keep both passing when adding configs or docs.
- Default-path golden tests (see `tests/README.md`) fail when default behaviour
  changes; regenerate their fixtures only for an intended, documented change.

## Documentation

- Document user-visible behaviour on the page of its topic and add a
  `CHANGELOG.md` entry. Regenerate the CLI and options references when CLIs or
  config fields change; `--check` fails on stale pages.
- Documentation describes supported APIs and behaviour. Experiment receipts,
  incident reports, migration records and study plans are kept with the
  relevant project, not in this repository.

## Commits and changes

- Name new branches for their topic or capability, not `codex/...`.
- Make small, focused commits with one-line imperative messages, staging
  explicit paths. Run the hooks and the relevant tests before committing.
- Never commit results, checkpoints, datasets, generated figures or files over
  500 KB; `outputs/` is for local artifacts only.
- Do not push, rewrite history, or delete branches, tags or worktrees without
  the maintainer's explicit approval.
- Run scientific jobs from an immutable commit; each run records the imported
  package path, source commit, resolved configuration, seeds and environment in
  its output directory.
