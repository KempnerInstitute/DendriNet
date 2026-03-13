# Documentation

This directory contains the maintained Sphinx documentation for the current
codebase. The `.rst` files here are the authoritative user-facing docs.

## Key Pages

- `installation.rst`
- `quickstart.rst`
- `network_architectures.rst`
- `training_strategies.rst`
- `datasets.rst`
- `analysis.rst`
- `configuration_guide.rst`
- `sweeps.rst`
- `config_reference.rst`
- `fsdp_troubleshooting.rst`

## Build

```bash
cd docs
make html
```

or

```bash
python -m sphinx -b html docs docs/_build/html
```

View the generated site at `docs/_build/html/index.html`.
