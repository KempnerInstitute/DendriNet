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
- `CONFIGURATION_GUIDE.rst`
- `sweeps.rst`
- `config_reference.rst`
- `FSDP_TROUBLESHOOTING.rst`

## Build

```bash
cd docs
make html
```

or

```bash
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` after the build finishes.
