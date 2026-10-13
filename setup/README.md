# Setup Directory

- `environment.yml`: a conda environment with Python 3.10 and pip.
- `constraints/`: pip constraints files, each pinning a stack the core test
  suite passed on. `core-py310-torch2.9.1.txt` pins Python 3.10, PyTorch 2.9.1
  (CUDA 12.8 wheels from PyPI) and NumPy 2.2.
- `freeze_constraints.py`: writes such a file from the running interpreter
  (the dependency closure of `pyproject.toml` and the requested extras).

## Usage

```bash
conda env create -f setup/environment.yml
conda activate dendritic_env
pip install -e ".[test]" -c setup/constraints/core-py310-torch2.9.1.txt
```

See [INSTALL.md](../INSTALL.md) for complete instructions.
