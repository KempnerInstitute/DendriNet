# Setup Directory

Contains the conda environment specification. The package supports Python
3.10+; the default environment pins Python 3.11 and a CUDA 12.1 PyTorch stack.

## Usage

```bash
conda env create -f setup/environment.yml
conda activate dendritic_env
pip install -e .
```

See [docs/installation.rst](../docs/installation.rst) for complete instructions.
