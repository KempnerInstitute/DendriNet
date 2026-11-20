# DendriNet

DendriNet is a PyTorch library for networks built from dendritic neurons.
It provides branched compartments, sparse excitatory and inhibitory synapses,
shunting and additive integration, population networks, training and analysis.

## Install and train a first model

```bash
python -m pip install dendritic-modeling
dendritic-example synthetic_population --output demo.yaml
dendritic-train demo.yaml --epochs 1
```

This small two-class demonstration generates its data locally. It does not
download a dataset or reproduce a paper result. To start with MNIST instead,
export `mnist_population`; its first run downloads MNIST.

Python 3.10+ and PyTorch 2.8+ are required. The tested CPU environments use
Linux and Python 3.10, 3.11 and 3.12. CUDA is optional. Exact scientific
reproduction requires the study's recorded dependencies, source and settings.

## Graphical model editor

Install the companion distribution with `python -m pip install --upgrade dendrinet-studio` to draw dendritic
populations, trace incoming E and I connections, and compare training runs on
your computer. Launch it with `dendrinet-studio --workspace my-runs`. Designs
export as YAML that the core training command accepts. The
[Studio guide](https://kempnerinstitute.github.io/DendriNet/studio.html)
includes a first run with generated data, a worked synapse-count example,
and [instructions for training on a cluster GPU](https://kempnerinstitute.github.io/DendriNet/studio.html#studio-cluster-gpu).

- [Documentation](https://kempnerinstitute.github.io/DendriNet/)
- [Source and examples](https://github.com/KempnerInstitute/DendriNet)
- [Issues](https://github.com/KempnerInstitute/DendriNet/issues)
- [License: MIT](https://github.com/KempnerInstitute/DendriNet/blob/develop/LICENSE)

The import name is `dendritic_modeling`. Optional extras include `test`, `dev`,
`docs`, `gpu`, `plotting`, `transformers` and `compression`.
