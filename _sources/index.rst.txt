DendriNet
=========

DendriNet is a PyTorch library for networks of dendritic neurons. Its Python
package is ``dendritic_modeling``. A DendriNet neuron is a tree of dendritic
branches: each branch reads a sparse set of excitatory and inhibitory inputs,
integrates them by shunting (divisive) or additive inhibition, and passes its
output toward the soma. Neurons are grouped into named excitatory and
inhibitory populations that can be feedforward or recurrent, rate-based or
spiking. Experiments are described in YAML files that the library loads,
validates, trains, analyzes and sweeps.

.. figure:: _static/architecture/system-overview.svg
   :alt: Overview: YAML configuration is loaded into typed settings, the factory builds an encoder, a dendritic core and a decoder, and trainers, sweeps and the analysis manager produce run artifacts.
   :width: 100%

   From a YAML file to trained models and analysis results.

Install the released library and train a small population network on generated
data for one CPU epoch:

.. code-block:: bash

   python -m pip install dendritic-modeling==0.5.0
   dendritic-example synthetic_population --output demo.yaml
   dendritic-train demo.yaml --epochs 1

For a graphical first run, install the companion package with
``python -m pip install dendrinet-studio==0.1.0`` and follow :ref:`studio-first-run`.

Every model has three parts: an **encoder** that prepares the input, a
**core** that holds the dendritic network (or a baseline), and a **decoder**
that maps the core output to the task. :doc:`network_architectures` describes
this structure and the families of cores.

Where to start
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - Page
   * - Install DendriNet and train a first model
     - :doc:`installation`, :doc:`quickstart`
   * - Draw a network, trace E and I inputs and compare training runs
     - :doc:`studio`, :ref:`studio-first-run`
   * - Find a configuration to start from
     - :doc:`examples`
   * - Build populations of dendritic neurons
     - :doc:`population_networks`
   * - Choose integration rules, gates and synapse types
     - :doc:`dendritic_neurons`, :doc:`synapses`, :doc:`temporal_dynamics`
   * - Write and validate a configuration file
     - :doc:`configuration_guide`
   * - Choose a learning rule or train on several GPUs
     - :doc:`training_strategies`, :doc:`distributed_training`
   * - Find the files a run writes
     - :doc:`runs_and_outputs`
   * - Analyze trained models and saved checkpoints
     - :doc:`analysis`, :doc:`analyzers`, :doc:`checkpoint_analysis`
   * - Run parameter sweeps
     - :doc:`sweeps`
   * - Pin a project's code to an exact library commit
     - :doc:`project_provenance`
   * - Look up a component, an option or a command
     - :doc:`available_components`, :doc:`options_reference`,
       :doc:`cli_reference`
   * - Add components from your own package
     - :doc:`extending`

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   releases
   studio
   quickstart
   examples

.. toctree::
   :maxdepth: 1
   :caption: Building models

   network_architectures
   dendritic_neurons
   synapses
   population_networks
   temporal_dynamics
   structured_ei_cores
   conductance_options
   comparison_models

.. toctree::
   :maxdepth: 1
   :caption: Configuration

   configuration_guide
   config_reference

.. toctree::
   :maxdepth: 1
   :caption: Training

   training_strategies
   distributed_training
   performance_kernels

.. toctree::
   :maxdepth: 1
   :caption: Data and analysis

   datasets
   analysis
   analyzers
   checkpoint_analysis

.. toctree::
   :maxdepth: 1
   :caption: Running experiments

   runs_and_outputs
   sweeps
   model_compression
   project_provenance

.. toctree::
   :maxdepth: 1
   :caption: Reference

   available_components
   options_reference
   cli_reference
   api_reference

.. toctree::
   :maxdepth: 1
   :caption: Development

   extending
   engineering_architecture
   contributing
