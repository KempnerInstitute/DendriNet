Dendritic Modeling Framework Documentation
==========================================

.. image:: _static/dendritic_diagram.png
   :alt: Dendritic neuron architecture
   :align: center
   :width: 600px

**A comprehensive PyTorch-based framework for biologically-inspired neural networks with dendritic computation, excitation-inhibition dynamics, and advanced analysis tools.**

Welcome to the Dendritic Modeling Framework - your complete toolkit for building, training, and analyzing neural networks that capture the sophisticated computational properties of biological dendrites. Whether you're a neuroscience researcher exploring dendritic mechanisms or an ML practitioner seeking bio-inspired architectures, this framework provides the tools you need.

**Quick Start**
----------------

Get up and running in minutes:

.. code-block:: bash

   # Installation
   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling
   conda env create -f environment.yml
   conda activate dendritic_env
   pip install -e .
   
   # Run your first experiment
   python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml

**Key Features**
----------------

* **Advanced Architectures**: EINet, DendriNet, and hybrid models with configurable dendritic morphologies
* **Biological Realism**: Shunting inhibition, Dale's principle, sparse connectivity patterns
* **Scalability**: From single GPU experiments to large-scale FSDP training on clusters
* **Analysis Suite**: Information theory, ablation studies, visualization tools
* **Performance Optimization**: Mixed precision, gradient checkpointing, dynamic sparse training
* **Extensive Datasets**: Vision, synthetic, and biological task datasets

**Documentation Structure**
---------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction
   installation
   quickstart
   core_concepts

.. toctree::
   :maxdepth: 2
   :caption: Architecture & Design

   architectures/index
   configuration/index
   training/index
   datasets/index

.. toctree::
   :maxdepth: 2
   :caption: Analysis & Visualization

   analysis/index
   visualization/index
   performance/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   sweeps/index
   scaling/index
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   faq
   glossary
   troubleshooting
   contributing
   appendix

**Navigation Guide**
---------------------

**Getting started with dendritic modeling?**
   Start with :doc:`introduction` → :doc:`installation` → :doc:`quickstart`

**Want to understand the science?**
   Read :doc:`core_concepts` → :doc:`architectures/index` → :doc:`analysis/index`

**Building custom models?**
   Check :doc:`configuration/index` → :doc:`architectures/custom_design` → :doc:`api/index`

**Scaling to large datasets?**
   See :doc:`performance/index` → :doc:`scaling/index` → :doc:`troubleshooting`

**Running experiments?**
   Follow :doc:`training/index` → :doc:`sweeps/index` → :doc:`tutorials/index`

**Research Applications**
-------------------------

The framework facilitates research in several important areas:

* **Dendritic Computation**: Understanding how dendrites process information
* **E/I Balance**: Exploring excitation-inhibition dynamics in networks  
* **Sparse Connectivity**: Developing efficient bio-inspired architectures
* **Learning Mechanisms**: Testing biologically-plausible training strategies
* **Information Processing**: Analyzing neural information flow and capacity

**Performance & Scalability**
-----------------------------

.. list-table:: Framework Capabilities
   :header-rows: 1
   :widths: 30 35 35

   * - Scale
     - Model Size
     - Recommended Setup
   * - Small Experiments
     - < 1M parameters
     - Single GPU, Standard training
   * - Medium Models
     - 1M - 100M parameters  
     - Multi-GPU, Mixed precision
   * - Large Models
     - 100M - 1B+ parameters
     - FSDP, Gradient checkpointing
   * - Research Clusters
     - Multi-billion parameters
     - Multi-node FSDP, CPU offload

**Current Features** (v0.1.0)
---------------------------

* **FSDP Support**: Scale to billion-parameter dendritic models
* **Unified Sweeps**: Comprehensive parameter exploration framework
* **Advanced Analysis**: Information-theoretic tools with multiple estimators
* **Performance Optimization**: Mixed precision, profiling, memory management
* **Rich Visualizations**: Professional figures and interactive plots

**Latest Updates**

* **v0.1.0**: Initial release with core dendritic modeling capabilities
* **Documentation**: Comprehensive user guides, tutorials, and API reference
* **Analysis Tools**: Information theory, ablation studies, visualization suite
* **Training Strategies**: Multiple bio-inspired learning approaches

**Community & Support**
-----------------------

* **GitHub Issues**: Report bugs and request features
* **Documentation**: Comprehensive guides and API reference
* **Examples**: Jupyter notebooks and complete workflows
* **Contributing**: Guidelines for extending the framework

**Citation**: If you use this framework in your research, please cite our paper (see :doc:`appendix` for details).

**Need Help?** Check our :doc:`faq`, search the :doc:`glossary`, or open an issue on `GitHub <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`  
* :ref:`search`
