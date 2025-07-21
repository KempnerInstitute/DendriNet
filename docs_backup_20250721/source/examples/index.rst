Examples
========

This section provides practical examples and tutorials demonstrating how to use the Dendritic Modeling Framework for various applications.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started Examples

   basic_classification
   quickstart_tutorial
   configuration_examples

.. toctree::
   :maxdepth: 2
   :caption: Network Architecture Examples

   excitation_inhibition_networks
   dendritic_computation
   custom_architectures

.. toctree::
   :maxdepth: 2
   :caption: Training Examples

   training_strategies
   branch_wise_learning
   distributed_training

.. toctree::
   :maxdepth: 2
   :caption: Analysis Examples

   information_analysis
   biological_plausibility
   ablation_studies

.. toctree::
   :maxdepth: 2
   :caption: Application Examples

   image_classification
   sequence_modeling
   neuroscience_applications

.. toctree::
   :maxdepth: 2
   :caption: Advanced Examples

   custom_components
   research_workflows
   performance_optimization

Overview
--------

The examples are organized into several categories, progressing from basic usage to advanced applications:

Getting Started Examples
~~~~~~~~~~~~~~~~~~~~~~~

**Basic Classification**: Simple MNIST classification example showing fundamental framework usage.

**Quickstart Tutorial**: Step-by-step tutorial covering installation, basic usage, and first experiments.

**Configuration Examples**: Comprehensive examples of YAML and Python configuration systems.

Network Architecture Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Excitation-Inhibition Networks**: Detailed examples of E/I network creation and configuration.

**Dendritic Computation**: Examples focusing on dendritic processing and shunting inhibition.

**Custom Architectures**: Building custom network architectures and hybrid models.

Training Examples
~~~~~~~~~~~~~~~~

**Training Strategies**: Comparison of different training strategies (MLE, branch-wise, curriculum).

**Branch-wise Learning**: Detailed examples of biologically-inspired training approaches.

**Distributed Training**: Scaling training across multiple GPUs and nodes.

Analysis Examples
~~~~~~~~~~~~~~~~

**Information Analysis**: Computing and interpreting information theory metrics.

**Biological Plausibility**: Analyzing adherence to biological principles.

**Ablation Studies**: Systematic component analysis and importance ranking.

Application Examples
~~~~~~~~~~~~~~~~~~~

**Image Classification**: Computer vision applications with dendritic networks.

**Sequence Modeling**: Temporal processing and sequence learning.

**Neuroscience Applications**: Modeling biological neural computation.

Advanced Examples
~~~~~~~~~~~~~~~~

**Custom Components**: Creating custom layers, loss functions, and training strategies.

**Research Workflows**: Complete research pipelines from hypothesis to publication.

**Performance Optimization**: Optimizing models for speed and memory efficiency.

Quick Start
-----------

For newcomers to the framework, we recommend starting with these examples:

1. **Basic Classification** - Learn fundamental concepts
2. **Quickstart Tutorial** - Complete walkthrough
3. **Configuration Examples** - Master the configuration system
4. **Excitation-Inhibition Networks** - Understand core architectures

Example Code Structure
---------------------

All examples follow a consistent structure:

.. code-block:: text

   example_name/
   ├── README.md              # Example description and instructions
   ├── config.yaml           # Configuration file
   ├── main.py               # Main execution script
   ├── analysis.py           # Analysis and visualization
   ├── requirements.txt      # Additional dependencies
   └── results/              # Output directory
       ├── figures/          # Generated figures
       ├── models/           # Saved models
       └── logs/             # Training logs

Running Examples
---------------

Each example can be run independently:

.. code-block:: bash

   # Navigate to example directory
   cd docs/source/examples/basic_classification
   
   # Install additional dependencies if needed
   pip install -r requirements.txt
   
   # Run the example
   python main.py
   
   # Or use configuration file
   python main.py --config config.yaml

Most examples also include Jupyter notebooks for interactive exploration:

.. code-block:: bash

   # Launch Jupyter notebook
   jupyter notebook example_notebook.ipynb

Example Categories
-----------------

Beginner Examples
~~~~~~~~~~~~~~~~

These examples are designed for users new to the framework:

* **Basic Classification**: MNIST classification with E/I networks
* **Simple Regression**: Boston housing price prediction
* **Data Loading**: Using built-in and custom datasets
* **Model Evaluation**: Basic performance evaluation

Intermediate Examples
~~~~~~~~~~~~~~~~~~~~

Examples for users familiar with the basics:

* **Custom Networks**: Building application-specific architectures
* **Training Strategies**: Comparing different learning approaches
* **Hyperparameter Tuning**: Systematic parameter optimization
* **Analysis Workflows**: Comprehensive model analysis

Advanced Examples
~~~~~~~~~~~~~~~~

Examples for experienced users and researchers:

* **Novel Architectures**: Cutting-edge network designs
* **Research Pipelines**: Complete research workflows
* **Performance Optimization**: Speed and memory optimization
* **Biological Modeling**: Detailed neuroscience applications

Contributing Examples
--------------------

We welcome contributions of new examples! To contribute:

1. **Choose a Topic**: Select a topic not covered by existing examples
2. **Follow Structure**: Use the standard example structure
3. **Include Documentation**: Provide clear README and comments
4. **Test Thoroughly**: Ensure the example runs correctly
5. **Submit Pull Request**: Follow the contribution guidelines

Example Guidelines:

* **Clear Purpose**: Each example should have a clear learning objective
* **Complete Code**: Include all necessary code and dependencies
* **Good Documentation**: Provide detailed explanations and comments
* **Reproducible Results**: Ensure consistent results across runs
* **Appropriate Complexity**: Match complexity to target audience

Interactive Examples
-------------------

Many examples include interactive components:

**Jupyter Notebooks**: Interactive exploration and experimentation

**Streamlit Apps**: Web-based interactive demonstrations

**Plotting Tools**: Interactive visualizations and analysis

**Configuration GUIs**: Graphical configuration interfaces

To run interactive examples:

.. code-block:: bash

   # Jupyter notebooks
   jupyter notebook
   
   # Streamlit apps
   streamlit run app.py
   
   # Interactive plotting
   python interactive_plot.py

Example Datasets
---------------

Examples use various datasets:

**Built-in Datasets**:
* MNIST (handwritten digits)
* CIFAR-10 (natural images)
* Boston Housing (regression)
* Synthetic datasets (lines, circles, etc.)

**External Datasets**:
* ImageNet (computer vision)
* Penn Treebank (natural language)
* Neuromorphic datasets (spike trains)

**Custom Datasets**:
* Biological neural recordings
* Synthetic dendritic data
* Task-specific datasets

Getting Help
-----------

If you encounter issues with examples:

1. **Check Requirements**: Ensure all dependencies are installed
2. **Read Documentation**: Review the example README and comments
3. **Check Issues**: Look for similar issues on GitHub
4. **Ask Questions**: Use GitHub discussions for questions
5. **Report Bugs**: Submit bug reports with example details

Common Issues:

* **Missing Dependencies**: Install requirements.txt
* **Path Issues**: Run from correct directory
* **Memory Issues**: Reduce batch size or model size
* **CUDA Issues**: Check GPU availability and drivers

Next Steps
---------

After exploring examples:

1. **User Guide**: Return to :doc:`../user_guide/index` for detailed documentation
2. **API Reference**: Consult :doc:`../api/index` for API details
3. **Advanced Topics**: Explore :doc:`../advanced/index` for advanced techniques
4. **Contributing**: Consider contributing your own examples 