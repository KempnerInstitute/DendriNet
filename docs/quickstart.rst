Quickstart Guide
================

Get up and running with the Dendritic Modeling Framework in under 5 minutes! This guide will walk you through your first experiment step-by-step.

.. note::
   Already have the framework installed? Jump to :ref:`first-experiment`.

Prerequisites Check
------------------

Before we start, let's make sure you have everything you need:

.. code-block:: bash

   # Check Python version (need 3.8+)
   python --version
   
   # Check if you have conda
   conda --version
   
   # Check if you have git
   git --version

If any of these commands fail, please install the missing software first.

Installation (2 minutes)
------------------------

Quick installation using conda:

.. code-block:: bash

   # 1. Clone the repository
   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling

   # 2. Create conda environment (this may take 1-2 minutes)
   conda env create -f environment.yml
   
   # 3. Activate environment
   conda activate dendritic_env
   
   # 4. Install the package
   pip install -e .

Verify the installation:

.. code-block:: bash

   python -c "import dendritic_modeling; print('✓ Installation successful!')"

Expected output: ``✓ Installation successful!``

.. _first-experiment:

Your First Experiment (3 minutes)
---------------------------------

Let's run a simple dendritic network experiment on MNIST:

Step 1: Run the Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run your first dendritic experiment
   python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml

You should see output similar to:

.. code-block:: text

   [INFO] Loaded configuration from configs/config_exp.yaml
   [INFO] Configuration validation passed
   [INFO] Setting up EINet model...
   [INFO] Training on MNIST dataset
   [INFO] Starting training for 50 epochs
   
   Epoch 1/50: Train Loss: 2.145, Train Acc: 0.234, Valid Acc: 0.312
   Epoch 2/50: Train Loss: 1.856, Train Acc: 0.445, Valid Acc: 0.523
   ...
   Epoch 50/50: Train Loss: 0.123, Train Acc: 0.967, Valid Acc: 0.945
   
   [INFO] Training completed successfully!
   [INFO] Results saved to: results/exp_run_[timestamp]/

**Training Time**: ~2-3 minutes on a modern GPU, ~5-10 minutes on CPU.

Step 2: Check Your Results
~~~~~~~~~~~~~~~~~~~~~~~~~

After training completes, explore your results:

.. code-block:: bash

   # List the output directory
   ls results/exp_run_*/
   
   # You should see:
   # checkpoints/  - Model checkpoints
   # logs/        - Training logs
   # analysis/    - Analysis results  
   # config.json  - Saved configuration

Step 3: View Training Progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # View the training log
   cat results/exp_run_*/logs/training.log

You should see detailed training information including loss curves and accuracy metrics.

Understanding What Just Happened
--------------------------------

You just trained an **EINet (Excitation-Inhibition Network)** with these key features:

**Model Architecture**:
- 1 layer with 10 excitatory neurons
- 40 inhibitory neurons providing feedback
- Shunting inhibition (V = E/(E+I+1))
- Parametric tanh reactivation functions

**Training Details**:
- Dataset: MNIST (handwritten digits)
- Strategy: Standard backpropagation  
- Loss: Categorical cross-entropy
- Optimizer: Adam with learning rate 0.01

**Performance**:
- Expected accuracy: ~94-95% on MNIST
- Parameters: ~50K (much smaller than standard MLPs)
- Biological realism: High (separate E/I populations)

Quick Customization
------------------

Try modifying the experiment with command-line overrides:

Different Learning Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Try two-step training (local pretraining + global fine-tuning)
   python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml \
       --learning_strategy two_step

Different Dataset
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Train on CIFAR-10 instead of MNIST
   python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml \
       --task.dataset cifar10

Larger Network
~~~~~~~~~~~~~

Create a quick custom config:

.. code-block:: yaml

   # Save as my_config.yaml
   model:
     task: classification
     core_network:
       type: EINet
       parameters:
         excitatory_layer_sizes: [64, 32]    # Larger network
         inhibitory_layer_sizes: [32, 16]
         use_shunting: true
   train:
     learning_strategy: standard
     trainer:
       epochs: 30
   task:
     dataset: mnist

.. code-block:: bash

   # Run your custom config
   python src/dendritic_modeling/scripts/train_experiments.py my_config.yaml

Exploring Results in Detail
---------------------------

Performance Metrics
~~~~~~~~~~~~~~~~~~

Your results directory contains:

.. code-block:: text

   results/exp_run_[timestamp]/
   ├── analysis/
   │   └── performance_metrics.json    # Accuracy, loss, etc.
   ├── checkpoints/
   │   ├── best_model.pt              # Best performing model
   │   └── final_model.pt             # Final epoch model
   ├── logs/
   │   ├── training.log               # Detailed training log
   │   └── loss_curves.png            # Training/validation curves
   └── config.json                    # Experiment configuration

Load and Inspect Results
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   import torch
   
   # Load performance metrics
   with open('results/exp_run_[timestamp]/analysis/performance_metrics.json') as f:
       metrics = json.load(f)
       print(f"Final accuracy: {metrics['test_accuracy']:.3f}")
   
   # Load the trained model
   model = torch.load('results/exp_run_[timestamp]/checkpoints/best_model.pt')
   print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

Visualizing Network Activity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you enabled visualization in your config, you'll also find:

.. code-block:: text

   results/exp_run_[timestamp]/
   ├── figures/
   │   ├── weight_distributions.png    # Synaptic weight patterns
   │   ├── activation_patterns.png     # Neural activation analysis
   │   └── branch_information.png      # Dendritic branch activity

Common Issues and Solutions
--------------------------

"ModuleNotFoundError"
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Make sure you activated the environment
   conda activate dendritic_env
   
   # Reinstall if needed
   pip install -e . --force-reinstall

"CUDA out of memory"
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Use CPU training
   python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml \
       --trainer.device cpu

"Training is slow"
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reduce epochs for quick testing
   python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml \
       --trainer.epochs 10

What's Next?
-----------

Now that you've run your first experiment, here's what to explore next:

**Understand the Science**
   Read :doc:`core_concepts` to understand shunting inhibition, E/I dynamics, and dendritic computation.

**Design Custom Networks**
   Explore :doc:`architectures/index` to learn about EINet, DendriNet, and hybrid architectures.

**Master Configuration**
   Check out :doc:`configuration/index` for complete control over your experiments.

**Analyze Your Models**
   Learn about :doc:`analysis/index` for information theory, ablation studies, and visualization.

**Scale Up**
   When ready for larger experiments, see :doc:`performance/index` and :doc:`scaling/index`.

Example Tutorials
-----------------

Try these step-by-step tutorials:

**Beginner**:
- :doc:`tutorials/mnist_classification` - Complete MNIST walkthrough
- :doc:`tutorials/comparing_architectures` - EINet vs MLP comparison

**Intermediate**:
- :doc:`tutorials/information_analysis` - Analyze information flow
- :doc:`tutorials/custom_datasets` - Use your own data

**Advanced**:
- :doc:`tutorials/large_scale_training` - Multi-GPU training with FSDP
- :doc:`tutorials/parameter_sweeps` - Systematic parameter exploration

Interactive Examples
--------------------

For hands-on learning, explore our Jupyter notebooks:

.. code-block:: bash

   # Start Jupyter (if installed)
   jupyter notebook docs/notebooks/
   
   # Or use Jupyter Lab
   jupyter lab docs/notebooks/

Available notebooks:
- ``quickstart_interactive.ipynb`` - This guide as an interactive notebook with runnable code
- ``architecture_comparison.ipynb`` - Compare EINet, DendriNet, and MLP architectures
- ``information_analysis.ipynb`` - Explore information theory analysis tools
- ``parameter_sweeps.ipynb`` - Learn to run systematic parameter exploration

.. note::
   Interactive notebooks are also available on Google Colab for easy experimentation without local installation.

Getting Help
------------

If you run into issues:

1. **Check the logs**: Training logs often contain helpful error messages
2. **Consult troubleshooting**: See :doc:`troubleshooting` for common solutions
3. **Ask for help**: Create an issue on `GitHub <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_
4. **Join discussions**: Participate in community discussions

Congratulations!
----------------

You've successfully:
- Installed the Dendritic Modeling Framework
- Trained your first dendritic neural network  
- Achieved ~95% accuracy on MNIST
- Learned about E/I dynamics and shunting inhibition
- Explored the results structure

You're now ready to dive deeper into the fascinating world of dendritic computation!

.. tip::
   **Bookmark this page** - you'll likely want to reference these commands as you explore more advanced features.

.. note::
   **Feedback Welcome!** 
   
   Found this tutorial helpful? Have suggestions for improvement? 
   
   - Open an issue on `GitHub <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_
   - Edit this page directly via the "Edit on GitHub" link
   - Rate this tutorial: `Quick Feedback Form <https://forms.gle/example>`_

.. seealso::
   * :doc:`core_concepts` - Deep dive into the science
   * :doc:`configuration/index` - Complete configuration reference  
   * :doc:`tutorials/index` - Step-by-step tutorials
   * :doc:`api/index` - Full API documentation 