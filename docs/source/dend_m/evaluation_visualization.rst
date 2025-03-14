.. _evaluation_visualization:

Evaluation and Visualization
============================

This document explains the evaluation metrics and visualization tools available in the dendritic modeling framework for analyzing dendritic network behavior.

Overview
--------

The dendritic modeling framework includes specialized tools for evaluating model performance and visualizing the unique computational properties of dendritic networks. These tools help researchers understand how information flows through dendritic branches and how the network processes information.

Configuration
-------------

Visualization settings are configured in the ``visualization`` section of the configuration file:

.. code-block:: yaml

    visualization:
      enabled: false                      # Whether to generate visualizations
      visualize_epochs: [10, 30]          # Epochs at which to produce visualizations
      plots:
        - weights                         # Plot dendritic or EI-net weights
        - activations                     # Plot per-digit activation distribution
        - gradients                       # Plot synapse or branch gradients
        - profiles                       # Plot dendritic compartment profiles
        - branch_info                    # Plot mutual information
        - ablation                       # Plot ablation results

Performance Evaluation
----------------------

The framework provides standard evaluation metrics for classification and regression tasks.

Basic Evaluation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``utils.py`` module includes functions for evaluating model performance:

.. code-block:: python

    def accuracy_score(classifier, inputs, labels):
        """
        Compute classification accuracy.
        
        Parameters
        ----------
        classifier : torch.nn.Module
            The classifier model
        inputs : torch.Tensor
            Input tensor
        labels : torch.Tensor
            Label tensor
            
        Returns
        -------
        float
            Classification accuracy
        """

A more comprehensive evaluation function is also available:

.. code-block:: python

    def evaluate_accuracy(classifier, train_ds=None, valid_ds=None, test_ds=None, 
                          save_path=None, filename='accuracy'):
        """
        Evaluate accuracy on train, validation, and test datasets.
        
        Parameters
        ----------
        classifier : torch.nn.Module
            The classifier model
        train_ds, valid_ds, test_ds : torch.utils.data.Dataset or None
            The datasets to evaluate on
        save_path : str or None
            Path to save results
        filename : str
            Filename for saving results
            
        Returns
        -------
        tuple
            (train_accuracy, valid_accuracy, test_accuracy)
        """

Weight Visualization
--------------------

The framework provides tools for visualizing the weights of dendritic networks.

EINet Parameter Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``plot_einet_params`` function visualizes the weights of ExcitationInhibitionNetwork models:

.. code-block:: python

    def plot_einet_params(einet, save_root, reshape_fn=None, reshape_exc_syn=False, 
                           logspace=False, save_in_dir=False, filename='image'):
        """
        Plot EINet parameters including weights, branch connections, etc.
        
        Parameters
        ----------
        einet : ExcitationInhibitionNetwork
            The network to visualize
        save_root : str
            Directory to save visualizations
        reshape_fn : callable or None
            Function to reshape weights for visualization (e.g., to 2D images)
        reshape_exc_syn : bool
            Whether to reshape excitatory synaptic weights
        logspace : bool
            Whether weights are in log space
        save_in_dir : bool
            Whether to save in a subdirectory
        filename : str
            Base filename for saving
        """

This function creates visualizations of the network's weight matrices, which is especially useful for understanding what patterns the dendritic branches have learned to recognize.

Excitation-Inhibition Weight Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more detailed visualization of excitatory and inhibitory weights:

.. code-block:: python

    def plot_excitation_inhibition_weights(soma_branch, save_path, 
                                        excitatory_synapse_weight, 
                                        excitatory_synapse_pruned,
                                        excitatory_synapse_mask=None,
                                        inhibitory_synapse_weight=None, 
                                        inhibitory_synapse_pruned=None,
                                        inhibitory_synapse_mask=None,
                                        reshape_fn=None, reshape_exc_syn=False, 
                                        reshape_inh_syn=False, logspace=False,
                                        reactivation_curve=None, linspace=None,
                                        save_in_dir=False, filename='image'):
        """
        Detailed visualization of excitatory and inhibitory weights.
        """

Activation Visualization
------------------------

The framework provides tools for visualizing activations of dendritic branches.

EINet Activation Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``plot_einet_activations`` function visualizes activations in the network:

.. code-block:: python

    def plot_einet_activations(einet, input_list, save_root, save_in_dir=False, filename='image'):
        """
        Plot EINet activations for different inputs.
        
        Parameters
        ----------
        einet : ExcitationInhibitionNetwork
            The network to analyze
        input_list : list
            List of input tensors (typically one per class)
        save_root : str
            Directory to save visualizations
        save_in_dir : bool
            Whether to save in a subdirectory
        filename : str
            Base filename for saving
        """

This function helps understand how the network processes different inputs, showing which branches activate for each class.

Activation Component Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more detailed analysis of activation components:

.. code-block:: python

    def einet_activations_to_csv(einet, input_list, save_root, filename='activation_components'):
        """
        Save activation components to CSV for detailed analysis.
        """

Additional activation visualization functions include:

- ``plot_activation_tuning``: Visualize the tuning of branches to different inputs
- ``plot_voltage_before_activation``: Visualize membrane potentials before activation

Gradient Visualization
----------------------

The framework provides tools for visualizing gradients in the network.

EINet Gradient Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``plot_einet_gradients`` function visualizes gradients:

.. code-block:: python

    def plot_einet_gradients(model, inputs, labels, save_root, reshape_fn=None, 
                           reshape_exc_syn=False, save_in_dir=False, filename='image'):
        """
        Plot gradients in the EINet.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        inputs : torch.Tensor
            Input tensor to compute gradients
        labels : torch.Tensor
            Label tensor to compute gradients
        save_root : str
            Directory to save visualizations
        reshape_fn : callable or None
            Function to reshape gradients for visualization
        reshape_exc_syn : bool
            Whether to reshape excitatory synaptic gradients
        save_in_dir : bool
            Whether to save in a subdirectory
        filename : str
            Base filename for saving
        """

This function helps understand how the network learns by visualizing the flow of gradients through the dendritic branches.

Branch Information Analysis
---------------------------

The framework includes tools for information-theoretic analysis of branch behavior.

Branch Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``compute_MI`` function calculates mutual information between branch activations and labels:

.. code-block:: python

    def compute_MI(activations, labels, n_bins=10):
        """
        Estimate mutual information between activations and labels.
        
        Parameters
        ----------
        activations : torch.Tensor
            Activation values
        labels : torch.Tensor
            Label values
        n_bins : int
            Number of bins for discretization
            
        Returns
        -------
        float
            Mutual information value
        """

This function helps quantify how much information about the class labels is present in each branch's activation.

A comprehensive branch information analysis function is also available:

.. code-block:: python

    def analyze_branch_information(model, dataset, batch_size=64, device='cpu', n_bins=10):
        """
        Analyze information content in each branch layer.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze (must support forward_with_branch_outputs)
        dataset : torch.utils.data.Dataset
            Dataset to analyze
        batch_size : int
            Batch size for processing
        device : str
            Device to use for computation
        n_bins : int
            Number of bins for MI estimation
            
        Returns
        -------
        dict
            Dictionary mapping layer indices to MI values
        """

This function helps understand how information flows through the layers of the dendritic network.

Ablation Studies
----------------

The framework provides tools for ablation studies to understand the contribution of different branches.

Branch Ablation
^^^^^^^^^^^^^^^

The ``ablate_branch`` function allows ablating specific branches:

.. code-block:: python

    def ablate_branch(model, layer_idx, branch_ids=None, mode='direct'):
        """
        Zero out weights for branch(es) at the specified layer.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to ablate
        layer_idx : int
            Index of the layer to ablate
        branch_ids : list or None
            IDs of branches to ablate, or None for all branches
        mode : str
            Ablation mode: 'direct' or 'full'
        """

This function helps understand the role of specific branches by measuring performance changes after ablation.

A comprehensive ablation analysis function is also available:

.. code-block:: python

    def analyze_branch_ablation(model, dataset, layer_idx, branch_ids, 
                               mode='direct', device='cpu'):
        """
        Perform ablation analysis for specific branches.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Dataset to evaluate on
        layer_idx : int
            Index of the layer to ablate
        branch_ids : list or None
            IDs of branches to ablate, or None for all branches
        mode : str
            Ablation mode: 'direct' or 'full'
        device : str
            Device to use for computation
            
        Returns
        -------
        tuple
            (baseline_accuracy, results_dict)
        """

This function helps quantify the importance of each branch for the network's performance.

Dendritic Tree Visualization
----------------------------

The framework includes tools for visualizing the structure of dendritic trees.

Tree Graph Visualization
^^^^^^^^^^^^^^^^^^^^^^^^

The ``build_dendritic_tree_graph`` and ``visualize_dendritic_tree`` functions create visualizations of the dendritic tree structure:

.. code-block:: python

    def build_dendritic_tree_graph(model, branch_info=None, ablation_info=None):
        """
        Build a graph representation of the dendritic tree.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to visualize
        branch_info : dict or None
            Branch information from analyze_branch_information
        ablation_info : dict or None
            Ablation results from analyze_branch_ablation
            
        Returns
        -------
        networkx.Graph
            Graph representation of the dendritic tree
        """

These visualizations help understand the hierarchical structure of the dendritic network and the information flow through it.

Comprehensive Visualization Script
----------------------------------

The ``visualize.py`` script provides a comprehensive visualization pipeline:

.. code-block:: python

    def get_and_save_performance_and_visuals(clf, data, n_tasks, input_list, 
                                           reshape_fn, reshape_exc_syn, logspace,
                                           save_path, filename):
        """
        Generate and save comprehensive visualizations for a model.
        
        Parameters
        ----------
        clf : torch.nn.Module
            The classifier model
        data : dict
            Dictionary with 'train', 'valid', and 'test' datasets
        n_tasks : int
            Number of tasks
        input_list : list
            List of input tensors
        reshape_fn : callable
            Function to reshape weights for visualization
        reshape_exc_syn : bool
            Whether to reshape excitatory synaptic weights
        logspace : bool
            Whether weights are in log space
        save_path : str
            Directory to save visualizations
        filename : str
            Base filename for saving
        """

This function generates a complete set of visualizations for a trained model, including performance metrics, parameter visualizations, gradient visualizations, and activation visualizations.

Best Practices for Evaluation and Visualization
-----------------------------------------------

1. **Compare with baselines**: Always compare dendritic networks with standard neural networks
2. **Analyze branch specialization**: Use branch information analysis to understand specialization
3. **Perform ablation studies**: Use ablation to understand the role of each branch
4. **Visualize weights**: Examine weight matrices to understand learned patterns
5. **Monitor activations**: Track how activations change for different inputs
6. **Use appropriate metrics**: Choose metrics that align with your research questions

Visualizations are particularly important for dendritic networks because they help reveal the specialized computational roles of different branches and how they work together to process information. 