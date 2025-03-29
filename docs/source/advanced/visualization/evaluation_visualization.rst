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
        - branch_info                     # Plot branch information
        - ablation                        # Plot ablation results
      parameters:
        save_figures: true
        figure_format: "png"

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

Branch Information Analysis
---------------------------

The framework includes tools for information-theoretic analysis of branch behavior.

Branch Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``analyze_dendritic_information`` function calculates various information metrics for branch activations:

.. code-block:: python

    def analyze_dendritic_information(model, dataset, device='cuda',
                                      use_post_shunting=False, output_format='auto'):
        """
        Analyze dendritic information metrics for a trained model.
        
        Parameters
        ----------
        model : torch.nn.Module
            The trained model.
        dataset : torch.utils.data.Dataset
            The dataset to evaluate on.
        device : str
            Device to perform computation on ('cuda' or 'cpu').
        use_post_shunting : bool
            Whether to use activations after potential shunting.
        output_format : str
            'auto', 'legacy', or 'new'. Specifies the output dictionary format.
            
        Returns
        -------
        dict
            Dictionary containing various information metrics like mutual information, entropy, etc.
        """

This function helps quantify how much information about the class labels is present in each branch's activation.

Ablation Studies
----------------

The framework provides tools for ablation studies to understand the contribution of different branches.

Branch Ablation
^^^^^^^^^^^^^^^

The ``analyze_branch_ablation`` function performs comprehensive ablation analysis:

.. code-block:: python

    def analyze_branch_ablation(model, dataset, ablation_mode='direct', device='cpu'):
        """
        Perform ablation analysis for all branches in the model.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Dataset to evaluate on
        ablation_mode : str
            Ablation mode: 'direct' or 'indirect'
        device : str
            Device to use for computation
            
        Returns
        -------
        dict
            Dictionary containing ablation results including:
            - baseline_acc: Accuracy before ablation
            - branch_importances: Importance matrix for each branch
        """

This function helps quantify the importance of each branch for the network's performance by systematically ablating branches and measuring the change in accuracy.

Layer-wise Ablation
^^^^^^^^^^^^^^^^^^

The ``analyze_layerwise_ablation`` function performs ablation at the layer level:

.. code-block:: python

    def analyze_layerwise_ablation(model, dataset, train_data=None, valid_data=None, test_data=None, save_path=None, device='cuda'):
        """
        Perform ablation analysis for each layer in the model.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Main dataset for evaluation
        train_data : torch.utils.data.Dataset, optional
            Training dataset (used for retraining mode if applicable)
        valid_data : torch.utils.data.Dataset, optional
            Validation dataset (used for retraining mode if applicable)
        test_data : torch.utils.data.Dataset, optional
            Test dataset for evaluation
        save_path : str, optional
            Path to save results
        device : str, default='cuda'
            Device to use for computation ('cuda' or 'cpu')
            
        Returns
        -------
        dict
            Dictionary containing ablation results including accuracy per layer.
        """

This function helps understand the importance of each layer in the dendritic network.

Using the Visualization Manager
------------------------------

The framework provides a PlotManager class that can automate many of these visualizations:

.. code-block:: python

    from dendritic_modeling.visualization_manager import PlotManager
    
    # Create a plot manager
    plot_manager = PlotManager(
        model=model,
        data={'train': train_data, 'valid': valid_data, 'test': test_data},
        save_path="visualization_outputs",
        device='cuda',
        plots=['weights', 'activations', 'branch_info', 'ablation']
    )
    
    # Generate all plots
    plot_manager.generate_plots()
    
    # Generate plots for specific epoch
    plot_manager.generate_plots(epoch=10)

Best Practices for Evaluation and Visualization
-----------------------------------------------

1. **Compare with baselines**: Always compare dendritic networks with standard neural networks
2. **Analyze branch specialization**: Use branch information analysis to understand specialization
3. **Perform ablation studies**: Use ablation to understand the role of each branch
4. **Visualize weights**: Examine weight matrices to understand learned patterns
5. **Monitor activations**: Track how activations change for different inputs
6. **Use appropriate metrics**: Choose metrics that align with your research questions

Visualizations are particularly important for dendritic networks because they help reveal the specialized computational roles of different branches and how they work together to process information. 