.. _ablation_studies:

Ablation Studies
================

This document explains how to conduct ablation studies using the dendritic modeling framework to understand the contribution of different components to network performance.

Overview
--------

Ablation studies involve systematically removing or disabling components of a neural network to measure their impact on performance. In dendritic networks, ablation is particularly valuable for understanding the computational role of different dendritic branches, layers, or activation mechanisms.

The dendritic modeling framework provides specialized tools for conducting ablation studies, focusing on:

1. **Branch ablation**: Removing individual dendritic branches
2. **Layer ablation**: Removing entire layers of branches
3. **Feature ablation**: Removing specific input features
4. **Mechanism ablation**: Disabling computational mechanisms (e.g., shunting inhibition)

These studies help identify which components are essential for performance and which components are specialized for specific inputs or tasks.

Branch Ablation
---------------

The most common type of ablation study in dendritic networks is branch ablation, which involves removing individual dendritic branches and measuring the impact on performance.

Branch Ablation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides functions for branch ablation in ``utils.py``:

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
            Ablation mode: 'direct' (zero out specific branches) or 
            'full' (zero out all except specified branches)
        """

This function zeros out the weights of specified branches, effectively removing them from the network.

A comprehensive branch ablation analysis function is also available:

.. code-block:: python

    def analyze_branch_ablation(model, dataset, layer_idx=None, branch_ids=None,
                                mode='direct', device='cpu'):
        """
        Perform ablation analysis for specific branches or dendritic orders.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Dataset to evaluate on
        layer_idx : int or None
            Index of the layer to ablate. If None and branch_ids is None,
            perform dendritic order analysis.
        branch_ids : list or None
            IDs of branches to ablate, or None for all branches
        mode : str
            Ablation mode: 'direct' or 'full'
        device : str
            Device to use for computation
            
        Returns
        -------
        When layer_idx is None and branch_ids is None:
            Dict with dendritic order analysis results
        Otherwise:
            Tuple (baseline_accuracy, results_dict)
        """

This function has two modes of operation:
1. **Branch-specific ablation**: When `layer_idx` is specified, it ablates specific branches and measures the change in performance
2. **Dendritic order ablation**: When both `layer_idx` and `branch_ids` are None, it performs ablation analysis by dendritic order

The dendritic order analysis is especially useful for understanding how different levels of the dendritic hierarchy contribute to network performance.

Example: Branch Ablation
~~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of how to conduct a branch ablation study:

.. code-block:: python

    import torch
    from dendritic_modeling.utils import analyze_branch_ablation
    
    # Load model and dataset
    model = torch.load("path/to/model.pt")
    
    # Perform branch ablation analysis on layer 0
    baseline_acc, branch_results = analyze_branch_ablation(
        model, dataset, layer_idx=0, branch_ids=None, mode='direct'
    )
    
    # Print results
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    for branch_id, (acc, delta) in branch_results.items():
        print(f"Branch {branch_id} ablated: {acc:.4f} (Δ = {delta:.4f})")
    
    # Identify most important branches
    importance = {branch_id: results[branch_id][1] for branch_id in branch_results}
    most_important = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Most important branches:")
    for branch_id, imp in most_important:
        print(f"Branch {branch_id}: Importance = {imp:.4f}")

Example: Dendritic Order Ablation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of how to conduct a dendritic order ablation study:

.. code-block:: python

    from dendritic_modeling.utils import analyze_branch_ablation
    
    # Load model and dataset
    model = torch.load("path/to/model.pt")
    
    # Perform dendritic order ablation analysis
    results = analyze_branch_ablation(
        model, dataset, layer_idx=None, branch_ids=None, device='cuda'
    )
    
    # Print results
    print("Dendritic Order Ablation Results:")
    for i, order in enumerate(results["order"]):
        exc_drop = results["exc_drop"][i]
        inh_drop = results["inh_drop"][i]
        print(f"Order {order}:")
        print(f"  Excitatory ablation: {exc_drop:.4f}")
        print(f"  Inhibitory ablation: {inh_drop:.4f}")
    
    # Calculate percentage drops
    baseline_acc = measure_performance(model, dataset, device='cuda')
    exc_pct = [(drop / baseline_acc) * 100 for drop in results["exc_drop"]]
    inh_pct = [(drop / baseline_acc) * 100 for drop in results["inh_drop"]]
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(results["order"], exc_pct, 'b-', label='Excitatory')
    plt.plot(results["order"], inh_pct, 'r-', label='Inhibitory')
    plt.xlabel('Dendritic Order')
    plt.ylabel('Performance Drop (%)')
    plt.title('Dendritic Order Ablation Results')
    plt.legend()
    plt.grid(True)
    plt.savefig('dendritic_order_ablation.png')
    plt.close()

This code performs ablation for each dendritic order and visualizes the impact on performance.

Layer Ablation
--------------

Layer ablation involves removing entire layers of branches to understand the contribution of each layer to performance.

Layer Ablation Functions
~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides functions for layer ablation:

.. code-block:: python

    def ablate_layer(model, layer_idx):
        """
        Zero out weights for an entire layer.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to ablate
        layer_idx : int
            Index of the layer to ablate
        """

This function zeros out all weights in a specified layer, effectively removing it from the network.

A comprehensive layer ablation analysis function is also available:

.. code-block:: python

    def analyze_layer_ablation(model, dataset, device='cpu'):
        """
        Perform ablation analysis for each layer.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Dataset to evaluate on
        device : str
            Device to use for computation
            
        Returns
        -------
        tuple
            (baseline_accuracy, results_dict)
        """

This function ablates each layer in the network and measures the change in performance, helping identify the layers that are most important for the task.

Example: Layer Ablation
~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of how to conduct a layer ablation study:

.. code-block:: python

    from dendritic_modeling.plot_utils import analyze_layer_ablation
    
    # Load model and dataset
    model = torch.load("path/to/model.pt")
    
    # Perform layer ablation analysis
    baseline_acc, layer_results = analyze_layer_ablation(model, dataset)
    
    # Print results
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    for layer_idx, acc in layer_results.items():
        print(f"Layer {layer_idx} ablated: {acc:.4f} (Δ = {acc - baseline_acc:.4f})")
    
    # Identify most important layers
    importance = {layer_idx: baseline_acc - acc for layer_idx, acc in layer_results.items()}
    most_important = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("Layer importance (descending):")
    for layer_idx, imp in most_important:
        print(f"Layer {layer_idx}: Importance = {imp:.4f}")

This code performs ablation for each layer in the network and ranks the layers by importance.

Feature Ablation
----------------

Feature ablation involves removing specific input features to understand their importance for the network's decision-making.

Feature Ablation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides functions for feature ablation:

.. code-block:: python

    def ablate_features(model, inputs, feature_indices):
        """
        Zero out specific input features.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        inputs : torch.Tensor
            Input tensor
        feature_indices : list
            Indices of features to ablate
            
        Returns
        -------
        torch.Tensor
            Model output with ablated features
        """

This function zeros out specific input features and computes the network's output, helping understand which features are most important for the task.

Example: Feature Ablation
~~~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of how to conduct a feature ablation study:

.. code-block:: python

    import torch
    import numpy as np
    from dendritic_modeling.plot_utils import ablate_features
    
    # Load model and dataset
    model = torch.load("path/to/model.pt")
    
    # Get a batch of inputs
    inputs, labels = next(iter(dataloader))
    
    # Baseline performance
    baseline_output = model(inputs)
    baseline_acc = (baseline_output.argmax(dim=1) == labels).float().mean().item()
    
    # Perform feature ablation
    feature_importance = {}
    for feature_idx in range(inputs.shape[1]):
        # Ablate feature
        ablated_output = ablate_features(model, inputs, [feature_idx])
        ablated_acc = (ablated_output.argmax(dim=1) == labels).float().mean().item()
        
        # Calculate importance
        importance = baseline_acc - ablated_acc
        feature_importance[feature_idx] = importance
    
    # Identify most important features
    most_important = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Most important features:")
    for feature_idx, imp in most_important:
        print(f"Feature {feature_idx}: Importance = {imp:.4f}")

This code performs ablation for each input feature and identifies the features with the largest impact on performance.

Mechanism Ablation
------------------

Mechanism ablation involves disabling specific computational mechanisms to understand their contribution to performance.

Mechanism Ablation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides functions for mechanism ablation:

.. code-block:: python

    def disable_mechanism(model, mechanism_name):
        """
        Disable a specific computational mechanism.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to modify
        mechanism_name : str
            Name of mechanism to disable ('shunting', 'reactivation', etc.)
        """

This function disables specific computational mechanisms, helping understand their contribution to performance.

Example: Shunting Inhibition Ablation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of how to ablate shunting inhibition:

.. code-block:: python

    # Load model and dataset
    model = torch.load("path/to/model.pt")
    
    # Get baseline performance
    baseline_acc = evaluate_accuracy(model, test_dataset)
    
    # Disable shunting inhibition
    for layer in model.ei_layers:
        layer.use_shunting = False
    
    # Evaluate performance without shunting inhibition
    ablated_acc = evaluate_accuracy(model, test_dataset)
    
    # Calculate importance
    importance = baseline_acc - ablated_acc
    print(f"Shunting inhibition importance: {importance:.4f}")

This code disables shunting inhibition and measures the change in performance.

Visualizing Ablation Results
----------------------------

The framework provides functions for visualizing ablation results:

.. code-block:: python

    def plot_branch_ablation(baseline_acc, branch_results, save_path):
        """
        Plot the results of branch ablation.
        
        Parameters
        ----------
        baseline_acc : float
            Baseline accuracy
        branch_results : dict
            Results of branch ablation
        save_path : str
            Path to save the plot
        """

This function creates a bar chart showing the impact of ablating each branch.

.. figure:: figures/pdf/place_holder.pdf
   :alt: Branch Ablation Results
   :width: 600px
   :align: center

   Example visualization of branch ablation results, showing the importance of each branch. [PLACEHOLDER FOR DIAGRAM]

Comprehensive Ablation Analysis
-------------------------------

For a comprehensive understanding of network behavior, you can combine different types of ablation studies:

.. code-block:: python

    def comprehensive_ablation_analysis(model, dataset):
        """
        Perform comprehensive ablation analysis.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Dataset to evaluate on
            
        Returns
        -------
        dict
            Dictionary with ablation results
        """
        results = {}
        
        # Baseline performance
        baseline_acc = evaluate_accuracy(model, dataset)
        results['baseline'] = baseline_acc
        
        # Layer ablation
        _, layer_results = analyze_layer_ablation(model, dataset)
        results['layer_ablation'] = layer_results
        
        # Branch ablation for each layer
        branch_results = {}
        for layer_idx in range(len(model.ei_layers)):
            _, layer_branch_results = analyze_branch_ablation(
                model, dataset, layer_idx, None, 'direct'
            )
            branch_results[layer_idx] = layer_branch_results
        results['branch_ablation'] = branch_results
        
        # Mechanism ablation
        mechanism_results = {}
        
        # Shunting inhibition
        model_copy = deepcopy(model)
        for layer in model_copy.ei_layers:
            layer.use_shunting = False
        mechanism_results['shunting'] = evaluate_accuracy(model_copy, dataset)
        
        # Reactivation
        model_copy = deepcopy(model)
        for layer in model_copy.ei_layers:
            layer.reactivate = False
        mechanism_results['reactivation'] = evaluate_accuracy(model_copy, dataset)
        
        results['mechanism_ablation'] = mechanism_results
        
        return results

This function performs a comprehensive ablation analysis, including layer ablation, branch ablation, and mechanism ablation.

Best Practices for Ablation Studies
-----------------------------------

1. **Start with trained models**: Conduct ablation studies on well-trained models that have reached good performance
2. **Ablate systematically**: Systematically ablate each component to ensure comprehensive understanding
3. **Compare across models**: Compare ablation results across different models to identify consistent patterns
4. **Control for variability**: Run multiple trials to account for variability in results
5. **Correlate with branch specialization**: Correlate ablation results with branch specialization metrics
6. **Use multiple datasets**: Test ablation on multiple datasets to ensure generalizable insights
7. **Combine with visualization**: Combine ablation studies with visualizations of weights and activations

Ablation in Research
--------------------

Ablation studies are a powerful tool for understanding dendritic networks in research:

1. **Hypothesis testing**: Test hypotheses about the computational role of different components
2. **Model comparison**: Compare the robustness of different model architectures
3. **Feature importance**: Identify which input features are most important for performance
4. **Mechanism validation**: Validate the importance of biologically-inspired mechanisms
5. **Network pruning**: Identify branches that can be pruned without significant performance loss
6. **Transfer learning**: Understand which components are specialized for specific tasks

By conducting systematic ablation studies, researchers can gain insights into the inner workings of dendritic networks and identify the key computational principles that drive their performance.

.. figure:: figures/pdf/place_holder.pdf
   :alt: Ablation Study Workflow
   :width: 600px
   :align: center

   Workflow for conducting comprehensive ablation studies with dendritic networks. [PLACEHOLDER FOR DIAGRAM] 