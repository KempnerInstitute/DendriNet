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
            'indirect' (zero out all except specified branches)
        """

This function zeros out the weights of specified branches, effectively removing them from the network.

A comprehensive branch ablation analysis function is also available:

.. code-block:: python

    def analyze_branch_ablation(model, dataset, layer_idx=None, branch_ids=None,
                                mode='direct', device='cpu'):
        """
        Perform ablation analysis for specific branches.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Dataset to evaluate on
        layer_idx : int or None
            Index of the layer to ablate
        branch_ids : list or None
            IDs of branches to ablate, or None for all branches
        mode : str
            Ablation mode: 'direct' or 'indirect'
        device : str
            Device to use for computation
            
        Returns
        -------
        tuple
            (baseline_accuracy, results_dict)
        """

This function ablates specific branches and measures the change in performance, helping identify which branches are most important for the task.

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
    importance = {branch_id: delta for branch_id, (acc, delta) in branch_results.items()}
    most_important = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Most important branches:")
    for branch_id, imp in most_important:
        print(f"Branch {branch_id}: Importance = {imp:.4f}")

Layer-wise Ablation Analysis
~~~~~~~~~~~~~~~~~~~~~~~

The framework provides a layerwise ablation analysis function:

.. code-block:: python

    def analyze_layerwise_ablation(model, dataset, train_data=None, valid_data=None,
                                 ablation_mode='direct', device='cpu'):
        """
        Analyze the impact of ablating each layer on the model's performance.
        
        Parameters
        ----------
        model : nn.Module
            A trained model to analyze.
        dataset : torch.utils.data.Dataset
            Test dataset for evaluation.
        train_data : torch.utils.data.Dataset, optional
            Training dataset, used only if retraining is needed.
        valid_data : torch.utils.data.Dataset, optional
            Validation dataset, used only if retraining is needed.
        ablation_mode : str
            The ablation mode to use. Options: 'direct', 'indirect', 'retrain'.
            'direct': Simply zero out the layer's outputs.
            'indirect': Replace the layer with a pass-through.
            'retrain': Retrain the network with the layer fixed at initialization.
        device : str
            Device to use for computation.
            
        Returns
        -------
        dict
            Dictionary with ablation results, including baseline accuracy and 
            accuracy drop for each layer.
        """
        ...

To use this function in your analysis:

.. code-block:: python

    from dendritic_modeling.utils import analyze_layerwise_ablation
    
    # Run layerwise ablation analysis
    results = analyze_layerwise_ablation(
        model=trained_model, 
        dataset=test_dataset,
        ablation_mode='direct',
        device='cuda'
    )
    
    # Print results
    print(f"Baseline accuracy: {results['baseline_acc']:.4f}")
    for i, acc_drop in enumerate(results['accuracy_drops']):
        print(f"Layer {i} ablation: accuracy drop = {acc_drop:.4f}")

This can help identify which layers are most critical to model performance, providing insights into the hierarchical processing structure of the network.

Example for visualizing layerwise ablation results:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results['accuracy_drops'])), results['accuracy_drops'])
    plt.xlabel('Layer Index')
    plt.ylabel('Accuracy Drop')
    plt.title('Impact of Layer Ablation on Performance')
    plt.xticks(range(len(results['accuracy_drops'])))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

Visualizing Ablation Results
----------------------------

The dendritic modeling framework provides visualization tools for ablation studies.

Branch Importance Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After performing branch ablation, you can visualize the importance of each branch:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Assuming branch_results is the result of analyze_branch_ablation
    
    # Extract branch IDs and importance values
    branch_ids = np.array(list(branch_results.keys()))
    importance = np.array([delta for _, delta in branch_results.values()])
    
    # Sort by importance
    sort_idx = np.argsort(importance)[::-1]
    branch_ids = branch_ids[sort_idx]
    importance = importance[sort_idx]
    
    # Plot branch importance
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(branch_ids)), importance)
    plt.xticks(np.arange(len(branch_ids)), branch_ids)
    plt.xlabel('Branch ID')
    plt.ylabel('Importance (Accuracy Drop)')
    plt.title('Branch Importance Analysis')
    plt.tight_layout()
    plt.savefig('branch_importance.png')
    plt.close()

This code creates a bar chart showing the importance of each branch based on the accuracy drop when the branch is ablated.

Branch Activation Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also visualize the activation patterns of different branches to understand their function:

.. code-block:: python

    from dendritic_modeling.utils import record_branch_activations
    
    # Record branch activations for a subset of the dataset
    activations = record_branch_activations(
        model, 
        dataset,
        batch_size=32,
        device='cuda'
    )
    
    # Plot activation patterns for the first layer
    plt.figure(figsize=(15, 10))
    layer_activations = activations[0]  # First layer
    
    for i in range(min(10, layer_activations.shape[0])):
        plt.subplot(2, 5, i+1)
        plt.imshow(layer_activations[i].mean(dim=0).reshape(28, 28), cmap='viridis')
        plt.title(f'Branch {i}')
        plt.axis('off')
    
    plt.suptitle('Branch Activation Patterns (Layer 0)')
    plt.tight_layout()
    plt.savefig('branch_activation_patterns.png')
    plt.close()

This code visualizes the mean activation pattern for each branch in the first layer, helping understand what features each branch responds to.

Combined Analysis
----------------

For a comprehensive understanding of branch function, you can combine branch ablation with other analyses:

.. code-block:: python

    import torch
    from dendritic_modeling.utils import analyze_branch_ablation, record_branch_activations
    
    # Load model and dataset
    model = torch.load("path/to/model.pt")
    
    # Get branch activations
    activations = record_branch_activations(model, dataset, device='cuda')
    
    # Perform branch ablation
    baseline_acc, branch_results = analyze_branch_ablation(
        model, dataset, layer_idx=0, device='cuda'
    )
    
    # Combine results
    results = {}
    for branch_id, (acc, delta) in branch_results.items():
        # Get mean activation for this branch
        mean_activation = activations[0][branch_id].mean().item()
        
        # Store combined results
        results[branch_id] = {
            'importance': delta,
            'mean_activation': mean_activation
        }
    
    # Analyze correlation between activation and importance
    importance_values = [r['importance'] for r in results.values()]
    activation_values = [r['mean_activation'] for r in results.values()]
    
    correlation = np.corrcoef(importance_values, activation_values)[0, 1]
    print(f"Correlation between importance and activation: {correlation:.4f}")
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(activation_values, importance_values)
    plt.xlabel('Mean Activation')
    plt.ylabel('Importance (Accuracy Drop)')
    plt.title(f'Activation vs. Importance (r = {correlation:.4f})')
    plt.grid(True)
    plt.savefig('activation_vs_importance.png')
    plt.close()

This code analyzes the relationship between branch activation and importance, helping understand whether the most active branches are also the most important.

Best Practices for Ablation Studies
-----------------------------------

Here are some best practices for conducting ablation studies with dendritic networks:

1. **Use a separate test set**: Always evaluate ablation on a separate test set, not the training set, to avoid biased results.

2. **Multiple random seeds**: Run ablation studies with multiple random seeds and average the results to ensure robustness.

3. **Control for branch size**: Consider the size or capacity of branches when interpreting importance, as larger branches might naturally show higher importance.

4. **Consider different ablation modes**: Use both 'direct' and 'indirect' ablation modes to get a more complete picture of branch function.

5. **Combine with other analyses**: Integrate ablation results with other analyses like activation visualization or information analysis.

6. **Analyze by dendritic order**: Group branches by their dendritic order (distance from soma) to understand the hierarchical structure of computation.

7. **Compare across models**: Compare ablation results across different models or training runs to identify consistent patterns.

Conclusion
----------

Ablation studies are a powerful tool for understanding the function of dendritic networks. By systematically removing branches or layers and measuring the impact on performance, you can identify which components are critical for different aspects of computation.

The dendritic modeling framework provides specialized tools for conducting these studies, enabling researchers to gain insights into the computational principles of dendritic networks. 