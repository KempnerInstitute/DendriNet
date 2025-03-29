.. _branch_information:

Branch Information Analysis
==========================

The Dendritic Modeling framework provides tools to analyze the information content in dendritic branches. These analyses help understand how branches learn to represent features and how they contribute to the overall function of the network.

Overview
--------

Dendritic branches in biological neurons are known to perform local computations, with different branches potentially specializing in detecting different input patterns. Similarly, in our artificial dendritic networks, branches may learn to represent different features or aspects of the input data.

Branch information analysis helps answer questions such as:

1. Which branches carry the most information about the task?
2. Are there branches that specialize in certain classes or features?
3. How is information distributed across branches and layers?
4. How does branch information content evolve during training?

Available Analysis Tools
-----------------------

The framework provides several tools for branch information analysis:

analyze_dendritic_information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary function for branch information analysis:

.. code-block:: python

    from dendritic_modeling.utils import analyze_dendritic_information
    
    branch_info = analyze_dendritic_information(
        model=trained_model,
        dataset=test_dataset,
        device='cuda'
    )

This function computes various information metrics for each branch in the model, including:

- **Mutual Information**: How much information each branch provides about the class labels
- **Class Selectivity**: How selectively branches respond to different classes
- **Feature Correlation**: How branch activations correlate with input features
- **Activity Statistics**: Statistical properties of branch activations

Configuration-Based Analysis
---------------------------

You can enable branch information analysis in your configuration file:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - branch_info  # Enable branch information analysis

This will automatically run the analysis during evaluation and generate visualizations.

Interpreting the Results
-----------------------

Branch Information Plots
~~~~~~~~~~~~~~~~~~~~~~~

The analysis generates several plots:

1. **Mutual Information Distribution**: Shows the distribution of mutual information across branches
2. **Class Selectivity Map**: Visualizes which branches are selective for which classes
3. **Branch Activity Distribution**: Shows the distribution of branch activations
4. **Information by Layer**: Compares information content across layers

High mutual information indicates branches that are important for the classification task. Branches with high class selectivity may be specialized for detecting features specific to certain classes.

Example Analysis
--------------

Here's an example of how to analyze and interpret branch information:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from dendritic_modeling.utils import analyze_dendritic_information
    
    # Analyze branch information
    branch_info = analyze_dendritic_information(
        model=trained_model,
        dataset=test_dataset,
        device='cuda'
    )
    
    # Extract mutual information
    mutual_info = branch_info['mutual_information']
    
    # Plot mutual information by branch
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(mutual_info)), mutual_info)
    plt.xlabel('Branch Index')
    plt.ylabel('Mutual Information (bits)')
    plt.title('Information Content in Dendritic Branches')
    
    # Find branches with highest information content
    top_branches = np.argsort(mutual_info)[-10:]
    print(f"Top 10 branches by information content: {top_branches}")
    
    # Analyze class selectivity
    class_selectivity = branch_info['class_selectivity']
    
    # Plot class selectivity for top branches
    plt.figure(figsize=(12, 8))
    plt.imshow(class_selectivity[top_branches], cmap='viridis', aspect='auto')
    plt.xlabel('Class')
    plt.ylabel('Branch Index')
    plt.title('Class Selectivity of Top Information-Carrying Branches')
    plt.colorbar(label='Selectivity')
    
    plt.tight_layout()
    plt.show()

Related Analysis Tools
--------------------

Branch information analysis is often used alongside:

- :doc:`ablation_studies`: To validate the importance of high-information branches
- :doc:`../visualization/visualization_guide`: To visualize branch activations for different inputs
- :doc:`ablation_information_analysis`: For a deeper analysis combining ablation and information theory

See Also
--------

- :doc:`../network_structures/network_structures`: For details on branch architecture
- :doc:`../learning/learning_strategies`: For learning strategies that might affect branch specialization 