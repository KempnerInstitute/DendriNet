.. _ablation_information_analysis:

Ablation Studies and Information Analysis
=========================================

This document provides detailed guidance on performing ablation studies and information-theoretic analyses of dendritic networks, with a specific focus on layer-based and branch-based synaptic weights.

Overview
--------

Understanding how dendritic networks process information requires specialized analysis techniques. Two particularly valuable approaches are:

1. **Ablation studies**: Systematically removing or modifying components to understand their contribution
2. **Information analysis**: Applying information theory to analyze how information flows through the network

These techniques help reveal the computational mechanisms of dendritic networks beyond standard performance metrics.

Ablation Studies
----------------

Ablation studies involve systematically removing or modifying components of the network to understand their contribution to overall performance.

Types of Ablation
^^^^^^^^^^^^^^^^^

The framework supports several types of ablation studies:

1. **Pathway ablation**: Removing entire excitatory or inhibitory pathways
2. **Layer ablation**: Removing specific layers
3. **Branch ablation**: Removing specific branches
4. **Feature ablation**: Removing specific input features
5. **Mechanism ablation**: Replacing mechanisms like top-k selection with alternatives

Basic Ablation Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure ablation studies in your evaluation:

.. code-block:: yaml

    evaluate:
      ablation:
        enabled: true
        save_results: true
        output_dir: "ablation_results/"
        
        # Components to ablate
        components:
          - type: "pathway"
            target: "inhibitory"  # Options: "excitatory", "inhibitory"
          
          - type: "layer"
            pathway: "excitatory"  # Which pathway the layer belongs to
            index: 0               # Layer index
          
          - type: "branch"
            pathway: "excitatory"
            layer: 0
            indices: [0, 1, 2]     # Branch indices
          
          - type: "mechanism"
            target: "topk"         # Target mechanism
            replacement: "average"  # Replacement mechanism
        
        # Metrics to compute
        metrics:
          - "accuracy"
          - "loss"
          - "per_class_accuracy"

This configuration specifies which components to ablate and which metrics to measure after ablation.

Pathway Ablation
^^^^^^^^^^^^^^^^

Pathway ablation examines the contribution of entire excitatory or inhibitory pathways:

.. code-block:: yaml

    evaluate:
      ablation:
        components:
          - type: "pathway"
            target: "inhibitory"     # Remove inhibitory pathway
            method: "zero_output"    # Method: zero_output, remove, random_weights

Analyzing pathway ablation results reveals:
1. The relative importance of excitatory vs. inhibitory processing
2. Whether the pathways are redundant or complementary
3. How the pathways interact in different tasks or data regimes

Layer Ablation
^^^^^^^^^^^^^^

Layer ablation examines the contribution of specific layers:

.. code-block:: yaml

    evaluate:
      ablation:
        components:
          - type: "layer"
            pathway: "excitatory"    # Which pathway
            index: 0                 # Layer index (0-based)
            method: "identity"       # Method: identity, zero_output, random_weights

Analyzing layer ablation results reveals:
1. The role of different layers in the processing hierarchy
2. Whether deeper layers extract more complex features
3. Layer-specific specialization for different aspects of the task

Branch Ablation
^^^^^^^^^^^^^^^

Branch ablation is particularly important for dendritic networks, as it examines the contribution of specific dendritic branches:

.. code-block:: yaml

    evaluate:
      ablation:
        components:
          - type: "branch"
            pathway: "excitatory"    # Which pathway
            layer: 0                 # Layer index
            selection: "top_active"  # How to select branches: top_active, random, specific
            count: 10                # How many branches to ablate
            method: "zero_weights"   # Method: zero_weights, random_weights, average

          # Or specify exact branches
          - type: "branch"
            pathway: "excitatory"
            layer: 0
            indices: [0, 1, 2, 5, 8]  # Specific branch indices
            method: "zero_weights"

Analyzing branch ablation results reveals:

1. Which branches are most important for overall performance
2. Whether branches specialize in specific features or patterns
3. How redundant or complementary different branches are
4. Class-specific branch specialization

Mechanism Ablation
^^^^^^^^^^^^^^^^^^

Mechanism ablation examines the contribution of specific computational mechanisms:

.. code-block:: yaml

    evaluate:
      ablation:
        components:
          - type: "mechanism"
            target: "topk"            # Target mechanism
            replacement: "average"    # Replacement: average, max, random

Common mechanisms to ablate include:
1. **topk**: The top-k selection mechanism
2. **gradient_scaling**: Gradient scaling methods
3. **branch_activation**: Activation functions in branches

Progressive Ablation
^^^^^^^^^^^^^^^^^^^^

Progressive ablation involves removing components incrementally to understand their cumulative impact:

.. code-block:: yaml

    evaluate:
      ablation:
        progressive:
          enabled: true
          component_type: "branch"    # Component type to progressively ablate
          pathway: "excitatory"       # Which pathway
          layer: 0                    # Which layer
          selection: "importance"     # How to select components: importance, random, activation
          step_size: 5                # How many components to remove in each step
          steps: 20                   # Total number of steps

Synaptic Weight Ablation
^^^^^^^^^^^^^^^^^^^^^^^^

For detailed analysis of synaptic weights, you can ablate specific weight connections:

.. code-block:: yaml

    evaluate:
      ablation:
        synaptic_weights:
          enabled: true
          selection:
            method: "magnitude"        # How to select weights: magnitude, random, gradient
            percentage: 10             # Percentage of weights to ablate
            absolute_threshold: 0.1    # Or use absolute threshold
          pathways: ["excitatory", "inhibitory"]  # Which pathways
          layers: [0, 1]               # Which layers

Ablation Analysis
^^^^^^^^^^^^^^^^^

After performing ablations, analyze the results to understand component contributions:

.. code-block:: yaml

    evaluate:
      ablation:
        analysis:
          enabled: true
          compute_importance: true     # Compute importance scores
          rank_components: true        # Rank components by importance
          class_specific: true         # Compute class-specific importance
          visualization:
            enabled: true
            type: "bar_chart"          # Visualization type

The framework may provide tools to visualize ablation results, showing the impact of different ablations on performance metrics.

Information Analysis
--------------------

Information analysis applies concepts from information theory to understand how dendritic networks process and transform information.

Information Measures
^^^^^^^^^^^^^^^^^^^^

Common information-theoretic measures include:

1. **Mutual information (MI)**: Measures shared information between variables
2. **Conditional entropy**: Measures uncertainty in one variable given another
3. **Information gain**: Measures information added by a component
4. **Channel capacity**: Measures maximum information transfer capability

Basic Information Analysis Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure information analysis in your evaluation:

.. code-block:: yaml

    evaluate:
      information_analysis:
        enabled: true
        measures: ["mutual_information", "entropy", "information_gain"]
        discretization:
          method: "binning"       # Method for discretizing continuous values
          num_bins: 30            # Number of bins for discretization
        num_samples: 1000         # Number of samples for estimation
        output_dir: "info_analysis/"

Layer-Based Information Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze information flow through network layers:

.. code-block:: yaml

    evaluate:
      information_analysis:
        layer_analysis:
          enabled: true
          input_output_mi: true          # MI between layer input and output
          layer_layer_mi: true           # MI between consecutive layers
          layer_target_mi: true          # MI between layer output and target
          pathway_pathway_mi: true       # MI between excitatory and inhibitory pathways

This analysis reveals:
1. How much information each layer preserves from its input
2. How much information different layers share
3. How information about the target variable flows through the network
4. How excitatory and inhibitory pathways interact informationally

Branch-Based Information Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze information processing within and between dendritic branches:

.. code-block:: yaml

    evaluate:
      information_analysis:
        branch_analysis:
          enabled: true
          branch_branch_mi: true         # MI between branches
          branch_target_mi: true         # MI between branch output and target
          synapse_branch_mi: true        # MI between synaptic inputs and branch output
          selectivity_analysis: true     # Analyze branch selectivity

This analysis reveals:
1. How specialized different branches are
2. Whether branches extract complementary or redundant information
3. How selective branches are to specific input patterns
4. How branch outputs combine to form layer outputs

Dendritic Information Analysis Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework provides a specialized visualization for performing ablation analysis by dendritic order (primary, secondary, etc.) called ``dendritic_ablation_all``. This can be enabled in your configuration file:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_ablation_all  # Activates the dendritic order ablation analysis

The implementation consists of the following steps:

1. **Baseline Performance Measurement**: The model's baseline performance is measured on the specified dataset.

2. **Order-wise Ablation**: The model is systematically ablated at each dendritic order:
   - Combined Ablation: Both excitatory and inhibitory branches at a specific dendritic order are ablated simultaneously.
   - Excitatory Ablation: All excitatory branches at a specific dendritic order are ablated.
   - Inhibitory Ablation: All inhibitory branches at a specific dendritic order are ablated.

3. **Result Analysis**: The performance drop for each ablation is calculated. 
   - Both absolute performance drops and percentage drops relative to baseline are computed.
   - If the results contain lists instead of single values (e.g., from multiple runs), they are appropriately processed.

4. **Visualization**: The results are visualized as bar plots showing the performance drop as a function of dendritic order, with separate bars for excitatory, inhibitory, and combined ablation.

The implementation handles potential edge cases:
- Results may contain lists of values for each order (e.g., when aggregating multiple runs)
- Some dendritic orders might not have both excitatory and inhibitory branches
- When only one pathway exists, the combined ablation result reflects that pathway's ablation
- Results are normalized relative to baseline performance for more interpretable comparisons

Example configuration:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_ablation_all
      parameters:
        dendritic_ablation_all:
          save_data: true  # Optionally save raw data for further analysis
          figsize: [10, 6]  # Customize figure size
          colors: ['blue', 'red', 'purple']  # Colors for excitatory, inhibitory, and combined lines

The analysis provides insights into:
- How information is distributed across dendritic orders
- The relative importance of proximal versus distal dendritic segments
- The complementary roles of excitatory and inhibitory pathways at different hierarchical levels
- Whether ablating both pathways has a superadditive or subadditive effect compared to ablating individual pathways

Information Analysis Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable the branch information analysis visualization:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_information_inputs    # Analyze information in branch inputs (pre-shunting)
        - dendritic_information_activations  # Analyze information in branch outputs (post-shunting)
      parameters:
        dendritic_information_inputs:
          mlp_hidden_dims: [100, 50]  # Hidden layer dimensions for the MLP classifier
          training_epochs: 20  # Number of epochs for MLP training
          batch_size: 64  # Batch size for MLP training
        dendritic_information_activations:
          mlp_hidden_dims: [100, 50]  # Hidden layer dimensions for the MLP classifier
          training_epochs: 20  # Number of epochs for MLP training
          batch_size: 64  # Batch size for MLP training

The implementation follows these steps:

1. **Activation Extraction**: The model is run on the dataset, and activations from branches at each dendritic order are extracted.
   - Branches are categorized by type (excitatory, inhibitory) and by dendritic order.
   - Activations are collected in appropriate tensors, handling potential device issues.

2. **Branch Categorization**: Branches are grouped by their dendritic depth and type.
   - The model's architecture is traversed to identify branches at each depth.
   - Each branch is classified as excitatory or inhibitory.

3. **MLP Classifier Training**: For each branch category (excitatory, inhibitory, combined at each depth):
   - A simple MLP classifier is trained to predict labels from branch activations.
   - Performance metrics (accuracy) are recorded.
   - Training follows a standard procedure with specified hyperparameters.

4. **Information Measurement**: The classification accuracy serves as a proxy for the amount of task-relevant information contained in each branch type/depth.
   - Higher accuracy indicates more discriminative information.
   - Results are also normalized relative to chance performance.

5. **Result Visualization**: Two types of visualizations are generated:
   - Raw accuracy plot: Shows the classification accuracy for each branch category.
   - Normalized information plot: Shows the percentage above random chance.

Analyzing Pre-Shunting Branch Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `dendritic_information_inputs` visualization analyzes the raw inputs to each branch before nonlinear processing:

- Excitatory inputs: :math:`w_i^{exc} \cdot x_i`
- Inhibitory inputs: :math:`w_i^{inh} \cdot x_i`
- Combined inputs: concatenation of excitatory and inhibitory inputs

This analysis reveals:
- Information content in the raw synaptic inputs
- How information is distributed across different branch depths
- The relative information content in excitatory vs. inhibitory input channels
- How combining excitatory and inhibitory inputs affects classification performance

To enable this analysis:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_information_inputs
      parameters:
        dendritic_information_inputs:
          mlp_hidden_dims: [100, 50]  # Hidden layer dimensions for the MLP classifier
          training_epochs: 20  # Number of epochs for MLP training
          batch_size: 64  # Batch size for MLP training

Analyzing Post-Shunting Branch Activations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `dendritic_information_activations` visualization analyzes the outputs after nonlinear processing:

- With shunting: :math:`\frac{w_i^{exc} \cdot x_i}{w_i^{exc} \cdot x_i + w_i^{inh} \cdot x_i + 1}`
- Without shunting: :math:`w_i^{exc} \cdot x_i - w_i^{inh} \cdot x_i`

This analysis reveals:
- Information content after nonlinear mixing operations
- How shunting inhibition transforms information
- How nonlinear operations enhance or suppress certain types of information

To enable this analysis:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_information_activations
      parameters:
        dendritic_information_activations:
          mlp_hidden_dims: [100, 50]  # Hidden layer dimensions for the MLP classifier
          training_epochs: 20  # Number of epochs for MLP training
          batch_size: 64  # Batch size for MLP training

Comparing Pre-Shunting vs. Post-Shunting Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using both visualization methods together provides a comprehensive view of information flow through dendritic branches:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_information_inputs     # Pre-shunting analysis
        - dendritic_information_activations  # Post-shunting analysis
      parameters:
        dendritic_information_inputs:
          mlp_hidden_dims: [100, 50]
          training_epochs: 20
          batch_size: 64
        dendritic_information_activations:
          mlp_hidden_dims: [100, 50]
          training_epochs: 20
          batch_size: 64

Comparing the results from both analyses reveals:
- How much information is preserved or lost during nonlinear processing
- Whether shunting enhances or diminishes specific types of information
- At which dendritic orders nonlinear processing has the greatest impact

The implementation handles several edge cases:
- Some dendritic orders might not have both excitatory and inhibitory branches
- NaN values are used for missing branch types
- Device management ensures tensors are on appropriate devices

Synaptic Weight Information Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Detailed analysis of how synaptic weights contribute to information processing:

.. code-block:: yaml

    evaluate:
      information_analysis:
        synaptic_analysis:
          enabled: true
          weight_information:
            enabled: true
            method: "perturbation"       # Method: perturbation, gradient, direct
          connection_importance:
            enabled: true
            metric: "information_gain"   # Metric for measuring importance

Receptive Field Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

Analyze the receptive fields of different branches:

.. code-block:: yaml

    evaluate:
      information_analysis:
        receptive_fields:
          enabled: true
          methods: ["reverse_correlation", "maximally_activating"]
          num_samples: 1000
          visualization:
            enabled: true
            format: "heatmap"

This analysis reveals:
1. What input patterns each branch responds to
2. How receptive fields differ between branches
3. Whether branches have specialized for specific features

Information Bottleneck Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply information bottleneck theory to understand the efficiency of information processing:

.. code-block:: yaml

    evaluate:
      information_analysis:
        bottleneck_analysis:
          enabled: true
          sweep_beta: true               # Sweep over beta values
          beta_values: [0.001, 0.01, 0.1, 1.0, 10.0]
          compute_ib_curve: true         # Compute IB curve

This analysis reveals:
1. How efficiently the network extracts task-relevant information
2. Whether the network compresses input information optimally
3. The trade-off between compression and task performance

Combining Ablation and Information Analysis
-------------------------------------------

For the most comprehensive understanding, combine ablation and information analysis:

.. code-block:: yaml

    evaluate:
      # First perform ablation
      ablation:
        enabled: true
        components:
          - type: "branch"
            # ... ablation details ...
      
      # Then analyze information changes
      information_analysis:
        enabled: true
        differential:
          enabled: true
          baseline: "no_ablation"        # Compare to non-ablated model
          compute_mi_difference: true    # Compute MI differences

This combined approach reveals:
1. How ablating components affects information flow
2. Whether removing components causes information loss or rerouting
3. The information-theoretic contribution of different components

Visualizing Information Analysis Results
----------------------------------------

The framework provides various visualizations for information analysis:

.. code-block:: yaml

    evaluate:
      information_analysis:
        visualization:
          enabled: true
          plots:
            - type: "mi_matrix"          # Mutual information matrix
            - type: "information_flow"   # Information flow diagram
            - type: "receptive_fields"   # Receptive field visualization
            - type: "bottleneck_curve"   # Information bottleneck curve
          interactive: true              # Whether to create interactive visualizations
          output_dir: "visualizations/info_analysis/"

Best Practices for Ablation and Information Analysis
----------------------------------------------------

1. **Start with coarse-grained ablations**: Begin with pathway or layer ablations before moving to more fine-grained branch ablations

2. **Compare to baseline**: Always compare ablation results to the full model baseline

3. **Combine multiple measures**: Use both performance metrics and information-theoretic measures

4. **Consider sample size**: Ensure sufficient samples for reliable information estimation

5. **Visualize results**: Create visualizations to help interpret complex information-theoretic results

6. **Perform statistical tests**: Assess the statistical significance of observed effects

7. **Cross-validate**: Ensure results are consistent across different data splits

Complete Analysis Example
-------------------------

Here's a complete example configuration for comprehensive ablation and information analysis:

.. code-block:: yaml

    evaluate:
      # Ablation studies
      ablation:
        enabled: true
        save_results: true
        output_dir: "ablation_results/"
        
        # Components to ablate
        components:
          # Pathway ablation
          - type: "pathway"
            target: "inhibitory"
            method: "zero_output"
          
          # Layer ablation
          - type: "layer"
            pathway: "excitatory"
            index: 0
            method: "identity"
          
          # Branch ablation
          - type: "branch"
            pathway: "excitatory"
            layer: 0
            selection: "top_active"
            count: 10
            method: "zero_weights"
          
          # Mechanism ablation
          - type: "mechanism"
            target: "topk"
            replacement: "average"
        
        # Progressive branch ablation
        progressive:
          enabled: true
          component_type: "branch"
          pathway: "excitatory"
          layer: 0
          selection: "importance"
          step_size: 5
          steps: 20
        
        # Metrics to compute
        metrics:
          - "accuracy"
          - "loss"
          - "per_class_accuracy"
        
        # Analysis of ablation results
        analysis:
          enabled: true
          compute_importance: true
          rank_components: true
          class_specific: true
          visualization:
            enabled: true
            type: "bar_chart"
      
      # Information analysis
      information_analysis:
        enabled: true
        measures: ["mutual_information", "entropy", "information_gain"]
        discretization:
          method: "binning"
          num_bins: 30
        num_samples: 1000
        output_dir: "info_analysis/"
        
        # Layer-based analysis
        layer_analysis:
          enabled: true
          input_output_mi: true
          layer_layer_mi: true
          layer_target_mi: true
          pathway_pathway_mi: true
        
        # Branch-based analysis
        branch_analysis:
          enabled: true
          branch_branch_mi: true
          branch_target_mi: true
          synapse_branch_mi: true
          selectivity_analysis: true
        
        # Synaptic weight analysis
        synaptic_analysis:
          enabled: true
          weight_information:
            enabled: true
            method: "perturbation"
          connection_importance:
            enabled: true
            metric: "information_gain"
        
        # Receptive field analysis
        receptive_fields:
          enabled: true
          methods: ["reverse_correlation", "maximally_activating"]
          num_samples: 1000
          visualization:
            enabled: true
            format: "heatmap"
        
        # Bottleneck analysis
        bottleneck_analysis:
          enabled: true
          sweep_beta: true
          beta_values: [0.001, 0.01, 0.1, 1.0, 10.0]
          compute_ib_curve: true
        
        # Visualizations
        visualization:
          enabled: true
          plots:
            - type: "mi_matrix"
            - type: "information_flow"
            - type: "receptive_fields"
            - type: "bottleneck_curve"
          interactive: true
          output_dir: "visualizations/info_analysis/"
        
      # Combined analysis
      combined_analysis:
        enabled: true
        ablation_info:
          enabled: true
          baseline: "no_ablation"
          compute_mi_difference: true 