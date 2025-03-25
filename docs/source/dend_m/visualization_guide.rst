.. _visualization_guide:

Network Visualization Guide
===========================
Overview
--------

This guide provides detailed information on visualizing dendritic networks, their architectures, and training dynamics.
The visualization tools in the dendritic modeling framework allow researchers to gain insights into network behavior,
analyze computational mechanisms, and create publication-quality figures.

Network Architecture Visualizations
-----------------------------------

Excitation-Inhibition Network (EINet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EINet architecture combines excitatory and inhibitory pathways in a biologically-inspired neural network design.
Key components to visualize include:

* **Excitatory pathways**: Forward-propagating activations
* **Inhibitory pathways**: Regulatory connections that modulate excitatory signals
* **Dendritic branches**: Hierarchical processing units within each layer
* **Sparse connectivity**: Visualizing the TopK connection patterns

.. code-block:: python

    import matplotlib.pyplot as plt
    from dendritic_modeling.plot_utils import plot_einet_params
    
    # Visualize the network parameters
    plot_einet_params(
        einet=model.net,
        save_root="visualization_outputs/architecture",
        reshape_fn=lambda x: x.view(28, 28),  # For MNIST reshape
        reshape_exc_syn=True,
        logspace=True
    )

Input Transformation Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input transformation networks preprocess data before it enters the dendritic network.
Visualizing these components helps understand how information is transformed and separated
into excitatory and inhibitory pathways.

.. code-block:: python

    # Visualize the input network architecture with excitatory and inhibitory paths
    def visualize_input_network(model, input_example):
        """Create visualization of input transformation network structure"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original input
        axes[0].imshow(input_example.reshape(28, 28), cmap='gray')
        axes[0].set_title("Original Input")
        
        # Get transformed inputs
        with torch.no_grad():
            exc_input, inh_input = model.net.input_transform(input_example.unsqueeze(0))
        
        # Excitatory path output
        exc_img = exc_input[0].cpu().reshape(-1, 1).numpy()
        axes[1].imshow(exc_img, cmap='hot', aspect='auto')
        axes[1].set_title("Excitatory Path Output")
        
        # Inhibitory path output
        inh_img = inh_input[0].cpu().reshape(-1, 1).numpy()
        axes[2].imshow(inh_img, cmap='cool', aspect='auto')
        axes[2].set_title("Inhibitory Path Output")
        
        plt.tight_layout()
        return fig

Input Modes Comparison
~~~~~~~~~~~~~~~~~~~~~~

The framework supports different input transformation modes that can be visualized for comparison:

* **Mode 0**: Original input directly fed to the network
* **Mode 1**: Same input used for both excitatory and inhibitory pathways
* **Mode 2**: Separate transformation network for excitatory and inhibitory inputs

.. code-block:: python

    # Example code to visualize different input modes
    def compare_input_modes(input_sample):
        """Compare the different input transformation modes"""
        models = [
            create_model(input_mode=0, config=config),
            create_model(input_mode=1, config=config),
            create_model(input_mode=2, config=config)
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        for i, model in enumerate(models):
            # Original input visualization
            axes[i, 0].imshow(input_sample.reshape(28, 28), cmap='gray')
            axes[i, 0].set_title(f"Mode {i}: Original")
            
            # Get transformed inputs
            with torch.no_grad():
                exc_input, inh_input = get_network_inputs(model, input_sample)
            
            # Visualize excitatory input
            exc_vis = visualize_tensor(exc_input)
            axes[i, 1].imshow(exc_vis, cmap='hot')
            axes[i, 1].set_title(f"Mode {i}: Excitatory")
            
            # Visualize inhibitory input
            inh_vis = visualize_tensor(inh_input)
            axes[i, 2].imshow(inh_vis, cmap='cool')
            axes[i, 2].set_title(f"Mode {i}: Inhibitory")
        
        plt.tight_layout()
        return fig

Training Strategy Visualizations
--------------------------------

Sequential vs. Joint Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using input transformation networks, you can train the input network and dendritic network sequentially or jointly.
Visualizing the training process helps understand the tradeoffs.

.. code-block:: python

    # Visualization code for sequential vs joint training performance
    def plot_training_comparison(sequential_history, joint_history):
        """Plot comparison of sequential vs joint training strategies"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        axes[0].plot(sequential_history['val_acc'], label='Sequential')
        axes[0].plot(joint_history['val_acc'], label='Joint')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Validation Accuracy')
        axes[0].legend()
        axes[0].set_title('Accuracy Comparison')
        
        # Loss comparison
        axes[1].plot(sequential_history['val_loss'], label='Sequential')
        axes[1].plot(joint_history['val_loss'], label='Joint')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Validation Loss')
        axes[1].legend()
        axes[1].set_title('Loss Comparison')
        
        plt.tight_layout()
        return fig

Independent Head Training
~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize the effects of training excitatory and inhibitory heads independently versus jointly.

.. code-block:: python

    # Visualization for independent head training
    def visualize_head_training(exc_features, inh_features, labels):
        """Visualize feature space learned by independently trained heads"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Use t-SNE to reduce dimensionality for visualization
        from sklearn.manifold import TSNE
        
        # Excitatory features
        tsne_exc = TSNE(n_components=2).fit_transform(exc_features)
        scatter_exc = axes[0].scatter(tsne_exc[:, 0], tsne_exc[:, 1], c=labels, cmap='tab10')
        axes[0].set_title('Excitatory Head Feature Space')
        
        # Inhibitory features
        tsne_inh = TSNE(n_components=2).fit_transform(inh_features)
        scatter_inh = axes[1].scatter(tsne_inh[:, 0], tsne_inh[:, 1], c=labels, cmap='tab10')
        axes[1].set_title('Inhibitory Head Feature Space')
        
        # Add color bar
        fig.colorbar(scatter_exc, ax=axes[0], label='Class')
        fig.colorbar(scatter_inh, ax=axes[1], label='Class')
        
        plt.tight_layout()
        return fig

Learning Strategies
~~~~~~~~~~~~~~~~~~~

Visualize and compare different learning strategies available in the framework:

* Maximum Likelihood Estimation (MLE)
* Freeze Layers
* Local Credit Assignment
* Two-Step Training
* Two-Step Training with KL

.. code-block:: python

    # Visualization code for comparing learning strategies
    def compare_learning_strategies(results_dict):
        """Compare performance of different learning strategies"""
        strategies = list(results_dict.keys())
        accuracies = [results_dict[s]['test_accuracy'] for s in strategies]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(strategies, accuracies)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Performance Comparison of Learning Strategies')
        
        plt.tight_layout()
        return fig

Computational Mechanism Visualizations
--------------------------------------

Dendritic Computation
~~~~~~~~~~~~~~~~~~~~~

Visualize the computational mechanisms within dendritic branches to understand how
they process information differently from traditional neural networks.

.. code-block:: python

    # Code to visualize dendritic computation mechanisms
    def visualize_dendritic_computation(model, input_sample):
        """Visualize how dendritic branches compute their outputs"""
        with torch.no_grad():
            # Forward pass with branch outputs
            output, branch_outputs = model.forward_with_branch_outputs(input_sample.unsqueeze(0))
            
            # Extract key components
            excitatory_inputs = model.net.layers[0].excitatory_input
            inhibitory_inputs = model.net.layers[0].inhibitory_input
            branch_integrations = model.net.layers[0].branch_voltage
            
            # Visualize the computation flow
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Excitatory inputs
            im1 = axes[0, 0].imshow(excitatory_inputs[0].cpu().numpy(), cmap='hot')
            axes[0, 0].set_title('Excitatory Inputs')
            fig.colorbar(im1, ax=axes[0, 0])
            
            # Inhibitory inputs
            im2 = axes[0, 1].imshow(inhibitory_inputs[0].cpu().numpy(), cmap='cool')
            axes[0, 1].set_title('Inhibitory Inputs')
            fig.colorbar(im2, ax=axes[0, 1])
            
            # Branch voltage (pre-activation)
            im3 = axes[1, 0].imshow(branch_integrations[0].cpu().numpy(), cmap='viridis')
            axes[1, 0].set_title('Branch Voltage (E/(E+I+1))')
            fig.colorbar(im3, ax=axes[1, 0])
            
            # Branch outputs (post-activation)
            im4 = axes[1, 1].imshow(branch_outputs[0][0].cpu().numpy(), cmap='plasma')
            axes[1, 1].set_title('Branch Outputs (After Activation)')
            fig.colorbar(im4, ax=axes[1, 1])
            
            plt.tight_layout()
            return fig

Network Pruning and Ablation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize the impact of pruning or ablating specific dendritic branches to understand
their contribution to the network's overall performance.

.. code-block:: python

    # Code to visualize ablation results
    def visualize_ablation_impact(model, dataset, layer_idx=0):
        """Visualize impact of ablating different branches"""
        from dendritic_modeling.utils import analyze_branch_ablation
        
        # Run ablation study
        baseline_acc, results = analyze_branch_ablation(
            model, dataset, layer_idx=layer_idx, 
            branch_ids=list(range(model.net.layers[layer_idx].n_branches)),
            mode='direct'
        )
        
        # Extract results
        branch_ids = list(results.keys())
        acc_drops = [baseline_acc - results[id][0] for id in branch_ids]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(branch_ids, acc_drops)
        
        # Highlight most important branches
        threshold = 0.05  # 5% accuracy drop
        for i, bar in enumerate(bars):
            if acc_drops[i] > threshold:
                bar.set_color('red')
        
        ax.axhline(y=0.01, color='grey', linestyle='--', alpha=0.7)
        ax.set_xlabel('Branch ID')
        ax.set_ylabel('Accuracy Drop When Ablated')
        ax.set_title(f'Branch Importance in Layer {layer_idx}')
        
        plt.tight_layout()
        return fig

Dendritic Order Ablation
~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides a built-in visualization for analyzing the impact of ablating entire dendritic orders (primary, secondary, etc.). This can be enabled in your configuration:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_ablation_all  # Ablation analysis by dendritic order

This visualization systematically ablates the excitatory and inhibitory branches at each dendritic order and measures the resulting performance drop:

.. code-block:: python

    from dendritic_modeling.utils import analyze_branch_ablation, measure_performance
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    def plot_dendritic_ablation_all(model, dataset, device='cuda'):
        """Plot the impact of ablating different dendritic orders"""
        
        # Perform dendritic order ablation
        results = analyze_branch_ablation(
            model, dataset, layer_idx=None, branch_ids=None, device=device
        )
        
        # Calculate baseline accuracy
        baseline_acc = measure_performance(model, dataset, device=device)
        
        # Calculate percentage drops
        exc_pct = [(drop / baseline_acc) * 100 for drop in results["exc_drop"]]
        inh_pct = [(drop / baseline_acc) * 100 for drop in results["inh_drop"]]
        
        # Create DataFame
        df = pd.DataFrame({
            'Order': results['order'],
            'Excitatory Drop (%)': exc_pct,
            'Inhibitory Drop (%)': inh_pct
        })
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(df['Order'], df['Excitatory Drop (%)'], 'b-', label='Excitatory')
        plt.plot(df['Order'], df['Inhibitory Drop (%)'], 'r-', label='Inhibitory')
        
        plt.xlabel('Dendritic Order')
        plt.ylabel('Performance Drop (%)')
        plt.title('Impact of Ablating Dendritic Orders')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf(), df

The resulting visualization reveals:
1. The relative importance of each dendritic order
2. Which orders are most critical for performance
3. The differential contributions of excitatory vs. inhibitory pathways

Dendritic Information Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another powerful visualization is the dendritic information analysis, which quantifies how much task-relevant information is contained in each dendritic order:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - dendritic_information  # Information content analysis

This visualization:
1. Extracts branch activations at each dendritic order
2. Trains simple MLP classifiers to predict labels from these activations
3. Uses classification accuracy as a proxy for information content
4. Creates visualizations showing both raw accuracy and normalized information content

.. code-block:: python

    from dendritic_modeling.utils import analyze_dendritic_information
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_dendritic_information(model, dataset, device='cuda'):
        """Visualize information content in dendritic orders"""
        
        # Analyze information content
        results = analyze_dendritic_information(
            model=model,
            dataset=dataset,
            device=device,
            mlp_hidden_dims=[100, 50],
            training_epochs=20,
            batch_size=64
        )
        
        # Extract results
        depth_names = results["branch_depths"]
        exc_acc = np.array([acc if acc is not None else np.nan for acc in results["exc_accuracy"]])
        inh_acc = np.array([acc if acc is not None else np.nan for acc in results["inh_accuracy"]])
        combined_acc = np.array([acc if acc is not None else np.nan for acc in results["combined_accuracy"]])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(depth_names))
        width = 0.25
        
        # Plot each type of accuracy
        ax.bar(x - width, exc_acc, width, label='Excitatory', color='green', alpha=0.7)
        ax.bar(x, inh_acc, width, label='Inhibitory', color='red', alpha=0.7)
        ax.bar(x + width, combined_acc, width, label='Combined', color='blue', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Branch Depth')
        ax.set_ylabel('Classification Accuracy')
        ax.set_title('Information Content in Dendritic Branches')
        ax.set_xticks(x)
        ax.set_xticklabels(depth_names)
        ax.legend()
        
        # Add random chance line
        num_classes = 10  # Default for MNIST
        random_chance = 1.0 / num_classes
        ax.axhline(y=random_chance, color='gray', linestyle='--', 
                   label=f'Random Chance ({random_chance:.3f})')
        
        return fig, results

The visualization manager also creates a normalized version showing the percentage above random chance:

.. code-block:: python

    # Create normalized visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate percentage above random chance
    norm_factor = (1 - random_chance)
    exc_pct = (exc_acc - random_chance) * 100 / norm_factor
    inh_pct = (inh_acc - random_chance) * 100 / norm_factor
    combined_pct = (combined_acc - random_chance) * 100 / norm_factor
    
    # Plot normalized data
    ax.bar(x - width, exc_pct, width, label='Excitatory', color='green', alpha=0.7)
    ax.bar(x, inh_pct, width, label='Inhibitory', color='red', alpha=0.7)
    ax.bar(x + width, combined_pct, width, label='Combined', color='blue', alpha=0.7)
    
    ax.set_xlabel('Branch Depth')
    ax.set_ylabel('Information Content (% above random)')
    ax.set_title('Normalized Information Content in Dendritic Branches')
    ax.set_xticks(x)
    ax.set_xticklabels(depth_names)
    ax.legend()
    
    # Draw line for random chance (0%)
    ax.axhline(y=0, color='gray', linestyle='--', label='Random Chance')

These analyses together provide deep insights into the computational roles of different dendritic orders and how they process information.

Advanced Visualization Topics
-----------------------------

Animated Visualizations
~~~~~~~~~~~~~~~~~~~~~~~

Creating animated visualizations can help understand the training dynamics over time.

.. code-block:: python

    # Code for animated visualizations
    def create_weight_evolution_animation(weight_snapshots, save_path):
        """Create an animation of weight evolution during training"""
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ims = []
        
        for weights in weight_snapshots:
            im = ax.imshow(weights, animated=True, cmap='viridis')
            title = ax.text(0.5, 1.05, f"Epoch {i}", transform=ax.transAxes, 
                           ha="center", fontsize=14)
            ims.append([im, title])
        
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
        ani.save(save_path, writer='pillow')
        return ani

3D Visualizations
~~~~~~~~~~~~~~~~~

Three-dimensional visualizations can reveal patterns that might not be apparent in 2D.

.. code-block:: python

    # Code for 3D visualizations
    def visualize_3d_feature_space(features, labels):
        """Create a 3D visualization of feature space"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Use PCA to reduce dimension to 3
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each class with different color
        for i in range(10):  # Assuming 10 classes
            mask = (labels == i)
            ax.scatter(
                features_3d[mask, 0], 
                features_3d[mask, 1], 
                features_3d[mask, 2],
                label=f'Class {i}'
            )
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title('3D Feature Space Visualization')
        ax.legend()
        
        plt.tight_layout()
        return fig

Interactive Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive visualizations allow exploring the data more dynamically.

.. code-block:: python

    # Code for interactive visualizations (for Jupyter notebooks)
    def create_interactive_weight_explorer(model):
        """Create interactive widget to explore model weights"""
        import ipywidgets as widgets
        from IPython.display import display
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create layer selector
        layer_selector = widgets.Dropdown(
            options=list(range(len(model.net.layers))),
            description='Layer:',
        )
        
        # Create branch selector
        branch_selector = widgets.IntSlider(
            min=0,
            max=model.net.layers[0].n_branches - 1,
            description='Branch:',
        )
        
        # Update function
        def update_plot(layer, branch):
            ax.clear()
            weights = model.net.layers[layer].branch_excitation.pruned_weight()[branch].detach().cpu()
            im = ax.imshow(weights.reshape(28, 28), cmap='viridis')
            ax.set_title(f'Layer {layer}, Branch {branch} Weights')
            fig.canvas.draw_idle()
        
        # Connect the widgets to the update function
        widgets.interactive(update_plot, layer=layer_selector, branch=branch_selector)
        
        return fig, layer_selector, branch_selector

Best Practices for Visualizations
---------------------------------

When creating visualizations for your dendritic network experiments, consider these best practices:

1. **Consistent color schemes**: Use consistent colormaps across related visualizations
   - Excitatory pathways: Use warm colors ('hot', 'plasma', 'YlOrRd')
   - Inhibitory pathways: Use cool colors ('cool', 'viridis', 'PuBu')

2. **Include proper labeling**: Always include clear axes labels, titles, and legends

3. **Comparison visualizations**: When comparing models or approaches, use the same scale and metrics

4. **Save high-resolution figures**: For publications, save figures at 300+ DPI
   ```python
   plt.savefig('figure.png', dpi=300, bbox_inches='tight')
   ```

5. **Create explanatory diagrams**: Combine visualizations with explanatory text and arrows

Visualization Resources
-----------------------

Here are some useful resources for creating effective visualizations:

- **Matplotlib documentation**: https://matplotlib.org/stable/index.html
- **Seaborn for statistical visualizations**: https://seaborn.pydata.org/
- **ColorMaps in Matplotlib**: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- **NetworkX for graph visualizations**: https://networkx.org/
- **Plotly for interactive visualizations**: https://plotly.com/python/


.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle, FancyArrowPatch

    def create_input_network_diagram(save_path='input_network_architecture.png'):
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 12})
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Colors
        input_color = '#D3D3D3'  # light gray
        hidden_color = '#AED6F1'  # light blue
        excit_color = '#F5B7B1'   # light red
        inhib_color = '#ABEBC6'   # light green
        dendritic_color = '#FAD7A0'  # light orange
        
        # Layer dimensions
        layer_width = 1.0
        layer_height = 0.8
        layer_spacing = 2.0
        
        # Draw input layer
        ax.add_patch(Rectangle((0, 1), layer_width, layer_height, 
                            fc=input_color, ec='black', lw=2))
        ax.text(0.5, 1.4, 'Input', ha='center', va='center', fontweight='bold')
        
        # Draw hidden layers
        for i, (size, name) in enumerate(zip([256, 128], ['Hidden Layer 1', 'Hidden Layer 2'])):
            x_pos = (i+1) * layer_spacing
            height = layer_height * (size/256)
            y_pos = 1 + (layer_height - height)/2
            
            ax.add_patch(Rectangle((x_pos, y_pos), layer_width, height, 
                                fc=hidden_color, ec='black', lw=2))
            ax.text(x_pos + 0.5, 0.7, name, ha='center', va='center')
            
            # Connect to previous layer
            ax.add_patch(FancyArrowPatch((i*layer_spacing + layer_width, 1.4), 
                                    (x_pos, 1.4),
                                    arrowstyle='->', lw=2,
                                    connectionstyle='arc3,rad=0.0'))
        
        # Draw output heads
        head_x = 3 * layer_spacing
        
        # Excitatory head
        ax.add_patch(Rectangle((head_x, 2.2), layer_width, layer_height, 
                            fc=excit_color, ec='black', lw=2))
        ax.text(head_x + 0.5, 2.6, 'Excitatory Head', ha='center', va='center', fontweight='bold')
        
        # Inhibitory head
        ax.add_patch(Rectangle((head_x, 0.0), layer_width, layer_height, 
                            fc=inhib_color, ec='black', lw=2))
        ax.text(head_x + 0.5, 0.4, 'Inhibitory Head', ha='center', va='center', fontweight='bold')
        
        # Connect feature extraction to heads
        ax.add_patch(FancyArrowPatch((2*layer_spacing + layer_width, 1.4), 
                                (head_x, 2.6),
                                arrowstyle='->', lw=2, color='#E74C3C',
                                connectionstyle='arc3,rad=0.2'))
        
        ax.add_patch(FancyArrowPatch((2*layer_spacing + layer_width, 1.4), 
                                (head_x, 0.4),
                                arrowstyle='->', lw=2, color='#229954',
                                connectionstyle='arc3,rad=-0.2'))
        
        # Dendritic network
        ax.add_patch(Rectangle((head_x + 2*layer_spacing, 1), layer_width*1.5, layer_height, 
                            fc=dendritic_color, ec='black', lw=2))
        ax.text(head_x + 2*layer_spacing + 0.75, 1.4, 'Dendritic Network', 
            ha='center', va='center', fontweight='bold')
        
        # Connect heads to dendritic network
        ax.add_patch(FancyArrowPatch((head_x + layer_width, 2.6), 
                                (head_x + 2*layer_spacing, 1.4),
                                arrowstyle='->', lw=2, color='#E74C3C',
                                connectionstyle='arc3,rad=-0.2'))
        
        ax.add_patch(FancyArrowPatch((head_x + layer_width, 0.4), 
                                (head_x + 2*layer_spacing, 1.4),
                                arrowstyle='->', lw=2, color='#229954',
                                connectionstyle='arc3,rad=0.2'))
        
        # Labels for excitatory and inhibitory paths
        ax.text(head_x + layer_spacing, 2.2, 'Excitatory Path', 
            ha='center', va='center', color='#E74C3C', fontweight='bold')
        ax.text(head_x + layer_spacing, 0.7, 'Inhibitory Path', 
            ha='center', va='center', color='#229954', fontweight='bold')
        
        # Feature extraction label
        ax.text(layer_spacing, 0.0, 'Feature Extraction', ha='center', va='center', fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(-0.5, head_x + 3*layer_spacing)
        ax.set_ylim(-0.5, 3.5)
        ax.axis('off')
        
        plt.title('Input Transformation Network Architecture', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Input network architecture diagram saved to {save_path}")

    create_input_network_diagram() 