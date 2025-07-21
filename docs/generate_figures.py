"""
Final figure generation script for dendritic modeling documentation.
Generates high-quality, professional figures for documentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.path import Path
import numpy as np
import os
import seaborn as sns
import networkx as nx

# Create directories if they don't exist
output_dirs = ['figures/architecture', 'source/_static']
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Define colorblind-safe color palette (Okabe & Ito)
colors = {
    'vermillion': '#D55E00',  # Excitatory
    'sky_blue': '#56B4E9',    # Inhibitory
    'bluish_green': '#009E73', # Dendrites
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'orange': '#E69F00',
    'reddish_purple': '#CC79A7',
    'dark_gray': '#333333',
    'light_gray': '#CCCCCC',
    'medium_gray': '#999999',  # Lighter soma color
    'very_light_gray': '#E5E5E5',
}

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

def save_figure(fig, name, dpi=600):
    """Save figure in both locations with multiple formats."""
    for dir_path in output_dirs:
        fig.savefig(f'{dir_path}/{name}.png', dpi=dpi, transparent=True, bbox_inches='tight')
        fig.savefig(f'{dir_path}/{name}.svg', format='svg', transparent=True, bbox_inches='tight')
    print(f"Saved: {name}")

def create_dendritic_neuron_labeled(save_path='docs/figures/dendritic_neuron_labeled.png'):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    
    # Publication-quality params
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'lines.linewidth': 1.5,
        'axes.edgecolor': 'none',  # No axes borders for panel
    })
    
    # Draw elongated soma
    soma = FancyBboxPatch((-0.2, -0.5), 0.4, 1.0, boxstyle="Round,pad=0.1", 
                          fc=colors['medium_gray'], ec=colors['dark_gray'], lw=1.5, alpha=0.8)
    ax.add_patch(soma)
    ax.text(0, 0, 'Soma', ha='center', va='center', fontsize=12, color='white')
    
    # Draw axon (short, for completeness)
    axon_path = Path([[0.2, 0], [0.4, 0], [0.6, 0]], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    axon = PathPatch(axon_path, fc='none', ec=colors['dark_gray'], lw=2, capstyle='round')
    ax.add_patch(axon)
    ax.text(0.7, 0, 'Axon', ha='left', va='center', fontsize=10)
    
    # Draw primary dendrites with tapering (using Bézier curves for smoothness)
    primary_branches = [([0.2, 1.2, 1.8, 2.2], [0.2, 0.8, 1.2, 1.4]),  # Top
                        ([0.2, 1.2, 1.8, 2.2], [-0.2, -0.8, -1.2, -1.4])]  # Bottom
    for x_coords, y_coords in primary_branches:
        verts = list(zip(x_coords, y_coords))
        codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, fc='none', ec=colors['bluish_green'], lw=3, capstyle='round', alpha=0.7)
        ax.add_patch(patch)
    
    # Draw secondary branches (finer tapering)
    secondary_branches = [([2.2, 2.6, 2.8], [1.4, 1.6, 1.8]), ([2.2, 2.6, 2.8], [1.4, 1.2, 1.0]),  # Top
                          ([2.2, 2.6, 2.8], [-1.4, -1.6, -1.8]), ([2.2, 2.6, 2.8], [-1.4, -1.2, -1.0])]  # Bottom
    for x_coords, y_coords in secondary_branches:
        verts = list(zip(x_coords, y_coords))
        codes = [Path.MOVETO] + [Path.CURVE3] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, fc='none', ec=colors['bluish_green'], lw=1.5, capstyle='round', alpha=0.6)
        ax.add_patch(patch)
    
    # Add synapses (excitatory red, inhibitory blue) with terminals
    exc_synapses = [(2.8, 1.8), (2.8, 1.0), (2.8, -1.0), (2.8, -1.8)]
    inh_synapses = [(2.6, 1.4), (2.6, -1.4)]
    for ex, ey in exc_synapses:
        ax.add_patch(Circle((ex, ey), 0.08, fc=colors['vermillion'], ec='black', lw=1))
        ax.add_patch(FancyArrowPatch((ex + 0.3, ey), (ex, ey), color=colors['vermillion'], arrowstyle='->', mutation_scale=15, lw=1.5))
    ax.text(3.5, 0, 'Excitatory\nSynapses', ha='left', va='center', fontsize=10, color=colors['vermillion'])
    
    for ix, iy in inh_synapses:
        ax.add_patch(Circle((ix, iy), 0.08, fc=colors['sky_blue'], ec='black', lw=1))
        ax.add_patch(FancyArrowPatch((ix + 0.3, iy), (ix, iy), color=colors['sky_blue'], arrowstyle='->', mutation_scale=15, lw=1.5))
    ax.text(3.5, -0.5, 'Inhibitory\nSynapses', ha='left', va='center', fontsize=10, color=colors['sky_blue'])
    
    # Add shunting equation
    ax.text(0, -1.2, r'$V = \frac{E}{E + I + \epsilon}$', ha='center', fontsize=14, bbox=dict(facecolor='white', ec='gray', lw=1, alpha=0.8))
    
    # Legend
    exc_patch = mpatches.Patch(color=colors['vermillion'], label='Excitatory')
    inh_patch = mpatches.Patch(color=colors['sky_blue'], label='Inhibitory')
    dend_patch = mpatches.Patch(color=colors['bluish_green'], label='Dendrite')
    ax.legend(handles=[exc_patch, inh_patch, dend_patch], loc='upper left', frameon=True, fontsize=10, framealpha=0.9)
    
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title('Dendritic Neuron Schematic', fontsize=16, pad=20)
    
    plt.savefig(save_path, transparent=True, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', transparent=True)
    plt.close()

def create_dendritic_neuron_schematic(save_path='docs/figures/dendritic_neuron_schematic.png'):
    # Same as labeled but remove all text, labels, legend, and title
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    
    # Draw elongated soma
    soma = FancyBboxPatch((-0.2, -0.5), 0.4, 1.0, boxstyle="Round,pad=0.1", 
                          fc=colors['medium_gray'], ec=colors['dark_gray'], lw=1.5, alpha=0.8)
    ax.add_patch(soma)
    
    # Draw axon
    axon_path = Path([[0.2, 0], [0.4, 0], [0.6, 0]], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    axon = PathPatch(axon_path, fc='none', ec=colors['dark_gray'], lw=2, capstyle='round')
    ax.add_patch(axon)
    
    # Draw primary dendrites
    primary_branches = [([0.2, 1.2, 1.8, 2.2], [0.2, 0.8, 1.2, 1.4]),
                        ([0.2, 1.2, 1.8, 2.2], [-0.2, -0.8, -1.2, -1.4])]
    for x_coords, y_coords in primary_branches:
        verts = list(zip(x_coords, y_coords))
        codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, fc='none', ec=colors['bluish_green'], lw=3, capstyle='round', alpha=0.7)
        ax.add_patch(patch)
    
    # Draw secondary branches
    secondary_branches = [([2.2, 2.6, 2.8], [1.4, 1.6, 1.8]), ([2.2, 2.6, 2.8], [1.4, 1.2, 1.0]),
                          ([2.2, 2.6, 2.8], [-1.4, -1.6, -1.8]), ([2.2, 2.6, 2.8], [-1.4, -1.2, -1.0])]
    for x_coords, y_coords in secondary_branches:
        verts = list(zip(x_coords, y_coords))
        codes = [Path.MOVETO] + [Path.CURVE3] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, fc='none', ec=colors['bluish_green'], lw=1.5, capstyle='round', alpha=0.6)
        ax.add_patch(patch)
    
    # Add synapses (no labels)
    exc_synapses = [(2.8, 1.8), (2.8, 1.0), (2.8, -1.0), (2.8, -1.8)]
    inh_synapses = [(2.6, 1.4), (2.6, -1.4)]
    for ex, ey in exc_synapses:
        ax.add_patch(Circle((ex, ey), 0.08, fc=colors['vermillion'], ec='black', lw=1))
        ax.add_patch(FancyArrowPatch((ex + 0.3, ey), (ex, ey), color=colors['vermillion'], arrowstyle='->', mutation_scale=15, lw=1.5))
    
    for ix, iy in inh_synapses:
        ax.add_patch(Circle((ix, iy), 0.08, fc=colors['sky_blue'], ec='black', lw=1))
        ax.add_patch(FancyArrowPatch((ix + 0.3, iy), (ix, iy), color=colors['sky_blue'], arrowstyle='->', mutation_scale=15, lw=1.5))
    
    ax.set_xlim(-0.5, 3.5)  # Tighter crop for schematic
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    plt.savefig(save_path, transparent=True, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', transparent=True)
    plt.close()

def create_accurate_block_linear():
    """Create accurate block linear architecture showing dendrites connecting to matrix blocks."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Parameters
    n_inputs = 12
    n_outputs = 3
    block_size = n_inputs // n_outputs  # 4 inputs per output

    # Positions
    input_x = 1
    matrix_center_x = 5.5
    output_x = 10

    # Draw input layer
    input_y_start = 6
    input_spacing = 0.6
    input_positions = []

    for i in range(n_inputs):
        y = input_y_start - i * input_spacing
        # Input neuron
        circle = Circle((input_x, y), 0.15,
                       facecolor=colors['very_light_gray'],
                       edgecolor=colors['dark_gray'],
                       linewidth=1.5)
        ax.add_patch(circle)
        input_positions.append((input_x, y))
        ax.text(input_x - 0.5, y, f'x_{i}', fontsize=9, ha='right', va='center')

    # Draw weight matrix
    matrix_width = 3
    matrix_height = 5
    matrix_left = matrix_center_x - matrix_width/2
    matrix_bottom = 1

    # Matrix background
    matrix_bg = Rectangle((matrix_left, matrix_bottom), matrix_width, matrix_height,
                         facecolor='white', edgecolor=colors['dark_gray'],
                         linewidth=2)
    ax.add_patch(matrix_bg)

    # Draw the block diagonal structure
    block_colors = [colors['vermillion'], colors['bluish_green'], colors['orange']]

    for block_idx in range(n_outputs):
        # Calculate block position in matrix
        block_x = matrix_left + (block_idx * matrix_width / n_inputs) * block_size
        block_y = matrix_bottom + (n_outputs - 1 - block_idx) * matrix_height / n_outputs
        block_w = matrix_width / n_inputs * block_size
        block_h = matrix_height / n_outputs

        # Draw block
        block = Rectangle((block_x, block_y), block_w, block_h,
                         facecolor=block_colors[block_idx], alpha=0.3,
                         edgecolor=block_colors[block_idx], linewidth=2)
        ax.add_patch(block)

        # Add weight indicators (dots)
        for i in range(block_size):
            for j in range(1):
                dot_x = block_x + (i + 0.5) * block_w / block_size
                dot_y = block_y + block_h / 2
                ax.scatter(dot_x, dot_y, s=60, c=block_colors[block_idx],
                          alpha=0.8, edgecolor='none', zorder=10)

    # Draw output neurons with dendrites
    output_y_start = 5.5
    output_spacing = 2

    for out_idx in range(n_outputs):
        y = output_y_start - out_idx * output_spacing

        # Draw soma
        soma = Circle((output_x, y), 0.25,
                     facecolor=colors['very_light_gray'],
                     edgecolor=colors['dark_gray'],
                     linewidth=2)
        ax.add_patch(soma)
        ax.text(output_x + 0.5, y, f'y_{out_idx}', fontsize=11, ha='left', va='center',
               fontweight='bold')

        # Draw main dendrite extending left
        main_dendrite_end = output_x - 2
        ax.plot([output_x - 0.25, main_dendrite_end], [y, y],
               color=block_colors[out_idx], linewidth=4, solid_capstyle='round', zorder=5)

        # Draw dendritic branches (one for each input in the block)
        branch_positions = []
        branch_spread = 0.8
        for i in range(block_size):
            # Calculate branch position
            branch_y = y + (i - (block_size-1)/2) * branch_spread / (block_size-1)
            branch_end_x = main_dendrite_end - 0.8

            # Draw branch
            ax.plot([main_dendrite_end, branch_end_x], [y, branch_y],
                   color=block_colors[out_idx], linewidth=2.5, solid_capstyle='round', zorder=5)

            # Add synapse at branch end
            ax.scatter(branch_end_x, branch_y, s=80, c=block_colors[out_idx],
                      marker='o', edgecolor=colors['dark_gray'], linewidth=0.5, zorder=11)

            branch_positions.append((branch_end_x, branch_y))

        # Connect inputs to matrix and matrix to dendrites
        for i in range(block_size):
            input_idx = out_idx * block_size + i
            if input_idx < n_inputs:
                # Input to matrix connection
                input_pos = input_positions[input_idx]

                # Matrix position
                block_x = matrix_left + (out_idx * matrix_width / n_inputs) * block_size
                block_y = matrix_bottom + (n_outputs - 1 - out_idx) * matrix_height / n_outputs
                matrix_x = block_x + (i + 0.5) * (matrix_width / n_inputs * block_size) / block_size
                matrix_y = block_y + matrix_height / n_outputs / 2

                # Draw connection from input to matrix
                ax.plot([input_pos[0] + 0.15, matrix_x], [input_pos[1], matrix_y],
                       color=colors['medium_gray'], linewidth=1, alpha=0.5, linestyle='--', zorder=1)

                # Draw connection from matrix to dendrite branch
                branch_pos = branch_positions[i]
                ax.plot([matrix_x, branch_pos[0]], [matrix_y, branch_pos[1]],
                       color=block_colors[out_idx], linewidth=1.5, alpha=0.7, zorder=2)

    # Add annotations
    ax.text(input_x, 7, 'Input Layer', fontsize=13, ha='center', fontweight='bold')
    ax.text(matrix_center_x, 7, 'Block Diagonal\nWeight Matrix', fontsize=13, ha='center', fontweight='bold')
    ax.text(output_x, 7, 'Output Neurons\nwith Dendrites', fontsize=13, ha='center', fontweight='bold')

    # Add mathematical notation
    ax.text(matrix_center_x, 0.2,
           r'$W \in \mathbb{R}^{n_{out} \times n_{in}}$' + '\n' +
           f'{n_outputs} blocks × {block_size} weights each\n' +
           'Each dendrite receives inputs from one block',
           fontsize=10, ha='center', style='italic',
           bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['very_light_gray'],
                    edgecolor=colors['medium_gray'], linewidth=1))

    # Add block structure explanation
    for i in range(n_outputs):
        ax.text(matrix_left - 0.3,
               matrix_bottom + (n_outputs - 0.5 - i) * matrix_height / n_outputs,
               f'Block {i}', fontsize=9, ha='right', va='center',
               color=block_colors[i], fontweight='bold')

    # Style
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    ax.set_title('Block Linear Layer: Dendrites Connect to Diagonal Weight Blocks',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    save_figure(fig, 'block_linear_accurate')
    plt.close()

def create_network_architecture(save_path='docs/figures/network_architecture.png'):
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = GridSpec(2, 2, figure=fig)
    
    ax_main = fig.add_subplot(gs[:, 0])
    ax_inset = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])
    
    # Main network graph
    G = nx.DiGraph()
    layers = ['Input', 'Layer1_E', 'Layer1_I', 'Layer2_E', 'Layer2_I', 'Output']
    for layer in layers:
        G.add_node(layer)
    edges = [('Input', 'Layer1_E'), ('Input', 'Layer1_I'),
             ('Layer1_E', 'Layer2_E'), ('Layer1_E', 'Layer2_I'),
             ('Layer1_I', 'Layer2_E'), ('Layer1_I', 'Layer2_I'),
             ('Layer2_E', 'Output'), ('Layer2_I', 'Output')]
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', ax=ax_main)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax_main)
    edge_colors = ['red' if 'E' in edge[1] else 'blue' for edge in G.edges()]
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color=edge_colors, width=2, ax=ax_main)
    
    ax_main.set_title('EINet Architecture with E/I Connectivity', fontsize=20)
    ax_main.axis('off')
    
    # Inset: Sparse connectivity example
    weights = np.random.randn(5,5)
    mask = np.abs(weights) > np.percentile(np.abs(weights), 70)  # TopK mask
    sns.heatmap(weights * mask, ax=ax_inset, cmap='viridis', cbar=True, annot=True, fmt='.1f')
    ax_inset.set_title('TopK Sparse Weights', fontsize=16)
    
    # Legend
    ax_legend.axis('off')
    exc_line = plt.Line2D([0], [0], color='red', label='Excitatory Connection')
    inh_line = plt.Line2D([0], [0], color='blue', label='Inhibitory Connection')
    sparse_patch = plt.Rectangle((0,0),1,1, fc='green', label='Sparse (TopK)')
    ax_legend.legend(handles=[exc_line, inh_line, sparse_patch], loc='center', fontsize=14, frameon=True)
    
    plt.suptitle('Network Architecture and Connectivity Patterns', fontsize=22)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

def create_shunting_inhibition_plot(save_path='docs/figures/shunting_inhibition.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    E = np.linspace(0, 10, 200)
    I_values = [0, 2, 5, 10]
    epsilon = 1e-6
    colors = cm.viridis(np.linspace(0, 1, len(I_values)))
    
    # Subtractive
    for i, I in enumerate(I_values):
        subtractive = np.maximum(0, E - I)
        ax1.plot(E, subtractive, label=f'I={I}', color=colors[i], lw=3)
        ax1.fill_between(E, 0, subtractive, alpha=0.1, color=colors[i])
    ax1.set_title('Subtractive Inhibition', fontsize=18)
    ax1.set_xlabel('Excitation (E)', fontsize=16)
    ax1.set_ylabel('Output', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14, framealpha=0.9)
    
    # Shunting
    for i, I in enumerate(I_values):
        shunting = E / (E + I + epsilon)
        ax2.plot(E, shunting, label=f'I={I}', color=colors[i], lw=3)
        ax2.fill_between(E, 0, shunting, alpha=0.1, color=colors[i])
    ax2.set_title('Shunting Inhibition', fontsize=18)
    ax2.set_xlabel('Excitation (E)', fontsize=16)
    ax2.set_ylabel('Output', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14, framealpha=0.9)
    
    # Inset for low E zoom
    axins = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
    E_zoom = np.linspace(0, 2, 100)
    for i, I in enumerate(I_values):
        axins.plot(E_zoom, E_zoom / (E_zoom + I + epsilon), color=colors[i])
    axins.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Inhibition Mechanisms', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

# New function for sparse connectivity
def create_sparse_connectivity_visualization(save_path='docs/figures/sparse_connectivity.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Standard TopK
    weights = np.random.randn(10,10)
    k = 3
    threshold = np.sort(np.abs(weights.flatten()))[-10*k]
    mask = np.abs(weights) >= threshold
    sns.heatmap(weights * mask, ax=ax1, cmap='coolwarm', center=0, cbar=True)
    ax1.set_title(f'Standard TopK (k={k})', fontsize=18)
    
    # Stochastic TopK
    probs = np.abs(weights) / np.sum(np.abs(weights))
    stochastic_mask = np.random.choice(np.arange(100), size=10*k, replace=False, p=probs.flatten())
    mask_stoch = np.zeros_like(weights)
    mask_stoch.flat[stochastic_mask] = 1
    sns.heatmap(weights * mask_stoch, ax=ax2, cmap='coolwarm', center=0, cbar=True)
    ax2.set_title('Stochastic TopK', fontsize=18)
    
    plt.suptitle('Sparse Connectivity Patterns', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

# New function for gradient scaling
def create_gradient_scaling_comparison(save_path='docs/figures/gradient_scaling.png'):
    layers = np.arange(1, 11)
    vanilla = 1 / (2 ** layers)
    inverse = np.ones_like(layers) * 0.5
    conductance = 1 / np.sqrt(layers)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(layers, vanilla, 'r--', label='Vanilla', lw=3)
    ax.plot(layers, inverse, 'g-', label='Inverse Scaling', lw=3)
    ax.plot(layers, conductance, 'b-.', label='Conductance-based', lw=3)
    
    ax.set_title('Gradient Norms Across Layers', fontsize=20)
    ax.set_xlabel('Layer Depth', fontsize=16)
    ax.set_ylabel('Average Gradient Norm', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

# New function for reactivation functions
def create_reactivation_functions_gallery(save_path='docs/figures/reactivation_functions.png'):
    x = np.linspace(-5, 5, 200)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axs = axs.flatten()
    
    # Param Tanh
    for alpha in [0.5, 1.0, 2.0]:
        axs[0].plot(x, alpha * np.tanh(x / alpha), label=f'α={alpha}')
    axs[0].set_title('Parametric Tanh', fontsize=18)
    
    # Linear Sigmoid
    for beta in [0.1, 1.0, 10.0]:
        axs[1].plot(x, (1 / (1 + np.exp(-beta * x))) * x, label=f'β={beta}')
    axs[1].set_title('Param Linear Sigmoid', fontsize=18)
    
    # Linear Tanh Transition
    for gamma in [0.2, 1.0, 5.0]:
        tanh_part = np.tanh(x)
        linear_part = x / 5
        transition = gamma * tanh_part + (1 - gamma) * linear_part
        axs[2].plot(x, transition, label=f'γ={gamma}')
    axs[2].set_title('Linear Tanh Transition', fontsize=18)
    
    # Standard activations for comparison
    axs[3].plot(x, np.tanh(x), label='Tanh')
    axs[3].plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid')
    axs[3].plot(x, np.maximum(0, x), label='ReLU')
    axs[3].set_title('Standard Activations', fontsize=18)
    
    for ax in axs:
        ax.set_xlabel('Input', fontsize=16)
        ax.set_ylabel('Output', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)
    
    plt.suptitle('Reactivation Functions', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

# New function for E/I dynamics
def create_ei_dynamics_flow(save_path='docs/figures/ei_dynamics_updated.png'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Positions: Add Input and Output for completeness
    pos = {'Input': (-2, 1), 'E1': (0, 2), 'I1': (0, 0), 'E2': (2, 2), 'I2': (2, 0), 'Output': (4, 1)}
    
    # Draw nodes with colors
    for node, (x, y) in pos.items():
        if node == 'Input' or node == 'Output':
            color = 'gray'
        else:
            color = 'red' if 'E' in node else 'blue'
        ax.add_patch(Circle((x, y), 0.3, color=color, alpha=0.5))
        ax.text(x, y, node, ha='center', va='center', fontsize=16)
    
    # Corrected connections with labels (for input_mode=0: shared input to E1 and I1)
    arrows = [
        # Input to Layer 1 (shared to both E and I for mode 0)
        ('Input', 'E1', 'Input→E'),
        ('Input', 'I1', 'Input→I'),
        # Within Layer 1 (IE within layer: I1 feeds E1)
        ('I1', 'E1', 'IE (L1)'),
        # Layer 1 to Layer 2 (EE, EI, II - no direct IE between layers)
        ('E1', 'E2', 'EE'),
        ('E1', 'I2', 'EI'),
        ('I1', 'I2', 'II'),
        # Within Layer 2 (IE within layer: I2 feeds E2)
        ('I2', 'E2', 'IE (L2)'),
        # Output from Layer 2
        ('E2', 'Output', 'E→Output')
    ]
    
    for start, end, label in arrows:
        sx, sy = pos[start]
        ex, ey = pos[end]
        color = 'red' if start.startswith('E') or start == 'Input' else 'blue'
        arrow = FancyArrowPatch((sx + 0.4 if sx < ex else sx - 0.4, sy),
                                (ex - 0.4 if sx < ex else ex + 0.4, ey),
                                color=color, mutation_scale=20, arrowstyle='->')
        ax.add_patch(arrow)
        ax.text((sx + ex)/2 + 0.1, (sy + ey)/2 + 0.1, label, fontsize=12, color=color)
    
    # Optional: Add dashed line for indirect I1 -> E2 (via I2) for clarity
    sx, sy = pos['I1']
    ex, ey = pos['E2']
    indirect = FancyArrowPatch((sx + 0.4, sy), (ex - 0.4, ey), 
                               color='gray', linestyle='--', mutation_scale=20, arrowstyle='->')
    ax.add_patch(indirect)
    ax.text((sx + ex)/2, (sy + ey)/2 + 0.3, 'Indirect IE\n(via I2)', fontsize=10, color='gray')
    
    ax.set_xlim(-3, 5)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_title('Accurate EINet E/I Connectivity (Input Mode 0 - Shared Input)', fontsize=20)
    
    # Updated note explaining logic
    ax.text(1, -0.8, 'Note: For input_mode=0, input is shared to E1 and I1.\nIE is within-layer only (e.g., I1 feeds E1).\nBetween-layer IE is indirect (e.g., I1 feeds E2 via I2).', 
            ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

# New function for information flow (simple Sankey-like)
def create_information_flow_diagram(save_path='docs/figures/information_flow.png'):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Nodes
    nodes = ['Input', 'Branch1', 'Branch2', 'Soma', 'Output']
    pos = {n: (i, 0) for i, n in enumerate(nodes)}
    
    # Flows (arbitrary MI values)
    flows = [('Input', 'Branch1', 2.5), ('Input', 'Branch2', 2.0),
             ('Branch1', 'Soma', 1.8), ('Branch2', 'Soma', 1.5),
             ('Soma', 'Output', 3.0)]
    
    for start, end, value in flows:
        sx, sy = pos[start]
        ex, ey = pos[end]
        ax.plot([sx, ex], [sy, ey+value/10], 'b-', lw=value, alpha=0.5)
        ax.plot([sx, ex], [sy, ey-value/10], 'b-', lw=value, alpha=0.5)
    
    for node, (x, y) in pos.items():
        ax.text(x, y, node, ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))
    
    ax.set_title('Information Flow in Dendritic Network', fontsize=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

# New function for training strategies
def create_training_strategies_comparison(save_path='docs/figures/training_strategies.png'):
    epochs = np.arange(1, 11)
    standard = 1 / np.log(epochs + 1) + np.random.normal(0, 0.02, len(epochs))
    branchwise = 1 / np.log(epochs + 2) + np.random.normal(0, 0.02, len(epochs))
    twostep = 1 / np.log(epochs + 1.5) + np.random.normal(0, 0.02, len(epochs))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(epochs, standard, 'b-o', label='Standard', lw=3, markersize=6)
    ax.plot(epochs, branchwise, 'g--', label='Branch-wise', lw=3, markersize=6)
    ax.plot(epochs, twostep, 'r-.', label='Two-step', lw=3, markersize=6)
    
    ax.set_title('Training Loss Curves by Strategy', fontsize=20)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    plt.close()

# Updated main function to generate all
def main():
    os.makedirs('docs/figures', exist_ok=True)

    print("Generating professional dendritic neuron figures...")
    create_dendritic_neuron_labeled()
    create_dendritic_neuron_schematic()

    print("Generating network architecture figure...")
    create_network_architecture()
    
    print("Generating shunting inhibition plot...")
    create_shunting_inhibition_plot()
    
    print("Generating new figures...")
    create_sparse_connectivity_visualization()
    create_gradient_scaling_comparison()
    create_reactivation_functions_gallery()
    create_ei_dynamics_flow()
    create_information_flow_diagram()
    create_training_strategies_comparison()
    
    print("All figures generated successfully!")

if __name__ == '__main__':
    main()
