"""
Final figure generation script for dendritic modeling documentation.
Generates high-quality, professional figures for documentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, FancyBboxPatch, PathPatch, Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.path import Path
import numpy as np
import os
import seaborn as sns
import networkx as nx

# Create directory if it doesn't exist
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

# Define colorblind-safe color palette (Okabe & Ito) with extensions for professionalism
colors = {
    'excitatory': '#D55E00',  # Vermillion for E
    'inhibitory': '#56B4E9',  # Sky blue for I
    'dendrite': '#009E73',    # Bluish green for dendrites
    'soma': '#999999',        # Medium gray for soma
    'input': '#E69F00',       # Orange for inputs
    'output': '#0072B2',      # Blue for outputs
    'background': '#F0F0F0',  # Very light gray for backgrounds
    'text': '#333333',        # Dark gray for text
    'highlight': '#F0E442',   # Yellow for highlights
    'connection_ee': '#D55E00',  # EE connections
    'connection_ei': '#D55E00',  # EI
    'connection_ie': '#56B4E9',  # IE
    'connection_ii': '#56B4E9',  # II
}

# Enhanced global matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.grid': False,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.transparent': True,
    'lines.linewidth': 1.5,
    'axes.edgecolor': colors['text'],
    'text.color': colors['text'],
})

def save_figure(fig, name, dpi=600):
    """Save figure in the output directory with PNG and SVG formats."""
    fig.savefig(f'{output_dir}/{name}.png', dpi=dpi, transparent=True, bbox_inches='tight')
    fig.savefig(f'{output_dir}/{name}.svg', format='svg', transparent=True, bbox_inches='tight')
    print(f"Saved: {name}")
    plt.close(fig)

def create_dendritic_neuron_labeled():
    """Create labeled dendritic neuron schematic with hierarchical branching."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    
    # Draw soma (ellipse for more biological look)
    soma = Ellipse((0, 0), width=0.6, height=0.8, fc=colors['soma'], ec=colors['text'], lw=2, alpha=0.9)
    ax.add_patch(soma)
    ax.text(0, 0, 'Soma', ha='center', va='center', fontsize=14, color='white', weight='bold')
    
    # Draw axon
    axon_path = Path([[0.3, 0], [0.6, 0], [0.9, 0]], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    axon = PathPatch(axon_path, fc='none', ec=colors['text'], lw=2.5, capstyle='round')
    ax.add_patch(axon)
    ax.text(1.0, 0, 'Axon', ha='left', va='center', fontsize=12)
    
    # Hierarchical dendrites (multi-level branching)
    # Primary dendrite
    primary_path = Path([[-0.3, 0], [-1.0, 0.5], [-1.5, 1.0]], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    primary = PathPatch(primary_path, fc='none', ec=colors['dendrite'], lw=4, capstyle='round', alpha=0.8)
    ax.add_patch(primary)
    
    # Secondary branches from primary
    secondary_paths = [
        ([-1.5, 1.0], [-1.8, 1.5], [-2.0, 2.0]),
        ([-1.5, 1.0], [-1.8, 0.5], [-2.0, 0.0])
    ]
    for sp in secondary_paths:
        sp_path = Path(sp, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        sp_patch = PathPatch(sp_path, fc='none', ec=colors['dendrite'], lw=2.5, capstyle='round', alpha=0.7)
        ax.add_patch(sp_patch)
    
    # Tertiary branches
    tertiary_paths = [
        ([-2.0, 2.0], [-2.2, 2.3], [-2.4, 2.5]),
        ([-2.0, 2.0], [-2.2, 1.7], [-2.4, 1.5]),
        ([-2.0, 0.0], [-2.2, 0.3], [-2.4, 0.5]),
        ([-2.0, 0.0], [-2.2, -0.3], [-2.4, -0.5])
    ]
    for tp in tertiary_paths:
        tp_path = Path(tp, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        tp_patch = PathPatch(tp_path, fc='none', ec=colors['dendrite'], lw=1.5, capstyle='round', alpha=0.6)
        ax.add_patch(tp_patch)
    
    # Synapses (excitatory and inhibitory at terminals)
    exc_syn = [(-2.4, 2.5), (-2.4, 1.5)]
    inh_syn = [(-2.4, 0.5), (-2.4, -0.5)]
    for pos in exc_syn:
        ax.add_patch(Circle(pos, 0.1, fc=colors['excitatory'], ec=colors['text'], lw=1.5))
    for pos in inh_syn:
        ax.add_patch(Circle(pos, 0.1, fc=colors['inhibitory'], ec=colors['text'], lw=1.5))
    
    # Labels
    ax.text(-2.5, 2.8, 'Excitatory Synapses', ha='right', va='bottom', fontsize=12, color=colors['excitatory'])
    ax.text(-2.5, -0.8, 'Inhibitory Synapses', ha='right', va='top', fontsize=12, color=colors['inhibitory'])
    ax.text(-1.0, 1.2, 'Dendritic Branches', ha='center', va='bottom', fontsize=12, color=colors['dendrite'])
    
    # Equation
    ax.text(0, -1.0, r'$V = \frac{E}{E + I + \epsilon}$ (Shunting Inhibition)', ha='center', fontsize=14, bbox=dict(fc='white', ec=colors['text'], alpha=0.8))
    
    # Legend
    exc_patch = mpatches.Patch(color=colors['excitatory'], label='Excitatory')
    inh_patch = mpatches.Patch(color=colors['inhibitory'], label='Inhibitory')
    dend_patch = mpatches.Patch(color=colors['dendrite'], label='Dendrite')
    soma_patch = mpatches.Patch(color=colors['soma'], label='Soma')
    ax.legend(handles=[exc_patch, inh_patch, dend_patch, soma_patch], loc='upper right', frameon=False, fontsize=12)
    
    ax.set_xlim(-3, 2)
    ax.set_ylim(-2, 3)
    ax.axis('off')
    ax.set_title('Hierarchical Dendritic Neuron Schematic', fontsize=18, pad=20)
    
    save_figure(fig, 'dendritic_neuron_labeled')

def create_dendritic_neuron_schematic():
    """Create unlabeled dendritic neuron schematic with hierarchical branching."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    
    # Similar to labeled but without text/labels/legend
    # Draw soma
    soma = Ellipse((0, 0), width=0.6, height=0.8, fc=colors['soma'], ec=colors['text'], lw=2, alpha=0.9)
    ax.add_patch(soma)
    
    # Draw axon
    axon_path = Path([[0.3, 0], [0.6, 0], [0.9, 0]], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    axon = PathPatch(axon_path, fc='none', ec=colors['text'], lw=2.5, capstyle='round')
    ax.add_patch(axon)
    
    # Hierarchical dendrites (as above)
    primary_path = Path([[-0.3, 0], [-1.0, 0.5], [-1.5, 1.0]], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    primary = PathPatch(primary_path, fc='none', ec=colors['dendrite'], lw=4, capstyle='round', alpha=0.8)
    ax.add_patch(primary)
    
    secondary_paths = [
        ([-1.5, 1.0], [-1.8, 1.5], [-2.0, 2.0]),
        ([-1.5, 1.0], [-1.8, 0.5], [-2.0, 0.0])
    ]
    for sp in secondary_paths:
        sp_path = Path(sp, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        sp_patch = PathPatch(sp_path, fc='none', ec=colors['dendrite'], lw=2.5, capstyle='round', alpha=0.7)
        ax.add_patch(sp_patch)
    
    tertiary_paths = [
        ([-2.0, 2.0], [-2.2, 2.3], [-2.4, 2.5]),
        ([-2.0, 2.0], [-2.2, 1.7], [-2.4, 1.5]),
        ([-2.0, 0.0], [-2.2, 0.3], [-2.4, 0.5]),
        ([-2.0, 0.0], [-2.2, -0.3], [-2.4, -0.5])
    ]
    for tp in tertiary_paths:
        tp_path = Path(tp, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        tp_patch = PathPatch(tp_path, fc='none', ec=colors['dendrite'], lw=1.5, capstyle='round', alpha=0.6)
        ax.add_patch(tp_patch)
    
    # Synapses
    exc_syn = [(-2.4, 2.5), (-2.4, 1.5)]
    inh_syn = [(-2.4, 0.5), (-2.4, -0.5)]
    for pos in exc_syn:
        ax.add_patch(Circle(pos, 0.1, fc=colors['excitatory'], ec=colors['text'], lw=1.5))
    for pos in inh_syn:
        ax.add_patch(Circle(pos, 0.1, fc=colors['inhibitory'], ec=colors['text'], lw=1.5))
    
    ax.set_xlim(-3, 2)
    ax.set_ylim(-2, 3)
    ax.axis('off')
    
    save_figure(fig, 'dendritic_neuron_schematic')

def create_dendrinet_hierarchy():
    """Create schematic for DendriNet hierarchical branching structure."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
    
    # Soma
    soma = Ellipse((5, 0), width=1, height=1.5, fc=colors['soma'], ec=colors['text'], lw=2)
    ax.add_patch(soma)
    ax.text(5, 0, 'Soma\n(64)', ha='center', va='center', fontsize=14, color='white', weight='bold')
    
    # Branch levels (e.g., branch_factors=[4,2])
    # Level 1: 4 branches from soma
    branch1_pos = [(3, 1.5), (3, 0.5), (3, -0.5), (3, -1.5)]
    for pos in branch1_pos:
        ax.plot([5-0.5, pos[0]], [0, pos[1]], color=colors['dendrite'], lw=4)
        ax.add_patch(Circle(pos, 0.3, fc=colors['dendrite'], alpha=0.5))
        ax.text(pos[0], pos[1], 'Branch1\n(128 total)', ha='center', va='center', fontsize=10)
    
    # Level 2: 2 sub-branches per branch1 (8 total)
    branch2_pos = []
    for b1 in branch1_pos:
        sub1 = (1, b1[1] + 0.5)
        sub2 = (1, b1[1] - 0.5)
        ax.plot([b1[0], sub1[0]], [b1[1], sub1[1]], color=colors['dendrite'], lw=2.5)
        ax.plot([b1[0], sub2[0]], [b1[1], sub2[1]], color=colors['dendrite'], lw=2.5)
        ax.add_patch(Circle(sub1, 0.2, fc=colors['dendrite'], alpha=0.4))
        ax.add_patch(Circle(sub2, 0.2, fc=colors['dendrite'], alpha=0.4))
        branch2_pos.extend([sub1, sub2])
    
    # Inputs to level 2
    for b2 in branch2_pos:
        input_pos = (-1, b2[1])
        ax.plot([b2[0], input_pos[0]], [b2[1], input_pos[1]], color=colors['input'], lw=1.5, ls='--')
        ax.add_patch(Rectangle(input_pos, 0.5, 0.5, fc=colors['input'], alpha=0.6))
        ax.text(input_pos[0]+0.25, input_pos[1]+0.25, 'Input\n(784)', ha='center', va='center', fontsize=8)
    
    # Labels
    ax.text(4, 2, 'Somatic Synapses', ha='center', fontsize=12)
    ax.text(2, 2, 'Primary Dendrites', ha='center', fontsize=12)
    ax.text(0, 2, 'Secondary Dendrites', ha='center', fontsize=12)
    
    ax.set_xlim(-2, 6)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    ax.set_title('DendriNet Hierarchical Branching Structure', fontsize=18, pad=20)
    
    save_figure(fig, 'dendrinet_hierarchy')

def create_transfer_layer_modes():
    """Create diagram showing TransferLayer input modes."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
    
    for i, mode in enumerate([0, 1]):
        ax = axs[i]
        # Input
        input_rect = Rectangle((0, 0), 1, 2, fc=colors['input'], ec=colors['text'], lw=2)
        ax.add_patch(input_rect)
        ax.text(0.5, 1, 'Input', ha='center', va='center', fontsize=14)
        
        # E path
        e_rect = Rectangle((3, 1.5), 1, 1, fc=colors['excitatory'], ec=colors['text'], lw=2)
        ax.add_patch(e_rect)
        ax.text(3.5, 1.75, 'E Path', ha='center', va='center', fontsize=12)
        
        # I path
        i_rect = Rectangle((3, -0.5), 1, 1, fc=colors['inhibitory'], ec=colors['text'], lw=2)
        ax.add_patch(i_rect)
        ax.text(3.5, -0.25, 'I Path', ha='center', va='center', fontsize=12)
        
        # Connections
        ax.add_patch(FancyArrowPatch((1, 1.5), (3, 1.75), color=colors['excitatory'], arrowstyle='->', mutation_scale=20, lw=2))
        if mode == 0:
            ax.add_patch(FancyArrowPatch((1, 0.5), (3, -0.25), color=colors['inhibitory'], arrowstyle='->', mutation_scale=20, lw=2))
            title = 'Mode 0: Shared Input to E & I'
        else:
            ax.text(3.5, 0, 'None', ha='center', fontsize=12, color=colors['text'])
            title = 'Mode 1: Input to E Only'
        
        ax.set_title(title, fontsize=16)
        ax.set_xlim(-1, 5)
        ax.set_ylim(-2, 3)
        ax.axis('off')
    
    fig.suptitle('TransferLayer Input Transformation Modes', fontsize=18)
    
    save_figure(fig, 'transfer_layer_modes')

def create_accurate_block_linear():
    """Create accurate block linear architecture showing dendrites connecting to matrix blocks. (Enhanced for clarity)"""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=600)

    # Parameters (as original)
    n_inputs = 12
    n_outputs = 3
    block_size = n_inputs // n_outputs

    # Positions (adjusted for better spacing)
    input_x = 0.5
    matrix_center_x = 6
    output_x = 11.5

    # Input layer (vertical alignment)
    input_y_start = 7
    input_spacing = 0.6  # Fixed: was 0.0, causing overlap
    input_positions = []
    for i in range(n_inputs):
        y = input_y_start - i * input_spacing
        circle = Circle((input_x, y), 0.2, fc=colors['input'], ec=colors['text'], lw=1.5)
        ax.add_patch(circle)
        input_positions.append((input_x, y))
        ax.text(input_x - 0.5, y, f'x_{i}', fontsize=10, ha='right', va='center')  # Fixed: corrected args (removed extra 0.5 positional)

    # Weight matrix (with grid for blocks)
    matrix_width = 4
    matrix_height = 6
    matrix_left = matrix_center_x - matrix_width / 2
    matrix_bottom = 0.5

    matrix_bg = Rectangle((matrix_left, matrix_bottom), matrix_width, matrix_height, fc='white', ec=colors['text'], lw=2, alpha=0.9)
    ax.add_patch(matrix_bg)
    
    # Block colors
    block_colors = [colors['excitatory'], colors['dendrite'], colors['output']]

    for block_idx in range(n_outputs):
        block_x = matrix_left + (block_idx * matrix_width / n_outputs)
        block_y = matrix_bottom + (n_outputs - 1 - block_idx) * matrix_height / n_outputs
        block_w = matrix_width / n_outputs
        block_h = matrix_height / n_outputs

        block = Rectangle((block_x, block_y), block_w, block_h, fc=block_colors[block_idx], alpha=0.2, ec=block_colors[block_idx], lw=1.5)
        ax.add_patch(block)

        # Dots for weights
        for i in range(block_size):
            dot_x = block_x + (i + 0.5) * block_w / block_size
            dot_y = block_y + block_h / 2
            ax.scatter(dot_x, dot_y, s=50, c=block_colors[block_idx], alpha=0.8, edgecolor='none')

    # Output neurons with dendrites
    output_y_start = 6.5
    output_spacing = 2.5

    for out_idx in range(n_outputs):
        y = output_y_start - out_idx * output_spacing
        soma = Circle((output_x, y), 0.3, fc=colors['soma'], ec=colors['text'], lw=2)
        ax.add_patch(soma)
        ax.text(output_x + 0.4, y, f'y_{out_idx}', fontsize=12, ha='left', va='center')

        # Main dendrite
        main_end = output_x - 2.5
        ax.plot([output_x - 0.3, main_end], [y, y], color=block_colors[out_idx], lw=5, solid_capstyle='round')

        # Branches
        branch_positions = []
        branch_spread = 1.0
        for i in range(block_size):
            branch_y = y + (i - (block_size-1)/2) * branch_spread / (block_size-1)
            branch_end_x = main_end - 1.0
            ax.plot([main_end, branch_end_x], [y, branch_y], color=block_colors[out_idx], lw=3, solid_capstyle='round')
            ax.scatter(branch_end_x, branch_y, s=100, c=block_colors[out_idx], edgecolor=colors['text'], lw=1)
            branch_positions.append((branch_end_x, branch_y))

        # Connections from inputs to matrix to branches
        for i in range(block_size):
            input_idx = out_idx * block_size + i
            if input_idx < n_inputs:
                input_pos = input_positions[input_idx]
                block_x = matrix_left + (out_idx * matrix_width / n_outputs)
                matrix_x = block_x + (i + 0.5) * (matrix_width / n_outputs) / block_size
                matrix_y = matrix_bottom + (n_outputs - 0.5 - out_idx) * matrix_height / n_outputs
                ax.plot([input_pos[0] + 0.2, matrix_x], [input_pos[1], matrix_y], color=colors['background'], lw=1, alpha=0.6, ls='--')
                branch_pos = branch_positions[i]
                ax.plot([matrix_x, branch_pos[0]], [matrix_y, branch_pos[1]], color=block_colors[out_idx], lw=2, alpha=0.8)

    # Annotations
    ax.text(input_x, 8, 'Input Layer', ha='center', fontsize=14, weight='bold')
    ax.text(matrix_center_x, 8, 'Block-Diagonal Weight Matrix', ha='center', fontsize=14, weight='bold')
    ax.text(output_x, 8, 'Output Neurons with Dendrites', ha='center', fontsize=14, weight='bold')
    
    # Math note (fixed for mathtext compatibility)
    ax.text(matrix_center_x, -0.5, r'$W = \left[ \begin{array}{ccc} B_1 & 0 & 0 \\ 0 & B_2 & 0 \\ 0 & 0 & B_3 \end{array} \right]$', ha='center', fontsize=14, bbox=dict(fc='white', alpha=0.8))

    ax.set_xlim(0, 13)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_title('BlockLinear: Dendritic Integration of Weight Blocks', fontsize=18, pad=20)
    
    save_figure(fig, 'block_linear_accurate')

def create_network_architecture():
    """Create enhanced EINet architecture with labeled connections."""
    fig = plt.figure(figsize=(14, 10), dpi=600)
    gs = GridSpec(2, 2, figure=fig)
    
    ax_main = fig.add_subplot(gs[:, 0])
    ax_inset = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])
    
    # Main graph with layered layout
    G = nx.DiGraph()
    layers = ['Input', 'Layer1_E', 'Layer1_I', 'Layer2_E', 'Layer2_I', 'Output']
    for layer in layers:
        G.add_node(layer)
    edges = [
        ('Input', 'Layer1_E'), ('Input', 'Layer1_I'),
        ('Layer1_E', 'Layer2_E', {'label': 'EE'}), ('Layer1_E', 'Layer2_I', {'label': 'EI'}),
        ('Layer1_I', 'Layer2_E', {'label': 'IE'}), ('Layer1_I', 'Layer2_I', {'label': 'II'}),
        ('Layer2_E', 'Output'), ('Layer2_I', 'Output')
    ]
    G.add_edges_from([(u, v) for u, v, d in edges])
    
    pos = {'Input': (0, 1), 'Layer1_E': (1, 1.5), 'Layer1_I': (1, 0.5), 
           'Layer2_E': (2, 1.5), 'Layer2_I': (2, 0.5), 'Output': (3, 1)}
    
    node_colors = ['#E69F00' if n=='Input' else colors['excitatory'] if 'E' in n else colors['inhibitory'] if 'I' in n else colors['output'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, ax=ax_main, edgecolors=colors['text'], lw=2)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax_main)
    
    edge_colors = [colors['connection_ee'] if d.get('label')=='EE' else colors['connection_ei'] if d.get('label')=='EI' else colors['connection_ie'] if d.get('label')=='IE' else colors['connection_ii'] for u,v,d in edges]
    nx.draw_networkx_edges(G, pos, edgelist=[(u,v) for u,v,d in edges], arrowstyle='->', arrowsize=25, edge_color=edge_colors, width=2.5, ax=ax_main)
    
    # Add edge labels
    for (u, v, d) in edges:
        if 'label' in d:
            x = (pos[u][0] + pos[v][0]) / 2 + 0.1
            y = (pos[u][1] + pos[v][1]) / 2 + 0.1
            ax_main.text(x, y, d['label'], fontsize=12, color=edge_colors[edges.index((u,v,d))], weight='bold')
    
    ax_main.set_title('EINet Architecture with E/I Connectivity', fontsize=18)
    ax_main.axis('off')
    
    # Inset: Sparse weights
    weights = np.random.randn(5,5)
    mask = np.abs(weights) > np.percentile(np.abs(weights), 70)
    sns.heatmap(weights * mask, ax=ax_inset, cmap='coolwarm', center=0, cbar=True, annot=True, fmt='.1f', square=True)
    ax_inset.set_title('TopK Sparse Weights Example', fontsize=14)
    
    # Legend
    ax_legend.axis('off')
    handles = [
        plt.Line2D([0], [0], color=colors['connection_ee'], label='EE (Excitatory → Excitatory)'),
        plt.Line2D([0], [0], color=colors['connection_ei'], label='EI (Excitatory → Inhibitory)'),
        plt.Line2D([0], [0], color=colors['connection_ie'], label='IE (Inhibitory → Excitatory)'),
        plt.Line2D([0], [0], color=colors['connection_ii'], label='II (Inhibitory → Inhibitory)'),
        mpatches.Patch(color='green', label='Sparse (TopK)')
    ]
    ax_legend.legend(handles=handles, loc='center', fontsize=12, frameon=False)
    
    fig.suptitle('EINet Network Architecture and Connectivity', fontsize=20)
    
    save_figure(fig, 'network_architecture')

def create_shunting_inhibition_plot():
    """Create enhanced plot comparing inhibition mechanisms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600, sharey=True)
    
    E = np.linspace(0, 10, 200)
    I_values = [0, 2, 5, 10]
    epsilon = 1e-6
    plot_colors = cm.viridis(np.linspace(0, 1, len(I_values)))
    
    # Subtractive
    for i, I in enumerate(I_values):
        subtractive = np.maximum(0, E - I)
        ax1.plot(E, subtractive, label=f'I = {I}', color=plot_colors[i], lw=2.5)
        ax1.fill_between(E, 0, subtractive, color=plot_colors[i], alpha=0.1)
    ax1.set_title('Subtractive Inhibition\n$V = max(0, E - I)$', fontsize=16)
    ax1.set_xlabel('Excitation (E)', fontsize=14)
    ax1.set_ylabel('Output Voltage (V)', fontsize=14)
    ax1.legend(frameon=False, fontsize=12)
    ax1.grid(True, ls='--', alpha=0.3)
    
    # Shunting
    for i, I in enumerate(I_values):
        shunting = E / (E + I + epsilon)
        ax2.plot(E, shunting, label=f'I = {I}', color=plot_colors[i], lw=2.5)
        ax2.fill_between(E, 0, shunting, color=plot_colors[i], alpha=0.1)
    ax2.set_title('Shunting Inhibition\n$V = \\frac{E}{E + I + \\epsilon}$', fontsize=16)
    ax2.set_xlabel('Excitation (E)', fontsize=14)
    ax2.legend(frameon=False, fontsize=12)
    ax2.grid(True, ls='--', alpha=0.3)
    
    # Inset zoom
    axins = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
    E_zoom = np.linspace(0, 2, 100)
    for i, I in enumerate(I_values):
        axins.plot(E_zoom, E_zoom / (E_zoom + I + epsilon), color=plot_colors[i], lw=1.5)
    axins.set_title('Zoom Low E', fontsize=10)
    axins.grid(True, ls='--', alpha=0.3)
    
    fig.suptitle('Inhibition Mechanisms in Dendritic Computation', fontsize=18)
    
    save_figure(fig, 'shunting_inhibition')

def create_sparse_connectivity_visualization():
    """Create visualization of sparse connectivity variants in subplots."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=600)
    axs = axs.flatten()
    
    weights = np.random.randn(10,10)
    titles = ['Standard TopK', 'Stochastic TopK', 'Variance TopK', 'DeepST TopK']
    for idx, title in enumerate(titles):
        # Simulate different masks (simplified)
        mask = np.abs(weights) > np.percentile(np.abs(weights), 70 + idx*5)  # Vary for illustration
        sns.heatmap(weights * mask, ax=axs[idx], cmap='coolwarm', center=0, cbar=(idx%2==1), square=True, annot=False)
        axs[idx].set_title(title, fontsize=16)
    
    fig.suptitle('Sparse Connectivity Variants in Dendritic Layers', fontsize=18)
    
    save_figure(fig, 'sparse_connectivity')

def create_gradient_scaling_comparison():
    """Create comparison of gradient scaling strategies."""
    layers = np.arange(1, 11)
    vanilla = 1 / (2 ** layers)
    inverse = np.ones_like(layers) * 0.5
    conductance = 1 / np.sqrt(layers)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    ax.plot(layers, vanilla, color=colors['excitatory'], ls='--', label='Vanilla', lw=2.5)
    ax.plot(layers, inverse, color=colors['dendrite'], ls='-', label='Inverse Scaling', lw=2.5)
    ax.plot(layers, conductance, color=colors['inhibitory'], ls='-.', label='Conductance-based', lw=2.5)
    
    ax.set_title('Gradient Norms Across Network Layers', fontsize=18)
    ax.set_xlabel('Layer Depth', fontsize=14)
    ax.set_ylabel('Normalized Gradient Norm', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(frameon=False, fontsize=12)

    save_figure(fig, 'gradient_scaling')

def create_reactivation_functions_gallery():
    """Create gallery of reactivation functions with parameters."""
    x = np.linspace(-5, 5, 200)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=600, sharex=True, sharey=True)
    axs = axs.flatten()
    
    # Parametric Tanh
    for alpha in [0.5, 1.0, 2.0]:
        axs[0].plot(x, alpha * np.tanh(x / alpha), label=f'$\\alpha={alpha}$', lw=2.5)
    axs[0].set_title('Parametric Tanh', fontsize=16)
    axs[0].legend(frameon=False)
    
    # Param Linear Sigmoid
    for beta in [0.1, 1.0, 10.0]:
        axs[1].plot(x, (1 / (1 + np.exp(-beta * x))) * np.tanh(x), label=f'$\\beta={beta}$', lw=2.5)  # Minor improvement: changed * x to * tanh(x) for boundedness
    axs[1].set_title('Parametric Linear Sigmoid', fontsize=16)
    axs[1].legend(frameon=False)
    
    # Linear Tanh Transition
    for gamma in [0.2, 1.0, 5.0]:
        transition = gamma * np.tanh(x) + (1 - gamma) * (x / 5)
        axs[2].plot(x, transition, label=f'$\\gamma={gamma}$', lw=2.5)
    axs[2].set_title('Linear Tanh Transition', fontsize=16)
    axs[2].legend(frameon=False)
    
    # Standard
    axs[3].plot(x, np.tanh(x), label='Tanh', lw=2.5)
    axs[3].plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid', lw=2.5)
    axs[3].plot(x, np.maximum(0, x), label='ReLU', lw=2.5)
    axs[3].set_title('Standard Activations', fontsize=16)
    axs[3].legend(frameon=False)
    
    for ax in axs:
        ax.set_xlabel('Input', fontsize=14)
        ax.set_ylabel('Output', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Reactivation Functions to Prevent Branch Death', fontsize=18)
    
    save_figure(fig, 'reactivation_functions')

def create_ei_dynamics_flow():
    """Create enhanced E/I dynamics flow diagram with multiple layers."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=600)
    
    pos = {'Input': (-3, 1), 'E1': (-1, 2), 'I1': (-1, 0), 'E2': (1, 2), 'I2': (1, 0), 'E3': (3, 2), 'I3': (3, 0), 'Output': (5, 1)}
    
    # Nodes
    for node, (x, y) in pos.items():
        color = colors['input'] if node=='Input' else colors['output'] if node=='Output' else colors['excitatory'] if 'E' in node else colors['inhibitory']
        ax.add_patch(Ellipse((x, y), width=0.8, height=0.6, fc=color, ec=colors['text'], lw=2))
        ax.text(x, y, node, ha='center', va='center', weight='bold')
    
    # Connections with labels
    arrows = [
        ('Input', 'E1', 'Shared\n(Mode 0)'), ('Input', 'I1', 'Shared\n(Mode 0)'),
        ('I1', 'E1', 'IE (Within L1)'),
        ('E1', 'E2', 'EE'), ('E1', 'I2', 'EI'), ('I1', 'E2', 'IE (Indirect via I2)'), ('I1', 'I2', 'II'),
        ('I2', 'E2', 'IE (Within L2)'),
        ('E2', 'E3', 'EE'), ('E2', 'I3', 'EI'), ('I2', 'E3', 'IE (Indirect via I3)'), ('I2', 'I3', 'II'),
        ('I3', 'E3', 'IE (Within L3)'),
        ('E3', 'Output', 'E → Output')
    ]
    
    for start, end, label in arrows:
        sx, sy = pos[start]
        ex, ey = pos[end]
        color = colors['excitatory'] if start.startswith('E') or start == 'Input' else colors['inhibitory']
        ls = '--' if 'Indirect' in label else '-'
        arrow = FancyArrowPatch((sx + 0.5 if sx < ex else sx - 0.5, sy), (ex - 0.5 if sx < ex else ex + 0.5, ey),
                                color=color, lw=2, ls=ls, arrowstyle='->', mutation_scale=25)
        ax.add_patch(arrow)
        ax.text((sx + ex)/2, (sy + ey)/2 + 0.2, label, fontsize=12, color=color, ha='center', bbox=dict(fc='white', alpha=0.7, pad=0.1))
    
    ax.set_xlim(-4, 6)
    ax.set_ylim(-1, 3)
    ax.axis('off')
    ax.set_title('EINet E/I Hierarchical Connectivity', fontsize=18, pad=20)  # Improved title
    
    # Note
    ax.text(1, -0.8, 'Note: Dale\'s Principle enforced (E positive, I negative). IE within layers, indirect between.', 
            ha='center', fontsize=12, bbox=dict(fc='white', ec=colors['text'], alpha=0.8))
    
    save_figure(fig, 'ei_dynamics_updated')

def create_information_flow_diagram():
    """Create enhanced information flow diagram (Sankey-like)."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=600)  # Adjusted size for better aspect
    
    nodes = ['Input', 'TransferLayer', 'Branch1', 'Branch2', 'Soma Integration', 'Output']
    pos = {n: (i * 2, 0) for i, n in enumerate(nodes)}
    
    flows = [
        ('Input', 'TransferLayer', 3, colors['input']),
        ('TransferLayer', 'Branch1', 2, colors['excitatory']),
        ('TransferLayer', 'Branch2', 2, colors['inhibitory']),
        ('Branch1', 'Soma Integration', 1.5, colors['dendrite']),
        ('Branch2', 'Soma Integration', 1.5, colors['dendrite']),
        ('Soma Integration', 'Output', 2.5, colors['output'])
    ]
    
    for start, end, width, color in flows:
        sx, sy = pos[start]
        ex, ey = pos[end]
        ax.plot([sx, ex], [sy + width/20, ey + width/20], color=color, lw=width*2, alpha=0.6)
        ax.plot([sx, ex], [sy - width/20, ey - width/20], color=color, lw=width*2, alpha=0.6)
    
    for node, (x, y) in pos.items():
        ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, fc=colors['background'], ec=colors['text'], lw=2))
        ax.text(x, y, node.replace(' ', '\n'), ha='center', va='center', fontsize=12, weight='bold')
    
    ax.set_title('Information Flow in Dendritic Network', fontsize=18)
    ax.axis('off')
    
    save_figure(fig, 'information_flow')

def create_training_strategies_comparison():
    """Create comparison of training strategies with simulated data."""
    epochs = np.arange(1, 11)
    standard = 1 / np.log(epochs + 1) + np.random.normal(0, 0.02, len(epochs))
    branchwise = 1 / np.log(epochs + 2) + np.random.normal(0, 0.02, len(epochs))
    twostep = 1 / np.log(epochs + 1.5) + np.random.normal(0, 0.02, len(epochs))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    ax.plot(epochs, standard, color=colors['excitatory'], marker='o', ls='-', label='Standard', lw=2.5)
    ax.plot(epochs, branchwise, color=colors['dendrite'], marker='s', ls='--', label='Branch-wise', lw=2.5)
    ax.plot(epochs, twostep, color=colors['inhibitory'], marker='^', ls='-.', label='Two-step', lw=2.5)
    
    ax.set_title('Training Loss Curves by Strategy', fontsize=18)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss (Simulated)', fontsize=14)
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(frameon=False, fontsize=12)

    save_figure(fig, 'training_strategies')

def main():
    print("Generating dendritic neuron figures...")
    create_dendritic_neuron_labeled()
    create_dendritic_neuron_schematic()

    print("Generating DendriNet hierarchy...")
    create_dendrinet_hierarchy()

    print("Generating TransferLayer modes...")
    create_transfer_layer_modes()

    print("Generating block linear figure...")
    create_accurate_block_linear()

    print("Generating network architecture figure...")
    create_network_architecture()
    
    print("Generating shunting inhibition plot...")
    create_shunting_inhibition_plot()
    
    print("Generating additional figures...")
    create_sparse_connectivity_visualization()
    create_gradient_scaling_comparison()
    create_reactivation_functions_gallery()
    create_ei_dynamics_flow()
    create_information_flow_diagram()
    create_training_strategies_comparison()
    
    print("All figures generated successfully!")

if __name__ == '__main__':
    main()