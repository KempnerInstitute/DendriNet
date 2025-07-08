"""
Generate schematic diagrams for dendritic modeling documentation.

This script creates various architectural diagrams showing:
- Basic dendritic neuron structure
- Excitatory and inhibitory inputs
- Multi-layer networks
- Branching patterns
- Input transformations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as mlines
from matplotlib import cm
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "../docs/figures/architecture"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_basic_dendritic_neuron():
    """Create a basic dendritic neuron schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Soma
    soma = Circle((5, 5), 1, color='#2c3e50', zorder=10)
    ax.add_patch(soma)
    ax.text(5, 5, 'Soma', ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    # Dendritic branches
    branch_positions = [
        (3, 7, 'Primary'),
        (7, 7, 'Primary'),
        (2, 9, 'Secondary'),
        (4, 9, 'Secondary'),
        (6, 9, 'Secondary'),
        (8, 9, 'Secondary')
    ]
    
    # Draw branches
    for i, (x, y, branch_type) in enumerate(branch_positions):
        color = '#3498db' if branch_type == 'Primary' else '#9b59b6'
        branch = Circle((x, y), 0.6, color=color, zorder=5)
        ax.add_patch(branch)
        
        # Connect to soma or parent branch
        if branch_type == 'Primary':
            line = mlines.Line2D([x, 5], [y-0.6, 6], color='black', linewidth=2)
        else:
            # Connect to nearest primary branch
            parent_x = 3 if x < 5 else 7
            line = mlines.Line2D([x, parent_x], [y-0.6, 7.6], color='black', linewidth=1.5)
        ax.add_line(line)
        
        # Add synaptic inputs
        n_synapses = 4 if branch_type == 'Primary' else 3
        for j in range(n_synapses):
            syn_x = x + (j - n_synapses/2) * 0.3
            syn_y = y + 1.2
            
            # Excitatory (triangles) and inhibitory (circles) synapses
            if j % 2 == 0:
                # Excitatory
                triangle = plt.Polygon([(syn_x, syn_y), (syn_x-0.1, syn_y+0.2), 
                                      (syn_x+0.1, syn_y+0.2)], color='#e74c3c')
                ax.add_patch(triangle)
            else:
                # Inhibitory
                inhib = Circle((syn_x, syn_y+0.1), 0.1, color='#3498db')
                ax.add_patch(inhib)
    
    # Labels
    ax.text(1, 11, 'Excitatory Synapses', color='#e74c3c', fontsize=10)
    ax.text(1, 10.5, 'Inhibitory Synapses', color='#3498db', fontsize=10)
    
    # Annotations
    ax.annotate('Primary Dendrites', xy=(3, 7), xytext=(0.5, 7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                fontsize=10, color='gray')
    
    ax.annotate('Secondary Dendrites', xy=(4, 9), xytext=(0.5, 9),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                fontsize=10, color='gray')
    
    # Output arrow
    arrow = FancyArrowPatch((5, 4), (5, 2), 
                           connectionstyle="arc3", 
                           arrowstyle='->', 
                           mutation_scale=20, 
                           lw=2, 
                           color='black')
    ax.add_patch(arrow)
    ax.text(5, 1.5, 'Output', ha='center', fontsize=10)
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Basic Dendritic Neuron Architecture', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'basic_dendritic_neuron.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_ei_input_transformation():
    """Create schematic showing E/I input transformation."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Input image
    img_box = FancyBboxPatch((0.5, 4), 2, 2, boxstyle="round,pad=0.1",
                             facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(img_box)
    ax.text(1.5, 5, 'Input\nImage', ha='center', va='center', fontsize=12)
    
    # Input network
    input_net_box = FancyBboxPatch((4, 3.5), 3, 3, boxstyle="round,pad=0.1",
                                  facecolor='#ecf0f1', edgecolor='black', linewidth=2)
    ax.add_patch(input_net_box)
    ax.text(5.5, 5, 'Input\nNetwork', ha='center', va='center', fontsize=12, weight='bold')
    
    # Excitatory pathway
    exc_box = FancyBboxPatch((9, 6), 2.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#ffe5e5', edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(exc_box)
    ax.text(10.25, 6.75, 'Excitatory\nTransform', ha='center', va='center', fontsize=10)
    
    # Inhibitory pathway
    inh_box = FancyBboxPatch((9, 2.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#e5f3ff', edgecolor='#3498db', linewidth=2)
    ax.add_patch(inh_box)
    ax.text(10.25, 3.25, 'Inhibitory\nTransform', ha='center', va='center', fontsize=10)
    
    # Dendritic computation
    dendrite_box = FancyBboxPatch((13, 3.5), 3, 3, boxstyle="round,pad=0.1",
                                 facecolor='#f0e5ff', edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(dendrite_box)
    ax.text(14.5, 5, 'Dendritic\nComputation\nE/(E+I+ε)', ha='center', va='center', fontsize=11)
    
    # Arrows
    arrows = [
        ((2.5, 5), (4, 5)),  # Input to network
        ((7, 5.5), (9, 6.5)),  # Network to excitatory
        ((7, 4.5), (9, 3.5)),  # Network to inhibitory
        ((11.5, 6.75), (13, 5.5)),  # Excitatory to dendrite
        ((11.5, 3.25), (13, 4.5)),  # Inhibitory to dendrite
        ((16, 5), (17.5, 5))  # Output
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, 
                               connectionstyle="arc3,rad=0.1", 
                               arrowstyle='->', 
                               mutation_scale=20, 
                               lw=2, 
                               color='black')
        ax.add_patch(arrow)
    
    # Labels
    ax.text(17.5, 5.3, 'Output', ha='left', fontsize=10)
    ax.text(8, 5.8, 'E', ha='center', fontsize=12, color='#e74c3c', weight='bold')
    ax.text(8, 4.2, 'I', ha='center', fontsize=12, color='#3498db', weight='bold')
    
    # Mathematical notation
    ax.text(14.5, 2, r'$V_{out} = \frac{E}{E + I + \epsilon}$', 
            ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
    
    ax.set_xlim(0, 18)
    ax.set_ylim(1, 8)
    ax.axis('off')
    ax.set_title('Excitatory/Inhibitory Input Transformation Pipeline', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ei_input_transformation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_multilayer_ei_network():
    """Create schematic of multi-layer E/I network."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Layer positions
    layer_x = [2, 6, 10, 14]
    n_neurons = [4, 6, 4, 2]  # Neurons per layer
    
    # Draw layers
    for layer_idx, (x, n) in enumerate(zip(layer_x, n_neurons)):
        layer_label = f'Layer {layer_idx}' if layer_idx < len(layer_x)-1 else 'Output'
        ax.text(x, 9, layer_label, ha='center', fontsize=12, weight='bold')
        
        # Draw E and I populations
        e_neurons = []
        i_neurons = []
        
        for i in range(n):
            y_e = 7 - i * 1.5
            y_i = 6.5 - i * 1.5
            
            # Excitatory neuron
            e_color = '#e74c3c' if layer_idx < len(layer_x)-1 else '#2c3e50'
            e_neuron = Circle((x, y_e), 0.4, color=e_color, zorder=5)
            ax.add_patch(e_neuron)
            ax.text(x, y_e, 'E', ha='center', va='center', color='white', fontsize=8, weight='bold')
            e_neurons.append((x, y_e))
            
            # Inhibitory neuron (not in output layer)
            if layer_idx < len(layer_x)-1:
                i_neuron = Circle((x + 0.8, y_i), 0.3, color='#3498db', zorder=5)
                ax.add_patch(i_neuron)
                ax.text(x + 0.8, y_i, 'I', ha='center', va='center', color='white', fontsize=8, weight='bold')
                i_neurons.append((x + 0.8, y_i))
        
        # Store neuron positions for connections
        if layer_idx == 0:
            prev_e = e_neurons
            prev_i = i_neurons
        elif layer_idx < len(layer_x):
            # Draw connections
            for e1, i1 in zip(prev_e, prev_i):
                for e2 in e_neurons[:len(e_neurons)//2]:  # Partial connectivity
                    # E to E connections
                    line = mlines.Line2D([e1[0], e2[0]], [e1[1], e2[1]], 
                                       color='#e74c3c', alpha=0.3, linewidth=1)
                    ax.add_line(line)
                    
                    # I to E connections
                    if i_neurons:  # Check if there are inhibitory neurons
                        line = mlines.Line2D([i1[0], e2[0]], [i1[1], e2[1]], 
                                           color='#3498db', alpha=0.3, linewidth=1)
                        ax.add_line(line)
            
            prev_e = e_neurons
            prev_i = i_neurons
    
    # Input arrow
    arrow = FancyArrowPatch((0, 5), (1.5, 5), 
                           arrowstyle='->', 
                           mutation_scale=20, 
                           lw=2, 
                           color='black')
    ax.add_patch(arrow)
    ax.text(-0.5, 5, 'Input', ha='right', fontsize=10)
    
    # Legend
    e_patch = patches.Patch(color='#e74c3c', label='Excitatory Neurons')
    i_patch = patches.Patch(color='#3498db', label='Inhibitory Neurons')
    ax.legend(handles=[e_patch, i_patch], loc='upper right')
    
    # Annotations
    ax.text(7, 1, 'Each E neuron receives both E and I inputs\nImplementing divisive normalization', 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1'))
    
    ax.set_xlim(-1, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Multi-Layer Excitatory-Inhibitory Network Architecture', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multilayer_ei_network.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_dendritic_branching_patterns():
    """Create schematic showing different branching patterns."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    patterns = [
        ('Uniform Branching', 'uniform'),
        ('Exponential Decay', 'exponential'),
        ('Custom Pattern', 'custom')
    ]
    
    for ax, (title, pattern) in zip(axes, patterns):
        # Soma
        soma = Circle((2, 1), 0.5, color='#2c3e50', zorder=10)
        ax.add_patch(soma)
        
        if pattern == 'uniform':
            # Uniform branching - equal at each level
            levels = [(2, 2.5, 1, 2), (1.5, 4, 0.8, 2), (2.5, 4, 0.8, 2),
                     (1, 5.5, 0.6, 2), (2, 5.5, 0.6, 2), (3, 5.5, 0.6, 2)]
        elif pattern == 'exponential':
            # Exponential - decreasing branches
            levels = [(2, 2.5, 1, 1), (1.5, 4, 0.8, 2), (2.5, 4, 0.8, 2),
                     (1.2, 5.5, 0.6, 4), (1.8, 5.5, 0.6, 4), (2.2, 5.5, 0.6, 4), (2.8, 5.5, 0.6, 4)]
        else:
            # Custom pattern
            levels = [(2, 2.5, 1, 1), (1, 4, 0.9, 3), (3, 4, 0.7, 2),
                     (0.5, 5.5, 0.5, 2), (1.5, 5.5, 0.6, 3), (3.5, 5.5, 0.5, 1)]
        
        # Draw branches
        for x, y, size, n_inputs in levels:
            branch = Circle((x, y), size*0.4, color='#9b59b6', alpha=0.7, zorder=5)
            ax.add_patch(branch)
            
            # Connect to parent
            if y == 2.5:
                line = mlines.Line2D([x, 2], [y-size*0.4, 1.5], color='black', linewidth=2)
            else:
                # Find nearest parent
                parent_y = 2.5 if y < 5 else 4
                parent_x = x
                line = mlines.Line2D([x, parent_x], [y-size*0.4, parent_y+0.3], color='black', linewidth=1.5)
            ax.add_line(line)
            
            # Add input indicators
            for i in range(n_inputs):
                input_x = x + (i - n_inputs/2) * 0.2
                input_y = y + size*0.5
                ax.plot(input_x, input_y, 'r^', markersize=4)
        
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12)
    
    fig.suptitle('Dendritic Branching Patterns', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dendritic_branching_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_shunting_inhibition_diagram():
    """Create diagram explaining shunting inhibition mechanism."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Circuit diagram
    ax1.text(0.5, 0.9, 'Shunting Inhibition Circuit', ha='center', transform=ax1.transAxes, 
             fontsize=14, weight='bold')
    
    # Dendrite
    dendrite = Rectangle((2, 3), 4, 2, facecolor='#f0e5ff', edgecolor='#9b59b6', linewidth=2)
    ax1.add_patch(dendrite)
    ax1.text(4, 4, 'Dendrite', ha='center', va='center', fontsize=12)
    
    # Excitatory input
    e_arrow = FancyArrowPatch((1, 5.5), (3, 5), 
                             arrowstyle='->', mutation_scale=20, 
                             lw=2, color='#e74c3c')
    ax1.add_patch(e_arrow)
    ax1.text(0.5, 5.5, 'E', ha='center', fontsize=12, color='#e74c3c', weight='bold')
    
    # Inhibitory input
    i_arrow = FancyArrowPatch((1, 2.5), (3, 3), 
                             arrowstyle='->', mutation_scale=20, 
                             lw=2, color='#3498db')
    ax1.add_patch(i_arrow)
    ax1.text(0.5, 2.5, 'I', ha='center', fontsize=12, color='#3498db', weight='bold')
    
    # Output
    out_arrow = FancyArrowPatch((6, 4), (7.5, 4), 
                               arrowstyle='->', mutation_scale=20, 
                               lw=2, color='black')
    ax1.add_patch(out_arrow)
    ax1.text(8, 4, 'Vout', ha='center', fontsize=12)
    
    # Equation
    ax1.text(4, 1.5, r'$V_{out} = \frac{E}{E + I + \epsilon}$', 
            ha='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black'))
    
    ax1.set_xlim(0, 9)
    ax1.set_ylim(0, 7)
    ax1.axis('off')
    
    # Right: Response curves
    ax2.text(0.5, 0.95, 'Divisive Normalization Effect', ha='center', transform=ax2.transAxes, 
             fontsize=14, weight='bold')
    
    E_values = np.linspace(0, 10, 100)
    I_values = [0, 2, 5, 10]
    epsilon = 0.1
    
    for I in I_values:
        V_out = E_values / (E_values + I + epsilon)
        ax2.plot(E_values, V_out, label=f'I = {I}', linewidth=2)
    
    ax2.set_xlabel('Excitatory Input (E)', fontsize=12)
    ax2.set_ylabel('Output (Vout)', fontsize=12)
    ax2.legend(title='Inhibitory Input', loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shunting_inhibition_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_network_architecture_overview():
    """Create comprehensive network architecture overview."""
    fig = plt.figure(figsize=(16, 10))
    
    # Main architecture flow
    ax = fig.add_subplot(111)
    
    # Components
    components = [
        ('Input\nImage', 1, 5, 1.5, 1, 'lightgray'),
        ('Input\nNetwork', 3.5, 5, 2, 1.5, '#ecf0f1'),
        ('E/I\nSeparation', 6.5, 5, 2, 1.5, '#ffe5e5'),
        ('Dendritic\nLayer 1', 10, 7, 2.5, 1.5, '#f0e5ff'),
        ('Dendritic\nLayer 2', 10, 3, 2.5, 1.5, '#f0e5ff'),
        ('Output\nLayer', 14, 5, 2, 1.5, '#e5f3ff')
    ]
    
    for label, x, y, w, h, color in components:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, weight='bold')
    
    # Connections
    connections = [
        ((2.5, 5), (2.5, 5)),
        ((5.5, 5), (5.5, 5)),
        ((8.5, 5.75), (8.5, 7)),
        ((8.5, 4.25), (8.5, 3)),
        ((12.5, 7), (13, 6)),
        ((12.5, 3), (13, 4)),
        ((16, 5), (17, 5))
    ]
    
    for i, (start, end) in enumerate(connections):
        if i < len(connections) - 1:
            arrow = FancyArrowPatch(start, end, 
                                   connectionstyle="arc3,rad=0.1", 
                                   arrowstyle='->', 
                                   mutation_scale=20, 
                                   lw=2, 
                                   color='black')
            ax.add_patch(arrow)
    
    # Detailed insets
    # Inset 1: Dendritic computation
    ax_inset1 = fig.add_axes([0.65, 0.65, 0.3, 0.25])
    ax_inset1.text(0.5, 0.9, 'Dendritic Computation', ha='center', transform=ax_inset1.transAxes,
                   fontsize=10, weight='bold')
    
    # Mini dendrite diagram
    dendrite = Circle((0.5, 0.5), 0.2, color='#9b59b6', transform=ax_inset1.transAxes)
    ax_inset1.add_patch(dendrite)
    ax_inset1.text(0.5, 0.5, 'D', ha='center', va='center', color='white', 
                   transform=ax_inset1.transAxes, fontsize=8, weight='bold')
    
    # Inputs
    ax_inset1.arrow(0.1, 0.7, 0.15, -0.1, head_width=0.05, head_length=0.05, 
                    fc='#e74c3c', ec='#e74c3c', transform=ax_inset1.transAxes)
    ax_inset1.text(0.05, 0.75, 'E', ha='center', color='#e74c3c', 
                   transform=ax_inset1.transAxes, fontsize=8, weight='bold')
    
    ax_inset1.arrow(0.1, 0.3, 0.15, 0.1, head_width=0.05, head_length=0.05, 
                    fc='#3498db', ec='#3498db', transform=ax_inset1.transAxes)
    ax_inset1.text(0.05, 0.25, 'I', ha='center', color='#3498db', 
                   transform=ax_inset1.transAxes, fontsize=8, weight='bold')
    
    ax_inset1.text(0.5, 0.1, r'$\frac{E}{E+I+\epsilon}$', ha='center', 
                   transform=ax_inset1.transAxes, fontsize=9)
    ax_inset1.axis('off')
    
    # Labels and annotations
    ax.text(17.5, 5, 'Output', ha='left', fontsize=10)
    
    # Title
    ax.text(0.5, 0.95, 'Dendritic Network Architecture Overview', 
            ha='center', transform=ax.transAxes, fontsize=18, weight='bold')
    
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'network_architecture_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating documentation figures...")
    
    # Create all figures
    create_basic_dendritic_neuron()
    print("✓ Basic dendritic neuron diagram created")
    
    create_ei_input_transformation()
    print("✓ E/I input transformation diagram created")
    
    create_multilayer_ei_network()
    print("✓ Multi-layer E/I network diagram created")
    
    create_dendritic_branching_patterns()
    print("✓ Dendritic branching patterns diagram created")
    
    create_shunting_inhibition_diagram()
    print("✓ Shunting inhibition diagram created")
    
    create_network_architecture_overview()
    print("✓ Network architecture overview created")
    
    print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}") 