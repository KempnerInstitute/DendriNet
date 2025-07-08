"""
Generate clean, accurate schematic diagrams for dendritic modeling documentation.

This script creates simplified, clear architectural diagrams that accurately
represent the codebase implementation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon
import matplotlib.lines as mlines

# Set clean style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['xtick.major.size'] = 0
plt.rcParams['ytick.major.size'] = 0

# Color scheme
COLORS = {
    'excitatory': '#e74c3c',
    'inhibitory': '#3498db', 
    'soma': '#2c3e50',
    'dendrite': '#9b59b6',
    'background': '#ecf0f1',
    'text': '#2c3e50',
    'arrow': '#34495e'
}

# Output directory
OUTPUT_DIR = "../docs/figures/architecture"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_basic_dendritic_neuron():
    """Create a clean, simple dendritic neuron schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    # Soma (larger, centered)
    soma = Circle((4, 2), 0.8, color=COLORS['soma'], zorder=10)
    ax.add_patch(soma)
    ax.text(4, 2, 'Soma', ha='center', va='center', color='white', 
            fontsize=14, weight='bold')
    
    # Primary branches (directly from soma)
    primary_branches = [
        (2.5, 4, 0),
        (5.5, 4, 1)
    ]
    
    # Draw primary branches
    for x, y, idx in primary_branches:
        # Branch circle
        branch = Circle((x, y), 0.5, color=COLORS['dendrite'], zorder=5)
        ax.add_patch(branch)
        ax.text(x, y, f'B{idx}', ha='center', va='center', color='white', 
                fontsize=10, weight='bold')
        
        # Connection from soma
        ax.plot([4, x], [2.8, y-0.5], 'k-', linewidth=3, zorder=4)
        
        # Synaptic inputs on branches
        # Excitatory inputs (top)
        for i in range(2):
            syn_x = x + (i - 0.5) * 0.6
            syn_y = y + 0.8
            exc = plt.Polygon([(syn_x, syn_y), (syn_x-0.15, syn_y+0.25), 
                              (syn_x+0.15, syn_y+0.25)], 
                             color=COLORS['excitatory'], zorder=6)
            ax.add_patch(exc)
            # Connection line
            ax.plot([syn_x, x], [syn_y, y+0.5], 'k-', linewidth=1, alpha=0.5)
        
        # Inhibitory inputs (sides)
        for i in range(2):
            syn_x = x + (i - 0.5) * 1.2
            syn_y = y
            inh = Circle((syn_x, syn_y), 0.12, color=COLORS['inhibitory'], zorder=6)
            ax.add_patch(inh)
            # Connection line
            ax.plot([syn_x, x + (i-0.5)*0.5], [syn_y, y], 'k-', linewidth=1, alpha=0.5)
    
    # Output arrow
    arrow = FancyArrowPatch((4, 1.2), (4, 0.2), 
                           arrowstyle='->', mutation_scale=25, 
                           lw=3, color=COLORS['arrow'])
    ax.add_patch(arrow)
    ax.text(4, -0.2, 'Output', ha='center', fontsize=12, weight='bold')
    
    # Legend
    exc_patch = patches.Patch(color=COLORS['excitatory'], label='Excitatory Input')
    inh_patch = patches.Patch(color=COLORS['inhibitory'], label='Inhibitory Input')
    ax.legend(handles=[exc_patch, inh_patch], loc='upper right', frameon=True,
              fancybox=True, shadow=True)
    
    # Title
    ax.text(4, 6, 'Dendritic Neuron Architecture', ha='center', 
            fontsize=18, weight='bold')
    
    # Mathematical formula
    ax.text(4, 5.2, r'$V_{out} = \frac{E}{E + I + \epsilon}$', 
            ha='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray'))
    
    ax.set_xlim(0, 8)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'basic_dendritic_neuron.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_ei_input_transformation():
    """Create clean E/I input transformation pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Components with better spacing
    components = [
        ('Input\nImage', 1, 3, 1.5, 1.5, COLORS['background']),
        ('Input\nNetwork', 4, 3, 2, 1.5, COLORS['background']),
        ('E', 8, 4.5, 1, 1, COLORS['excitatory']),
        ('I', 8, 1.5, 1, 1, COLORS['inhibitory']),
        ('Dendritic\nNeuron', 11.5, 3, 2, 2, COLORS['dendrite'])
    ]
    
    # Draw components
    for label, x, y, w, h, color in components:
        if label in ['E', 'I']:
            # Circular for E/I
            circle = Circle((x, y), w/2, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', 
                   color='white', fontsize=16, weight='bold')
        else:
            # Rectangular boxes
            box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='black', 
                                linewidth=2)
            ax.add_patch(box)
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=12, weight='bold')
    
    # Arrows with better routing
    arrows = [
        # Input to network
        ((1.75, 3), (3, 3), None),
        # Network to E
        ((5, 3.5), (7.5, 4.5), 'arc3,rad=0.2'),
        # Network to I  
        ((5, 2.5), (7.5, 1.5), 'arc3,rad=-0.2'),
        # E to dendrite
        ((8.5, 4.5), (10.5, 3.5), 'arc3,rad=-0.2'),
        # I to dendrite
        ((8.5, 1.5), (10.5, 2.5), 'arc3,rad=0.2'),
        # Output
        ((12.5, 3), (14, 3), None)
    ]
    
    for start, end, style in arrows:
        if style:
            arrow = FancyArrowPatch(start, end, 
                                   connectionstyle=style, 
                                   arrowstyle='->', 
                                   mutation_scale=20, 
                                   lw=2.5, 
                                   color=COLORS['arrow'])
        else:
            arrow = FancyArrowPatch(start, end, 
                                   arrowstyle='->', 
                                   mutation_scale=20, 
                                   lw=2.5, 
                                   color=COLORS['arrow'])
        ax.add_patch(arrow)
    
    # Labels
    ax.text(14.5, 3, 'Output', ha='left', fontsize=12, weight='bold')
    ax.text(6.5, 4.8, 'Excitatory\nPathway', ha='center', fontsize=10, 
            style='italic', color=COLORS['excitatory'])
    ax.text(6.5, 1.2, 'Inhibitory\nPathway', ha='center', fontsize=10, 
            style='italic', color=COLORS['inhibitory'])
    
    # Title
    ax.text(7.5, 6, 'Input Transformation Pipeline', ha='center', 
            fontsize=18, weight='bold')
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ei_input_transformation.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_multilayer_ei_network():
    """Create clean multi-layer E/I network diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Network structure from codebase
    layers = [
        {'name': 'Input', 'e_neurons': 4, 'i_neurons': 0, 'x': 2},
        {'name': 'Layer 1', 'e_neurons': 3, 'i_neurons': 2, 'x': 5},
        {'name': 'Layer 2', 'e_neurons': 2, 'i_neurons': 1, 'x': 8},
        {'name': 'Output', 'e_neurons': 2, 'i_neurons': 0, 'x': 11}
    ]
    
    # Draw layers
    neuron_positions = []
    
    for layer in layers:
        layer_neurons = []
        
        # Layer label
        ax.text(layer['x'], 7, layer['name'], ha='center', 
                fontsize=14, weight='bold')
        
        # E neurons
        e_y_start = 5.5 - (layer['e_neurons'] - 1) * 0.4
        for i in range(layer['e_neurons']):
            y = e_y_start + i * 0.8
            neuron = Circle((layer['x'], y), 0.3, 
                          color=COLORS['excitatory'], 
                          ec='black', linewidth=2, zorder=5)
            ax.add_patch(neuron)
            ax.text(layer['x'], y, 'E', ha='center', va='center', 
                   color='white', fontsize=10, weight='bold')
            layer_neurons.append(('E', layer['x'], y))
        
        # I neurons
        if layer['i_neurons'] > 0:
            i_y_start = 2.5 - (layer['i_neurons'] - 1) * 0.4
            for i in range(layer['i_neurons']):
                y = i_y_start + i * 0.8
                neuron = Circle((layer['x'], y), 0.25, 
                              color=COLORS['inhibitory'], 
                              ec='black', linewidth=2, zorder=5)
                ax.add_patch(neuron)
                ax.text(layer['x'], y, 'I', ha='center', va='center', 
                       color='white', fontsize=9, weight='bold')
                layer_neurons.append(('I', layer['x'], y))
        
        neuron_positions.append(layer_neurons)
    
    # Draw connections between layers (simplified)
    for i in range(len(neuron_positions) - 1):
        curr_layer = neuron_positions[i]
        next_layer = neuron_positions[i + 1]
        
        # Sample connections to avoid clutter
        for curr_neuron in curr_layer[::2]:  # Every other neuron
            for next_neuron in next_layer:
                if next_neuron[0] == 'E':  # Only to E neurons
                    # E to E: red, I to E: blue
                    color = COLORS['excitatory'] if curr_neuron[0] == 'E' else COLORS['inhibitory']
                    alpha = 0.3 if curr_neuron[0] == 'E' else 0.2
                    
                    ax.plot([curr_neuron[1], next_neuron[1]], 
                           [curr_neuron[2], next_neuron[2]], 
                           color=color, alpha=alpha, linewidth=1.5, zorder=1)
    
    # Input arrow
    arrow = FancyArrowPatch((0.5, 4), (1.5, 4), 
                           arrowstyle='->', mutation_scale=20, 
                           lw=2.5, color=COLORS['arrow'])
    ax.add_patch(arrow)
    ax.text(0.3, 4, 'Input', ha='right', fontsize=12)
    
    # Legend
    e_patch = patches.Patch(color=COLORS['excitatory'], label='Excitatory (E)')
    i_patch = patches.Patch(color=COLORS['inhibitory'], label='Inhibitory (I)')
    ax.legend(handles=[e_patch, i_patch], loc='lower center', 
              ncol=2, frameon=True, fancybox=True, shadow=True)
    
    # Title
    ax.text(6.5, 8, 'Multi-Layer E/I Network', ha='center', 
            fontsize=18, weight='bold')
    
    # Note
    ax.text(6.5, 0.5, 'E neurons implement: E/(E+I+ε)', ha='center', 
            fontsize=11, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['background']))
    
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multilayer_ei_network.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_shunting_inhibition_diagram():
    """Create clean shunting inhibition mechanism diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Mechanism diagram
    ax1.text(0.5, 0.95, 'Shunting Inhibition Mechanism', 
             ha='center', transform=ax1.transAxes, 
             fontsize=16, weight='bold')
    
    # Central dendrite
    dendrite = Circle((0.5, 0.5), 0.15, color=COLORS['dendrite'], 
                     transform=ax1.transAxes, zorder=5)
    ax1.add_patch(dendrite)
    ax1.text(0.5, 0.5, 'D', ha='center', va='center', color='white', 
             transform=ax1.transAxes, fontsize=14, weight='bold')
    
    # Excitatory input
    ax1.arrow(0.15, 0.7, 0.2, -0.1, head_width=0.05, head_length=0.03, 
              fc=COLORS['excitatory'], ec=COLORS['excitatory'], 
              transform=ax1.transAxes, linewidth=2)
    ax1.text(0.1, 0.75, 'E', ha='center', color=COLORS['excitatory'], 
             transform=ax1.transAxes, fontsize=14, weight='bold')
    
    # Inhibitory input
    ax1.arrow(0.15, 0.3, 0.2, 0.1, head_width=0.05, head_length=0.03, 
              fc=COLORS['inhibitory'], ec=COLORS['inhibitory'], 
              transform=ax1.transAxes, linewidth=2)
    ax1.text(0.1, 0.25, 'I', ha='center', color=COLORS['inhibitory'], 
             transform=ax1.transAxes, fontsize=14, weight='bold')
    
    # Output
    ax1.arrow(0.65, 0.5, 0.2, 0, head_width=0.05, head_length=0.03, 
              fc=COLORS['arrow'], ec=COLORS['arrow'], 
              transform=ax1.transAxes, linewidth=2)
    ax1.text(0.9, 0.5, 'Vout', ha='center', 
             transform=ax1.transAxes, fontsize=12)
    
    # Equation
    ax1.text(0.5, 0.15, r'$V_{out} = \frac{E}{E + I + \epsilon}$', 
             ha='center', transform=ax1.transAxes, fontsize=16,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                      edgecolor='gray'))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right: Response curves
    ax2.text(0.5, 0.95, 'Divisive Normalization Effect', 
             ha='center', transform=ax2.transAxes, 
             fontsize=16, weight='bold')
    
    E_values = np.linspace(0, 10, 100)
    I_values = [0, 1, 3, 10]
    epsilon = 0.1
    
    for I in I_values:
        V_out = E_values / (E_values + I + epsilon)
        ax2.plot(E_values, V_out, label=f'I = {I}', linewidth=2.5)
    
    ax2.set_xlabel('Excitatory Input (E)', fontsize=14)
    ax2.set_ylabel('Output (Vout)', fontsize=14)
    ax2.legend(title='Inhibitory Input', loc='lower right', 
               frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.1)
    
    # Add annotations
    ax2.annotate('Saturation', xy=(8, 0.1), xytext=(6, 0.3),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shunting_inhibition_diagram.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_network_architecture_overview():
    """Create simplified, clean network architecture overview."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Main components with better layout
    components = [
        ('Input', 2, 4, 1.5, 1.2),
        ('Input\nNetwork', 5, 4, 2, 1.5),
        ('E/I\nSeparation', 8.5, 4, 2, 1.5),
        ('Dendritic\nLayers', 12, 4, 2.5, 3),
        ('Output', 15.5, 4, 1.5, 1.2)
    ]
    
    # Draw main flow
    for i, (label, x, y, w, h) in enumerate(components):
        if i == 3:  # Dendritic layers - special treatment
            # Main box
            box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['dendrite'], 
                                edgecolor='black', linewidth=2,
                                alpha=0.3)
            ax.add_patch(box)
            
            # Sub-layers
            for j in range(3):
                sub_y = y + 1 - j
                sub_box = Rectangle((x-w/2+0.2, sub_y-0.3), w-0.4, 0.6,
                                   facecolor=COLORS['dendrite'], 
                                   edgecolor='black', linewidth=1)
                ax.add_patch(sub_box)
                ax.text(x, sub_y, f'Layer {j+1}', ha='center', va='center',
                       fontsize=10)
        else:
            color = COLORS['background'] if i < 3 else COLORS['soma']
            box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='black', 
                                linewidth=2)
            ax.add_patch(box)
        
        if i != 3:
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=12, weight='bold')
    
    # Arrows
    arrow_positions = [(3.25, 4), (6, 4), (9.5, 4), (13.25, 4), (14.75, 4)]
    for i in range(len(arrow_positions) - 1):
        arrow = FancyArrowPatch(arrow_positions[i], arrow_positions[i+1], 
                               arrowstyle='->', mutation_scale=20, 
                               lw=2.5, color=COLORS['arrow'])
        ax.add_patch(arrow)
    
    # E/I separation detail
    ax.plot([8.5, 8.5], [5.25, 6], 'k-', linewidth=2)
    ax.plot([8.5, 7.5], [6, 6.5], 'k-', linewidth=2)
    ax.plot([8.5, 9.5], [6, 6.5], 'k-', linewidth=2)
    
    exc_circle = Circle((7.5, 6.8), 0.3, color=COLORS['excitatory'], 
                       ec='black', linewidth=2)
    ax.add_patch(exc_circle)
    ax.text(7.5, 6.8, 'E', ha='center', va='center', color='white', 
           fontsize=10, weight='bold')
    
    inh_circle = Circle((9.5, 6.8), 0.3, color=COLORS['inhibitory'], 
                       ec='black', linewidth=2)
    ax.add_patch(inh_circle)
    ax.text(9.5, 6.8, 'I', ha='center', va='center', color='white', 
           fontsize=10, weight='bold')
    
    # Title
    ax.text(8.5, 8, 'Dendritic Network Architecture', ha='center', 
            fontsize=20, weight='bold')
    
    # Key equation
    ax.text(8.5, 0.5, r'Core Computation: $V_{out} = \frac{E}{E + I + \epsilon}$', 
            ha='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.8", facecolor='white', 
                     edgecolor='gray', linewidth=2))
    
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'network_architecture_overview.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_dendritic_computation_flow():
    """Create a flow diagram showing dendritic computation in detail."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Title
    ax.text(5, 9, 'Dendritic Computation Flow', ha='center', 
            fontsize=18, weight='bold')
    
    # Input stage
    ax.text(5, 7.5, 'Synaptic Inputs', ha='center', fontsize=14, weight='bold')
    
    # E inputs
    for i in range(3):
        x = 2 + i * 1.5
        triangle = plt.Polygon([(x, 6.5), (x-0.2, 6.8), (x+0.2, 6.8)], 
                              color=COLORS['excitatory'])
        ax.add_patch(triangle)
        ax.text(x, 6.2, f'E{i+1}', ha='center', fontsize=10)
    
    # I inputs
    for i in range(3):
        x = 2 + i * 1.5
        circle = Circle((x, 5.5), 0.15, color=COLORS['inhibitory'])
        ax.add_patch(circle)
        ax.text(x, 5.1, f'I{i+1}', ha='center', fontsize=10)
    
    # Summation
    ax.text(1, 4, 'Σ', fontsize=20, weight='bold', color=COLORS['excitatory'])
    ax.text(3.5, 4, 'Σ', fontsize=20, weight='bold', color=COLORS['inhibitory'])
    
    # Arrows to summation
    for i in range(3):
        x = 2 + i * 1.5
        ax.arrow(x, 6.3, -x+1.2, -1.8, head_width=0.1, head_length=0.1, 
                fc=COLORS['excitatory'], ec=COLORS['excitatory'], alpha=0.5)
        ax.arrow(x, 5.3, -x+3.7, -0.8, head_width=0.1, head_length=0.1, 
                fc=COLORS['inhibitory'], ec=COLORS['inhibitory'], alpha=0.5)
    
    # Division box
    div_box = FancyBboxPatch((4, 3), 2, 2, boxstyle="round,pad=0.1",
                            facecolor=COLORS['dendrite'], edgecolor='black', 
                            linewidth=2, alpha=0.5)
    ax.add_patch(div_box)
    ax.text(5, 4, '÷', fontsize=30, ha='center', va='center', weight='bold')
    
    # Arrows to division
    ax.arrow(1.5, 4, 2, 0.5, head_width=0.15, head_length=0.1, 
            fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(3.8, 4, 0.2, 0, head_width=0.15, head_length=0.1, 
            fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Output
    ax.arrow(5, 3, 0, -1, head_width=0.2, head_length=0.1, 
            fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    output_circle = Circle((5, 1.5), 0.4, color=COLORS['soma'], 
                          ec='black', linewidth=2)
    ax.add_patch(output_circle)
    ax.text(5, 1.5, 'Vout', ha='center', va='center', color='white', 
           fontsize=12, weight='bold')
    
    # Formula
    ax.text(8, 4, r'$V_{out} = \frac{\sum E_i}{\sum E_i + \sum I_j + \epsilon}$', 
            fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
                                  facecolor='white', edgecolor='gray'))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dendritic_computation_flow.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    print("Generating improved documentation figures...")
    
    # Create all figures
    create_basic_dendritic_neuron()
    print("✓ Basic dendritic neuron diagram created")
    
    create_ei_input_transformation()
    print("✓ E/I input transformation diagram created")
    
    create_multilayer_ei_network()
    print("✓ Multi-layer E/I network diagram created")
    
    create_shunting_inhibition_diagram()
    print("✓ Shunting inhibition diagram created")
    
    create_network_architecture_overview()
    print("✓ Network architecture overview created")
    
    create_dendritic_computation_flow()
    print("✓ Dendritic computation flow created")
    
    print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}") 