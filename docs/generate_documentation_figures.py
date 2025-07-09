#!/usr/bin/env python3
"""
Enhanced Documentation Figure Generation Script

This script generates professional, scientifically accurate figures for the dendritic modeling documentation.
Based on careful analysis of the actual codebase architecture.

Scientific Accuracy:
- Dendrites = branches (same thing)
- Each dendrite receives excitatory inputs, inhibitory inputs, AND inputs from previous dendrites
- Dendrites process sequentially: dendrite_n -> dendrite_n-1 -> ... -> dendrite_1 -> soma
- In multilayer networks: excitatory neurons output to both E and I populations in next layer
- Inhibitory neurons in each layer modulate excitatory neurons in same layer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from styles import COLORS, FONT_SIZES, text
from helpers import arrow, neuron, professional_box, draw_synapse, draw_layer_box, save_figure

def create_scientifically_accurate_dendritic_neuron():
    """
    Create a scientifically accurate dendritic neuron diagram based on actual codebase.
    
    Key principles from codebase analysis:
    - DendriNet creates sequential dendritic layers: dendrite_n -> dendrite_n-1 -> ... -> soma
    - Each dendrite receives: excitatory input, inhibitory input, AND output from previous dendrite
    - Final dendrite outputs to soma
    - Shunting inhibition: E/(E+I+ε) computed at each dendrite
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Background
    ax.set_facecolor(COLORS['bg'])
    
    # Soma (final integration point)
    soma_pos = (6, 2)
    neuron(ax, soma_pos, kind='S', radius=0.4, label='Soma')
    
    # Dendritic layers (sequential processing)
    dendrite_positions = [
        (6, 4),   # Dendrite 1 (final)
        (6, 6),   # Dendrite 2
        (6, 8),   # Dendrite 3
    ]
    
    # Draw dendrites
    for i, pos in enumerate(dendrite_positions):
        # Create dendrite processing unit
        professional_box(ax, (pos[0]-0.5, pos[1]-0.3), 1.0, 0.6,
                        facecolor=COLORS['dendrite'], 
                        edgecolor=COLORS['border'],
                        label=f'D{len(dendrite_positions)-i}',
                        label_pos='center')
    
    # Sequential dendrite-to-dendrite connections
    for i in range(len(dendrite_positions)-1):
        arrow(ax, dendrite_positions[i+1], dendrite_positions[i], 
              color=COLORS['dendrite'], lw=2)
    
    # Final dendrite to soma
    arrow(ax, dendrite_positions[-1], soma_pos, 
          color=COLORS['dendrite'], lw=2)
    
    # Excitatory inputs to each dendrite
    exc_input_positions = [
        [(3, 8.5), (2.5, 8), (3.5, 7.5)],  # to dendrite 3
        [(3, 6.5), (2.5, 6), (3.5, 5.5)],  # to dendrite 2  
        [(3, 4.5), (2.5, 4), (3.5, 3.5)],  # to dendrite 1
    ]
    
    for dendrite_idx, exc_positions in enumerate(exc_input_positions):
        for exc_pos in exc_positions:
            draw_synapse(ax, exc_pos, synapse_type='E', size=0.15)
            arrow(ax, exc_pos, dendrite_positions[dendrite_idx], 
                  color=COLORS['excitatory'], lw=1, alpha=0.6)
    
    # Inhibitory inputs to each dendrite
    inh_input_positions = [
        [(9, 8.5), (9.5, 8), (8.5, 7.5)],  # to dendrite 3
        [(9, 6.5), (9.5, 6), (8.5, 5.5)],  # to dendrite 2
        [(9, 4.5), (9.5, 4), (8.5, 3.5)],  # to dendrite 1
    ]
    
    for dendrite_idx, inh_positions in enumerate(inh_input_positions):
        for inh_pos in inh_positions:
            draw_synapse(ax, inh_pos, synapse_type='I', size=0.15)
            arrow(ax, inh_pos, dendrite_positions[dendrite_idx], 
                  color=COLORS['inhibitory'], lw=1, alpha=0.6)
    
    # Shunting inhibition equations
    equation_y = 9.5
    text(ax, 6, equation_y, 'Shunting Inhibition at Each Dendrite:', 
         size='title', weight='bold', color=COLORS['text'], ha='center')
    
    text(ax, 6, equation_y - 0.4, r'$V_{d_i} = \frac{E_i + D_{i+1}}{E_i + D_{i+1} + I_i + \varepsilon}$', 
         size='body', color=COLORS['text'], ha='center')
    
    text(ax, 6, equation_y - 0.8, r'where $E_i$ = excitatory input, $I_i$ = inhibitory input, $D_{i+1}$ = previous dendrite output', 
         size='small', color=COLORS['text'], ha='center')
    
    # Legend
    legend_elements = [
        patches.Patch(color=COLORS['excitatory'], label='Excitatory Input'),
        patches.Patch(color=COLORS['inhibitory'], label='Inhibitory Input'),
        patches.Patch(color=COLORS['dendrite'], label='Dendritic Processing'),
        patches.Patch(color=COLORS['soma'], label='Soma (Final Integration)'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', frameon=False)
    
    # Title and formatting
    ax.set_title('Scientifically Accurate Dendritic Neuron\n(Sequential Dendritic Processing)', 
                fontsize=16, fontweight='bold', pad=20, color=COLORS['text'])
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    return fig

def create_multilayer_ei_network():
    """
    Create scientifically accurate multilayer E/I network diagram.
    
    Key principles from codebase:
    - Each layer has excitatory and inhibitory neuron populations
    - Excitatory neurons output to both E and I populations in next layer
    - Inhibitory neurons in each layer modulate excitatory neurons in same layer
    - Each neuron implements dendritic processing internally
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor(COLORS['bg'])
    
    # Layer setup
    layer_configs = [
        {'x': 2, 'label': 'Layer 1', 'exc_count': 3, 'inh_count': 2},
        {'x': 6, 'label': 'Layer 2', 'exc_count': 3, 'inh_count': 2},
        {'x': 10, 'label': 'Layer 3', 'exc_count': 2, 'inh_count': 0},  # Output layer
    ]
    
    # Draw layers and neurons
    layer_neurons = {}
    for layer_config in layer_configs:
        x = layer_config['x']
        
        # Layer background
        draw_layer_box(ax, x-0.8, 1, 1.6, 7, layer_config['label'])
        
        # Excitatory neurons
        exc_neurons = []
        for i in range(layer_config['exc_count']):
            y = 6 - i * 0.5
            neuron(ax, (x, y), kind='E', radius=0.15, label=f'E{i+1}')
            exc_neurons.append((x, y))
        
        # Inhibitory neurons
        inh_neurons = []
        for i in range(layer_config['inh_count']):
            y = 3 - i * 0.5
            neuron(ax, (x, y), kind='I', radius=0.15, label=f'I{i+1}')
            inh_neurons.append((x, y))
        
        layer_neurons[layer_config['label']] = {
            'excitatory': exc_neurons,
            'inhibitory': inh_neurons
        }
    
    # Inter-layer connections
    for i in range(len(layer_configs) - 1):
        current_layer = f"Layer {i+1}"
        next_layer = f"Layer {i+2}"
        
        # Excitatory to excitatory connections
        for exc_src in layer_neurons[current_layer]['excitatory']:
            for exc_dst in layer_neurons[next_layer]['excitatory']:
                arrow(ax, exc_src, exc_dst, color=COLORS['excitatory'], 
                      lw=1, alpha=0.4)
        
        # Excitatory to inhibitory connections
        if layer_neurons[next_layer]['inhibitory']:  # If next layer has inhibitory neurons
            for exc_src in layer_neurons[current_layer]['excitatory']:
                for inh_dst in layer_neurons[next_layer]['inhibitory']:
                    arrow(ax, exc_src, inh_dst, color=COLORS['excitatory'], 
                          lw=1, alpha=0.3)
    
    # Intra-layer inhibitory connections
    for layer_name, neurons in layer_neurons.items():
        if neurons['inhibitory']:  # If layer has inhibitory neurons
            for inh_src in neurons['inhibitory']:
                for exc_dst in neurons['excitatory']:
                    # Draw curved arrow for within-layer inhibition
                    arrow(ax, inh_src, exc_dst, style='arc', 
                          color=COLORS['inhibitory'], lw=1, alpha=0.5)
    
    # Input and output arrows
    arrow(ax, (0.5, 4), (layer_configs[0]['x']-0.5, 4), 
          color=COLORS['arrow'], lw=2)
    text(ax, 0.5, 4.3, 'Input', size='body', weight='bold', ha='center')
    
    arrow(ax, (layer_configs[-1]['x']+0.5, 4), (12.5, 4), 
          color=COLORS['arrow'], lw=2)
    text(ax, 12.5, 4.3, 'Output', size='body', weight='bold', ha='center')
    
    # Key principles annotation
    text(ax, 7, 8.5, 'Key E/I Network Principles:', 
         size='title', weight='bold', color=COLORS['text'], ha='center')
    
    principles = [
        '• Each layer has E and I neuron populations',
        '• E neurons → both E and I in next layer',
        '• I neurons → E neurons in same layer',
        '• Each neuron performs dendritic processing'
    ]
    
    for i, principle in enumerate(principles):
        text(ax, 7, 8.1 - i*0.3, principle, size='body', 
             color=COLORS['text'], ha='center')
    
    # Legend
    legend_elements = [
        patches.Patch(color=COLORS['excitatory'], label='Excitatory Neurons'),
        patches.Patch(color=COLORS['inhibitory'], label='Inhibitory Neurons'),
        patches.Patch(color=COLORS['arrow'], label='Data Flow'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', frameon=False)
    
    ax.set_title('Scientifically Accurate Multi-Layer E/I Network\n(Based on Actual Codebase Architecture)', 
                fontsize=16, fontweight='bold', pad=20, color=COLORS['text'])
    
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    return fig

def create_dendritic_computation_flow():
    """
    Create detailed dendritic computation flow diagram showing the actual DendriNet forward pass.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor(COLORS['bg'])
    
    # Processing steps (based on actual forward pass)
    steps = [
        {'pos': (1, 4), 'label': 'Excitatory\nInput', 'type': 'input'},
        {'pos': (1, 6), 'label': 'Inhibitory\nInput', 'type': 'input'},
        {'pos': (4, 6), 'label': 'Dendrite 3\nE/(E+I+ε)', 'type': 'dendrite'},
        {'pos': (7, 6), 'label': 'Dendrite 2\nE/(E+I+prev)', 'type': 'dendrite'},
        {'pos': (10, 6), 'label': 'Dendrite 1\nE/(E+I+prev)', 'type': 'dendrite'},
        {'pos': (13, 4), 'label': 'Soma\nOutput', 'type': 'soma'},
    ]
    
    # Draw processing steps
    for step in steps:
        if step['type'] == 'dendrite':
            professional_box(ax, (step['pos'][0]-0.8, step['pos'][1]-0.6), 1.6, 1.2,
                           facecolor=COLORS['dendrite'], 
                           edgecolor=COLORS['border'],
                           label=step['label'],
                           label_pos='center')
        elif step['type'] == 'soma':
            neuron(ax, step['pos'], kind='S', radius=0.4, label='Soma')
        else:  # input
            color = COLORS['excitatory'] if 'Excitatory' in step['label'] else COLORS['inhibitory']
            neuron(ax, step['pos'], kind='E' if 'Excitatory' in step['label'] else 'I', 
                  radius=0.3, label=step['label'])
    
    # Main flow arrows
    flow_connections = [
        ((1, 4), (4, 6)),   # Exc input to dendrite 3
        ((1, 6), (4, 6)),   # Inh input to dendrite 3
        ((4, 6), (7, 6)),   # Dendrite 3 to dendrite 2
        ((7, 6), (10, 6)),  # Dendrite 2 to dendrite 1
        ((10, 6), (13, 4)), # Dendrite 1 to soma
    ]
    
    for start, end in flow_connections:
        arrow(ax, start, end, color=COLORS['arrow'], lw=2)
    
    # Additional inputs to later dendrites (showing reuse)
    additional_connections = [
        ((1, 4), (7, 6)),   # Exc input to dendrite 2
        ((1, 6), (7, 6)),   # Inh input to dendrite 2
        ((1, 4), (10, 6)),  # Exc input to dendrite 1
        ((1, 6), (10, 6)),  # Inh input to dendrite 1
    ]
    
    for start, end in additional_connections:
        color = COLORS['excitatory'] if start[1] == 4 else COLORS['inhibitory']
        arrow(ax, start, end, color=color, lw=1, alpha=0.5)
    
    # Mathematical formulation
    text(ax, 7, 2, 'Sequential Dendritic Processing:', 
         size='title', weight='bold', color=COLORS['text'], ha='center')
    
    text(ax, 7, 1.5, r'$output = dendrite_1(x, inhib, dendrite_2(x, inhib, dendrite_3(x, inhib, None)))$', 
         size='body', color=COLORS['text'], ha='center')
    
    ax.set_title('Dendritic Computation Flow\n(DendriNet Forward Pass)', 
                fontsize=16, fontweight='bold', pad=20, color=COLORS['text'])
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    return fig

def main():
    """Generate all documentation figures with scientific accuracy."""
    print("Generating scientifically accurate documentation figures...")
    
    # Set matplotlib style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'text.usetex': False,
        'mathtext.fontset': 'dejavusans'
    })
    
    figures = [
        (create_scientifically_accurate_dendritic_neuron, "scientifically_accurate_dendritic_neuron"),
        (create_multilayer_ei_network, "scientifically_accurate_multilayer_ei_network"),
        (create_dendritic_computation_flow, "dendritic_computation_flow"),
    ]
    
    for fig_func, filename in figures:
        try:
            fig = fig_func()
            save_figure(fig, f"figures/architecture/{filename}")
            plt.close(fig)
        except Exception as e:
            print(f"Error generating {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll figures generated successfully!")
    print("Both PNG (300 DPI) and SVG formats available in figures/architecture/")

if __name__ == "__main__":
    main()