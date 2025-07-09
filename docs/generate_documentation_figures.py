#!/usr/bin/env python3
"""
Enhanced Documentation Figure Generator

This script generates professional, publication-quality figures for the 
Dendritic Modeling Framework documentation. It creates comprehensive 
visual representations of neural architectures, computational processes, 
and network configurations.

Key Features:
- Professional color palette with accessibility considerations
- Consistent typography and styling
- High-resolution output (300 DPI PNG + SVG)
- Comprehensive figure coverage
- Biological accuracy and visual clarity
- Modular design with reusable components
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon, Ellipse
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
from matplotlib.patches import ConnectionPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Import our helper modules
try:
    from .styles import (
        COLORS, FONT_SIZES, apply_professional_style, 
        create_shadow_effect, text
    )
    from .helpers import (
        arrow, neuron, legend_synapses, legend_neurons, professional_box,
        draw_dendrite_branch, draw_synapse, draw_layer_box, add_formula_box,
        save_figure, compute_layer_positions
    )
except ImportError:
    # If running as a script, use absolute imports
    from styles import (
        COLORS, FONT_SIZES, apply_professional_style, 
        create_shadow_effect, text
    )
    from helpers import (
        arrow, neuron, legend_synapses, legend_neurons, professional_box,
        draw_dendrite_branch, draw_synapse, draw_layer_box, add_formula_box,
        save_figure, compute_layer_positions
    )

# Apply professional styling
apply_professional_style()

# Output directory
OUTPUT_DIR = "../docs/figures/architecture"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================================
# Enhanced Core Architectural Diagrams
# =====================================================================================

def create_basic_dendritic_neuron(branching=(2, 3), syn_probs=None):
    """
    Create a clean dendritic neuron diagram with reduced clutter.
    
    Parameters:
    -----------
    branching : tuple
        (primary_branches, secondary_branches) counts
    syn_probs : dict or None
        Synapse probabilities {'E': 0.7, 'I': 0.3}
    """
    if syn_probs is None:
        syn_probs = {'E': 0.7, 'I': 0.3}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Central soma
    soma_x, soma_y = 0, 0
    soma_circle = neuron(ax, (soma_x, soma_y), kind='S', radius=0.5, 
                        label='Soma', shadow_preset='hard')
    
    # Nucleus
    nucleus = Circle((soma_x, soma_y), 0.2, 
                    facecolor='white', 
                    edgecolor='black', 
                    linewidth=1)
    ax.add_patch(nucleus)
    text(ax, soma_x, soma_y, 'N', size='small', weight='bold',
         ha='center', va='center')
    
    # Axon with simplified structure
    axon_start = (soma_x, soma_y - 0.5)
    axon_end = (soma_x, soma_y - 2.5)
    ax.plot([axon_start[0], axon_end[0]], [axon_start[1], axon_end[1]], 
           color=COLORS['axon'], linewidth=4, solid_capstyle='round')
    
    # Axon terminals
    terminal_positions = [(soma_x - 0.3, soma_y - 2.8), (soma_x + 0.3, soma_y - 2.8)]
    for tx, ty in terminal_positions:
        ax.plot([axon_end[0], tx], [axon_end[1], ty], 
               color=COLORS['axon'], linewidth=2.5, solid_capstyle='round')
        terminal = Circle((tx, ty), 0.12, 
                         facecolor=COLORS['synapse'], 
                         edgecolor='black', 
                         linewidth=1)
        ax.add_patch(terminal)
    
    text(ax, soma_x, soma_y - 3.5, 'Axon', size='body', weight='bold',
         ha='center', va='center')
    
    # Dendrites - only two hierarchical levels
    np.random.seed(42)
    dendrite_angles = np.linspace(30, 330, num=branching[0], endpoint=False)
    
    for angle in dendrite_angles:
        angle_rad = np.radians(angle)
        
        # Primary dendrite
        start_pos = (soma_x + 0.5 * np.cos(angle_rad), 
                    soma_y + 0.5 * np.sin(angle_rad))
        primary_length = 2.0
        
        dendrite_line, primary_end = draw_dendrite_branch(
            ax, start_pos, primary_length, angle, 
            linewidth=4, shadow_preset='soft'
        )
        
        # Secondary branches
        for i in range(branching[1]):
            branch_start_ratio = 0.5 + i * 0.3
            branch_start = (
                start_pos[0] + branch_start_ratio * primary_length * np.cos(angle_rad),
                start_pos[1] + branch_start_ratio * primary_length * np.sin(angle_rad)
            )
            
            # Branch angles
            branch_offsets = [-20, 20] if branching[1] == 2 else [-25, 0, 25]
            for offset in branch_offsets:
                branch_angle = angle + offset
                branch_line, branch_end = draw_dendrite_branch(
                    ax, branch_start, 1.2, branch_angle, 
                    linewidth=2.5, shadow_preset='soft'
                )
                
                # Add synapses with controlled probability
                if np.random.random() < 0.8:  # 80% chance of synapse
                    synapse_type = 'E' if np.random.random() < syn_probs['E'] else 'I'
                    draw_synapse(ax, branch_end, synapse_type=synapse_type, 
                               size=0.18, shadow_preset='soft')
    
    # Add zoomed-in synapse inset
    inset_x, inset_y = 6, 4
    inset_size = 1.5
    
    # Inset box
    inset_box = professional_box(
        ax, (inset_x - inset_size/2, inset_y - inset_size/2), 
        inset_size, inset_size,
        facecolor=COLORS['bg_light'], 
        edgecolor=COLORS['border'],
        shadow_preset='soft'
    )
    
    # Enlarged synapses in inset
    exc_synapse = draw_synapse(ax, (inset_x - 0.3, inset_y + 0.2), 
                              synapse_type='E', size=0.35, shadow_preset='none')
    inh_synapse = draw_synapse(ax, (inset_x + 0.3, inset_y - 0.2), 
                              synapse_type='I', size=0.35, shadow_preset='none')
    
    text(ax, inset_x, inset_y - 0.8, 'Synapse Detail', 
         size='small', weight='bold', ha='center', va='center')
    
    # Professional legend
    legend_synapses(ax, loc='upper left')
    
    # Add computational formula
    add_formula_box(ax, 0, -5, 
                   r'$V_{out} = \frac{\sum E_i}{\sum E_i + \sum I_j + \epsilon}$',
                   fontsize=FONT_SIZES['body'])
    
    ax.set_xlim(-7, 8)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    fig.suptitle('Dendritic Neuron Architecture', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    # Save with unified function
    save_figure(fig, f"{OUTPUT_DIR}/basic_dendritic_neuron")
    plt.close(fig)

def create_ei_input_transformation():
    """Create a cleaner E/I input transformation diagram."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Input layer
    input_neurons = [(2, 4), (2, 3), (2, 2)]
    for i, (x, y) in enumerate(input_neurons):
        neuron(ax, (x, y), kind='E', radius=0.25, 
              label=f'X{i+1}', shadow_preset='soft')
    
    # E/I split box
    split_box = professional_box(
        ax, (4, 2), 2, 3,
        facecolor=COLORS['processing'],
        edgecolor=COLORS['border'],
        label='E/I Split',
        shadow_preset='soft'
    )
    
    # Excitatory pathway
    exc_box = professional_box(
        ax, (7, 3.5), 1.5, 1,
        facecolor=COLORS['excitatory'],
        edgecolor=COLORS['border'],
        label='E',
        shadow_preset='soft'
    )
    
    # Inhibitory pathway
    inh_box = professional_box(
        ax, (7, 1.5), 1.5, 1,
        facecolor=COLORS['inhibitory'],
        edgecolor=COLORS['border'],
        label='I',
        shadow_preset='soft'
    )
    
    # Shunting computation
    shunt_box = professional_box(
        ax, (10, 2), 2, 3,
        facecolor=COLORS['bg_light'],
        edgecolor=COLORS['border'],
        label='Shunting\nInhibition',
        shadow_preset='soft'
    )
    
    # Output
    neuron(ax, (13, 3), kind='E', radius=0.3, 
          label='Out', shadow_preset='soft')
    
    # Arrows with grid-based routing
    for x, y in input_neurons:
        arrow(ax, (x + 0.25, y), (4, y), style='straight', 
              color=COLORS['arrow'], lw=2)
    
    # E/I pathways
    arrow(ax, (6, 4), (7, 4), style='straight', 
          color=COLORS['excitatory'], lw=2.5)
    arrow(ax, (6, 2), (7, 2), style='straight', 
          color=COLORS['inhibitory'], lw=2.5)
    
    # To shunting computation
    arrow(ax, (8.5, 4), (10, 4), style='straight', 
          color=COLORS['excitatory'], lw=2.5)
    arrow(ax, (8.5, 2), (10, 2), style='straight', 
          color=COLORS['inhibitory'], lw=2.5)
    
    # To output
    arrow(ax, (12, 3), (12.7, 3), style='straight', 
          color=COLORS['arrow'], lw=2.5)
    
    # Add formula
    add_formula_box(ax, 11, 0.5, 
                   r'$\frac{E}{E+I+\epsilon}$',
                   fontsize=FONT_SIZES['subtitle'])
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Centered title
    fig.suptitle('E/I Input Transformation', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.97)
    
    save_figure(fig, f"{OUTPUT_DIR}/ei_input_transformation")
    plt.close(fig)

def create_multilayer_ei_network():
    """Create a scalable multilayer E/I network diagram."""
    
    layer_counts = [4, 6, 3, 1]
    layout = compute_layer_positions(layer_counts, layer_sep=3.5, neuron_sep=0.8)
    
    fig, ax = plt.subplots(figsize=layout['figsize'])
    
    # Draw layers
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    for i, (layer_name, count) in enumerate(zip(layer_names, layer_counts)):
        layer_info = layout['layers'][f'layer_{i}']
        
        # Draw layer box
        layer_box = professional_box(
            ax, (layer_info['x'] - 0.5, layout['ylim'][0] + 0.5), 
            1, layout['ylim'][1] - layout['ylim'][0] - 1,
            facecolor=COLORS['bg_light'],
            edgecolor=COLORS['border'],
            label=layer_name,
            shadow_preset='soft'
        )
        
        # Draw neurons
        for j, (x, y) in enumerate(layer_info['positions']):
            # Determine neuron type (E/I ratio)
            if i == 0 or i == len(layer_counts) - 1:  # Input/Output layers
                kind = 'E'
            else:  # Hidden layers with E/I mix
                kind = 'E' if j < count * 0.8 else 'I'
            
            neuron(ax, (x, y), kind=kind, radius=0.2, shadow_preset='soft')
    
    # Draw connections between layers
    for i in range(len(layer_counts) - 1):
        curr_layer = layout['layers'][f'layer_{i}']
        next_layer = layout['layers'][f'layer_{i+1}']
        
        # Sample connections to avoid clutter
        for j, (x1, y1) in enumerate(curr_layer['positions']):
            for k, (x2, y2) in enumerate(next_layer['positions']):
                if np.random.random() < 0.3:  # 30% connection probability
                    # Connection color based on source neuron type
                    conn_color = COLORS['excitatory'] if j < len(curr_layer['positions']) * 0.8 else COLORS['inhibitory']
                    linestyle = '-' if j < len(curr_layer['positions']) * 0.8 else '--'
                    
                    ax.plot([x1, x2], [y1, y2], 
                           color=conn_color, linewidth=1, 
                           linestyle=linestyle, alpha=0.6)
    
    # Add within-layer inhibition annotation
    if len(layer_counts) > 2:
        layer_1 = layout['layers']['layer_1']
        x_center = layer_1['x']
        y_range = [pos[1] for pos in layer_1['positions']]
        y_top, y_bottom = max(y_range), min(y_range)
        
        # Curly brace for within-layer inhibition
        ax.annotate('', xy=(x_center + 0.3, y_bottom), 
                   xytext=(x_center + 0.3, y_top),
                   arrowprops=dict(arrowstyle='-', 
                                  connectionstyle='arc3,rad=0.2',
                                  color=COLORS['inhibitory'], 
                                  lw=2))
        
        text(ax, x_center + 0.6, (y_top + y_bottom) / 2, 'I→E', 
             size='small', weight='bold', ha='center', va='center')
    
    # Legend
    legend_neurons(ax, loc='upper right')
    
    ax.set_xlim(layout['xlim'])
    ax.set_ylim(layout['ylim'])
    ax.set_aspect('equal')
    ax.axis('off')
    
    fig.suptitle('Multi-layer E/I Network', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/multilayer_ei_network")
    plt.close(fig)

def create_shunting_inhibition_diagram():
    """Create a focused shunting inhibition diagram with formula placement."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Central soma
    soma_x, soma_y = 5, 3
    neuron(ax, (soma_x, soma_y), kind='S', radius=0.4, 
          label='Soma', shadow_preset='hard')
    
    # Two main dendrites
    dendrite_configs = [
        {'angle': 135, 'label': 'Dendrite 1', 'y_offset': 0.5},
        {'angle': 225, 'label': 'Dendrite 2', 'y_offset': -0.5}
    ]
    
    for config in dendrite_configs:
        angle_rad = np.radians(config['angle'])
        
        # Primary dendrite
        start_pos = (soma_x + 0.4 * np.cos(angle_rad), 
                    soma_y + 0.4 * np.sin(angle_rad))
        dendrite_line, end_pos = draw_dendrite_branch(
            ax, start_pos, 2.5, config['angle'], 
            linewidth=4, shadow_preset='soft'
        )
        
        # Add E and I inputs
        exc_pos = (start_pos[0] + 1.0 * np.cos(angle_rad), 
                  start_pos[1] + 1.0 * np.sin(angle_rad))
        inh_pos = (start_pos[0] + 1.8 * np.cos(angle_rad), 
                  start_pos[1] + 1.8 * np.sin(angle_rad))
        
        # Excitatory input
        arrow(ax, (exc_pos[0] - 0.5 * np.cos(angle_rad + np.pi/2), 
                  exc_pos[1] - 0.5 * np.sin(angle_rad + np.pi/2)),
              exc_pos, style='straight', color=COLORS['excitatory'], lw=3)
        text(ax, exc_pos[0] - 0.8 * np.cos(angle_rad + np.pi/2), 
             exc_pos[1] - 0.8 * np.sin(angle_rad + np.pi/2), 
             'E', size='subtitle', weight='bold', color=COLORS['excitatory'],
             ha='center', va='center')
        
        # Inhibitory input
        arrow(ax, (inh_pos[0] - 0.5 * np.cos(angle_rad + np.pi/2), 
                  inh_pos[1] - 0.5 * np.sin(angle_rad + np.pi/2)),
              inh_pos, style='straight', color=COLORS['inhibitory'], lw=3)
        text(ax, inh_pos[0] - 0.8 * np.cos(angle_rad + np.pi/2), 
             inh_pos[1] - 0.8 * np.sin(angle_rad + np.pi/2), 
             'I', size='subtitle', weight='bold', color=COLORS['inhibitory'],
             ha='center', va='center')
        
        # Dendrite label
        text(ax, end_pos[0], end_pos[1] + config['y_offset'], config['label'], 
             size='body', weight='bold', ha='center', va='center')
    
    # Single formula positioned centrally
    add_formula_box(ax, soma_x, soma_y - 1.5, 
                   r'$V_{out} = \frac{\sum E_i}{\sum E_i + \sum I_j + \epsilon}$',
                   fontsize=FONT_SIZES['subtitle'],
                   facecolor=COLORS['info'],
                   alpha=0.1)
    
    # Upstream dendrite arrow (lighter to not compete)
    arrow(ax, (soma_x + 1.5, soma_y), (soma_x + 0.4, soma_y), 
          style='straight', color=COLORS['text_light'], lw=2, 
          ms=15)
    text(ax, soma_x + 1.8, soma_y, 'Upstream', size='small', 
         color=COLORS['text_light'], ha='left', va='center')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    fig.suptitle('Shunting Inhibition Mechanism', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/shunting_inhibition_diagram")
    plt.close(fig)

# =====================================================================================
# Additional Enhanced Figures
# =====================================================================================

def create_network_architecture_overview():
    """Create a comprehensive network architecture overview (vertical layout)."""
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Vertical swim-lane layout (tiers)
    tiers = [
        {'name': 'Input Modalities', 'y': 8, 'color': COLORS['input_layer']},
        {'name': 'E/I Split & Pre-Net', 'y': 6, 'color': COLORS['processing']},
        {'name': 'Dendritic Core', 'y': 4, 'color': COLORS['dendrite']},
        {'name': 'Output Head(s)', 'y': 2, 'color': COLORS['output_layer']},
    ]
    
    for tier in tiers:
        professional_box(ax, (2, tier['y']), 4, 1.5,
                        facecolor=tier['color'],
                        edgecolor=COLORS['border'],
                        label=tier['name'],
                        shadow_preset='soft')
    
    # Vertical flow arrows
    for i in range(len(tiers) - 1):
        start_y = tiers[i]['y']
        end_y = tiers[i+1]['y'] + 1.5
        arrow(ax, (4, start_y), (4, end_y), 
              style='straight', color=COLORS['arrow'], lw=3)
    
    # Add side annotations
    annotations = [
        {'text': 'Raw data intake', 'x': 0.5, 'y': 8.75},
        {'text': 'Feature processing', 'x': 0.5, 'y': 6.75},
        {'text': 'Shunting inhibition', 'x': 0.5, 'y': 4.75},
        {'text': 'Classification', 'x': 0.5, 'y': 2.75},
    ]
    
    for ann in annotations:
        text(ax, ann['x'], ann['y'], ann['text'], 
             size='small', color=COLORS['text_secondary'],
             ha='left', va='center', style='italic')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 10)
    ax.axis('off')
    
    fig.suptitle('Network Architecture Overview', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/network_architecture_overview")
    plt.close(fig)

def create_dendritic_computation_flow():
    """Create a clean dendritic computation flow diagram."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Input stage
    text(ax, 1, 4, 'Inputs', size='subtitle', weight='bold', 
         ha='center', va='center')
    
    # E inputs
    for i in range(3):
        draw_synapse(ax, (1, 5 - i * 0.5), synapse_type='E', 
                    size=0.2, shadow_preset='soft')
        text(ax, 0.5, 5 - i * 0.5, f'E{i+1}', size='small', 
             color=COLORS['excitatory'], ha='center', va='center')
    
    # I inputs
    for i in range(2):
        draw_synapse(ax, (1, 2.5 - i * 0.5), synapse_type='I', 
                    size=0.2, shadow_preset='soft')
        text(ax, 0.5, 2.5 - i * 0.5, f'I{i+1}', size='small', 
             color=COLORS['inhibitory'], ha='center', va='center')
    
    # Summation nodes with color-matched styling
    exc_sum = professional_box(ax, (3, 4), 1.5, 1,
                              facecolor=COLORS['excitatory'],
                              edgecolor=COLORS['border'],
                              label='Σ E',
                              shadow_preset='soft')
    
    inh_sum = professional_box(ax, (3, 2), 1.5, 1,
                              facecolor=COLORS['inhibitory'],
                              edgecolor=COLORS['border'],
                              label='Σ I',
                              shadow_preset='soft')
    
    # Arrows to summation
    for i in range(3):
        arrow(ax, (1.2, 5 - i * 0.5), (3, 4.5), 
              style='arc', color=COLORS['excitatory'], lw=1.5)
    
    for i in range(2):
        arrow(ax, (1.2, 2.5 - i * 0.5), (3, 2.5), 
              style='arc', color=COLORS['inhibitory'], lw=1.5)
    
    # Shunting computation
    shunt_box = professional_box(ax, (6, 2.5), 2.5, 2,
                                facecolor=COLORS['bg_light'],
                                edgecolor=COLORS['border'],
                                label='Shunting\nDivision',
                                shadow_preset='soft')
    
    # Formula
    add_formula_box(ax, 7.25, 3.5, 
                   r'$\frac{E}{E+I+\epsilon}$',
                   fontsize=FONT_SIZES['subtitle'])
    
    # Arrows to shunting
    arrow(ax, (4.5, 4.5), (6, 4), 
          style='straight', color=COLORS['excitatory'], lw=2.5)
    arrow(ax, (4.5, 2.5), (6, 3), 
          style='straight', color=COLORS['inhibitory'], lw=2.5)
    
    # Output
    neuron(ax, (9.5, 3.5), kind='E', radius=0.3, 
          label='Out', shadow_preset='soft')
    
    arrow(ax, (8.5, 3.5), (9.2, 3.5), 
          style='straight', color=COLORS['arrow'], lw=2.5)
    
    ax.set_xlim(0, 10.5)
    ax.set_ylim(1, 6)
    ax.axis('off')
    
    # Export with transparent background for slides
    fig.suptitle('Dendritic Computation Flow', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/dendritic_computation_flow", 
                formats=['png', 'svg'])
    plt.close(fig)

def create_configuration_parameters_diagram():
    """Create a configuration parameters diagram with better organization."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Main configuration categories
    categories = [
        {
            'name': 'Network Architecture',
            'x': 1, 'y': 6, 'width': 4.5, 'height': 1.5,
            'params': [
                'input_dim: 784',
                'excitatory_layer_sizes: [100, 10]',
                'inhibitory_layer_sizes: [50, 0]',
                'branch_factors: [5, 1]'
            ]
        },
        {
            'name': 'Synaptic Connectivity',
            'x': 6.5, 'y': 6, 'width': 4.5, 'height': 1.5,
            'params': [
                'ee_synapses_per_branch: [10, 1]',
                'ei_synapses_per_branch: [5, 0]',
                'ie_synapses_per_branch: [2, 0]',
                'topk_type: "standard"'
            ]
        },
        {
            'name': 'Transfer Function',
            'x': 1, 'y': 4, 'width': 4.5, 'height': 1.5,
            'params': [
                'use_shunting: True',
                'input_mode: 0',
                'multiplicative: True',
                'reactivation_type: "param_tanh"'
            ]
        },
        {
            'name': 'Training Parameters',
            'x': 6.5, 'y': 4, 'width': 4.5, 'height': 1.5,
            'params': [
                'learning_rate: 0.001',
                'batch_size: 64',
                'epochs: 50',
                'gradient_scaling: True'
            ]
        },
    ]
    
    for cat in categories:
        # Create main box
        main_box = professional_box(
            ax, (cat['x'], cat['y']), cat['width'], cat['height'],
            facecolor=COLORS['bg_light'],
            edgecolor=COLORS['border'],
            label=cat['name'],
            shadow_preset='soft'
        )
        
        # Add parameters
        for i, param in enumerate(cat['params']):
            text(ax, cat['x'] + 0.2, cat['y'] + cat['height'] - 0.3 - i*0.25, 
                 param, size='small', color=COLORS['text_secondary'],
                 ha='left', va='center', family='monospace')
    
    # Add configuration flow
    text(ax, 6, 2.5, 'Configuration flows: YAML → Config Classes → Network',
         size='body', color=COLORS['text_secondary'],
         ha='center', va='center', style='italic')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(2, 8)
    ax.axis('off')
    
    fig.suptitle('Configuration Parameters', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/configuration_parameters_diagram")
    plt.close(fig)

def create_comprehensive_dendritic_neuron():
    """Create a comprehensive dendritic neuron with all details."""
    # This redirects to the cleaner version
    return create_basic_dendritic_neuron(branching=(6, 4), syn_probs={'E': 0.6, 'I': 0.4})

# =====================================================================================
# Legacy compatibility functions
# =====================================================================================

def create_einet_architecture():
    """Create EINet architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create pipeline boxes
    stages = [
        {'name': 'Input\nData', 'x': 1, 'y': 2, 'width': 1.5, 'height': 2},
        {'name': 'Input\nNetwork', 'x': 3.5, 'y': 2, 'width': 1.5, 'height': 2},
        {'name': 'E/I\nSeparation', 'x': 6, 'y': 2, 'width': 1.5, 'height': 2},
        {'name': 'Dendritic\nLayers', 'x': 8.5, 'y': 2, 'width': 1.5, 'height': 2},
        {'name': 'Output', 'x': 11, 'y': 2, 'width': 1.5, 'height': 2},
    ]
    
    for stage in stages:
        professional_box(ax, (stage['x'], stage['y']), stage['width'], stage['height'],
                        facecolor=COLORS['bg_light'],
                        edgecolor=COLORS['border'],
                        label=stage['name'],
                        shadow_preset='soft')
    
    # Connect with arrows
    for i in range(len(stages) - 1):
        start_x = stages[i]['x'] + stages[i]['width']
        end_x = stages[i+1]['x']
        arrow(ax, (start_x, 3), (end_x, 3), 
              style='straight', color=COLORS['arrow'], lw=2)
    
    # Add key formula
    add_formula_box(ax, 8.75, 0.8, 
                   r'$V = \frac{E}{E+I+\epsilon}$',
                   fontsize=FONT_SIZES['body'])
    
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    fig.suptitle('EINet Architecture', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/einet_architecture")
    plt.close(fig)

def create_einet_classifier_dataflow():
    """Create EINet classifier dataflow diagram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    stages = [
        ("Input\nImage", 1),
        ("Input\nNetwork", 3),
        ("E/I\nSeparation", 5),
        ("Dendritic\nLayers", 7),
        ("Classifier\nHead", 9),
        ("Prediction", 11),
    ]
    
    for label, x in stages:
        professional_box(ax, (x - 0.6, 1.5), 1.2, 1.2,
                        facecolor=COLORS['bg_light'],
                        edgecolor=COLORS['border'],
                        label=label,
                        shadow_preset='soft')
    
    # Connect with arrows
    for (label, x_start), (_, x_end) in zip(stages[:-1], stages[1:]):
        arrow(ax, (x_start + 0.6, 2.1), (x_end - 0.6, 2.1), 
              style='straight', color=COLORS['arrow'], lw=2)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 3.5)
    ax.axis('off')
    
    fig.suptitle('EINetClassifier Forward Data-flow', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/einet_classifier_dataflow")
    plt.close(fig)

def _draw_dendrinet(ax, center_x, layer_counts, layer_sep=2.5, cmap='viridis'):
    """Helper to draw dendrinet with parameterized spacing."""
    cmap = plt.get_cmap(cmap)
    xs = np.arange(len(layer_counts)) * layer_sep + center_x
    
    for idx, (n, x) in enumerate(zip(layer_counts, xs)):
        ys = np.linspace(-1, 1, n)
        color = cmap(idx / len(layer_counts))
        
        # Draw neurons
        for y in ys:
            neuron(ax, (x, y), kind='E', radius=0.15, 
                  shadow_preset='soft')
        
        # Draw connections
        if idx > 0:
            prev_n = layer_counts[idx - 1]
            prev_x = xs[idx - 1]
            prev_ys = np.linspace(-1, 1, prev_n)
            
            for y0 in prev_ys:
                for y1 in ys:
                    if np.random.random() < 0.3:  # 30% connection probability
                        ax.plot([prev_x, x], [y0, y1], 
                               color=COLORS['arrow'], alpha=0.3, lw=1)
    
    # Compute figure size automatically
    figwidth = (len(layer_counts) - 1) * layer_sep + 2
    figheight = 4
    
    return figwidth, figheight

def create_dendrinet_1():
    """Generate schematic for Dendrinet variant 1."""
    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_dendrinet(ax, center_x=0, layer_counts=[4, 6, 3])
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    fig.suptitle('Dendrinet-1 Architecture', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/dendrinet_1")
    plt.close(fig)

def create_dendrinet_2():
    """Generate schematic for Dendrinet variant 2."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _draw_dendrinet(ax, center_x=0, layer_counts=[6, 8, 6, 3])
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    
    fig.suptitle('Dendrinet-2 Architecture', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/dendrinet_2")
    plt.close(fig)

# Additional legacy functions for compatibility
def create_input_network_architecture():
    """Create input network architecture diagram."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    
    stages = ["Raw Input", "Pre-Net", "Feature Map"]
    
    for idx, label in enumerate(stages):
        x = 1 + idx * 2
        professional_box(ax, (x - 0.8, 1.25 - 0.4), 1.6, 0.8,
                        facecolor=COLORS['bg_light'],
                        edgecolor=COLORS['border'],
                        label=label,
                        shadow_preset='soft')
        
        if idx < len(stages) - 1:
            arrow(ax, (x + 0.8, 1.25), (x + 1.2, 1.25), 
                  style='straight', color=COLORS['arrow'], lw=2)
    
    ax.set_xlim(0, 7)
    ax.set_ylim(0.2, 2.3)
    ax.axis('off')
    
    fig.suptitle('Input Network Architecture', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/input_network_architecture")
    plt.close(fig)

def create_dendritic_processing():
    """Create dendritic processing diagram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Central neuron
    neuron(ax, (4, 3), kind='S', radius=0.5, label='Soma', shadow_preset='hard')
    
    # Multiple dendrites
    for i, angle in enumerate([45, 90, 135]):
        angle_rad = np.radians(angle)
        start_pos = (4 + 0.5 * np.cos(angle_rad), 3 + 0.5 * np.sin(angle_rad))
        
        # Draw dendrite
        dendrite_line, end_pos = draw_dendrite_branch(
            ax, start_pos, 2.5, angle, linewidth=3, shadow_preset='soft'
        )
        
        # Add synapses
        for j in range(3):
            synapse_pos = (start_pos[0] + (j + 1) * 0.7 * np.cos(angle_rad),
                          start_pos[1] + (j + 1) * 0.7 * np.sin(angle_rad))
            synapse_type = 'E' if j % 2 == 0 else 'I'
            draw_synapse(ax, synapse_pos, synapse_type=synapse_type, 
                        size=0.15, shadow_preset='soft')
    
    # Legend
    legend_synapses(ax, loc='upper right')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    fig.suptitle('Dendritic Processing', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    save_figure(fig, f"{OUTPUT_DIR}/dendritic_processing")
    plt.close(fig)

def create_dendritic_computation():
    """Create dendritic computation diagram."""
    # This redirects to the cleaner version
    return create_dendritic_computation_flow()

# =====================================================================================
# Main execution
# =====================================================================================

def main():
    """Main function to generate all documentation figures."""
    
    print("Generating enhanced professional documentation figures...")
    print("=" * 60)
    
    # Core architectural diagrams
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
    
    create_configuration_parameters_diagram()
    print("✓ Configuration parameters diagram created")
    
    create_comprehensive_dendritic_neuron()
    print("✓ Comprehensive dendritic neuron diagram created")
    
    # Legacy compatibility diagrams
    create_einet_architecture()
    print("✓ EINet architecture diagram created")
    
    create_einet_classifier_dataflow()
    print("✓ EINet classifier dataflow diagram created")
    
    create_dendrinet_1()
    print("✓ Dendrinet-1 architecture diagram created")
    
    create_dendrinet_2()
    print("✓ Dendrinet-2 architecture diagram created")
    
    create_input_network_architecture()
    print("✓ Input network architecture diagram created")
    
    create_dendritic_processing()
    print("✓ Dendritic processing diagram created")
    
    create_dendritic_computation()
    print("✓ Dendritic computation diagram created")
    
    print("=" * 60)
    print(f"✓ All enhanced figures saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"✓ Generated both PNG (300 DPI) and SVG formats")
    print(f"✓ Professional styling with consistent color palette applied")
    print(f"✓ Modular design with reusable components")
    print(f"✓ Reduced code duplication and improved maintainability")
    print("=" * 60)

if __name__ == "__main__":
    main()