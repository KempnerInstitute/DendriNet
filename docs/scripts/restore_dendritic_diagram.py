#!/usr/bin/env python3
"""
Script to regenerate the original dendritic diagram that was lost.
This recreates the professional dendritic neuron diagram from the deleted generate_documentation_figures.py file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.patheffects as pe

# Professional color palette (from the deleted styles.py)
COLORS = {
    # Primary neural component colors
    'excitatory': '#D55E00',    # Orange (excitatory synapses/neurons)
    'inhibitory': '#56B4E9',    # Sky blue (inhibitory synapses/neurons)
    'soma': '#000000',          # Black (cell body)
    'dendrite': '#009E73',      # Green (dendritic branches)
    'axon': '#CC79A7',          # Pink (axonal projections)
    'synapse': '#F0E442',       # Yellow (synaptic terminals)
    
    # Interface and flow colors
    'arrow': '#0072B2',         # Blue (arrows and connections)
    'text': '#000000',          # Black (primary text)
    'text_secondary': '#666666', # Dark gray (secondary text)
    'text_light': '#999999',    # Light gray (annotations)
    
    # Background and interface
    'bg': '#FFFFFF',            # White (background)
    'bg_light': '#F8F8F8',      # Very light gray (backgrounds)
    'bg_medium': '#E8E8E8',     # Light gray (panels)
    'border': '#CCCCCC',        # Light gray (borders)
    
    # Semantic colors
    'success': '#27AE60',       # Green (success states)
    'warning': '#F39C12',       # Orange (warnings)
    'error': '#E74C3C',         # Red (errors)
    'info': '#3498DB',          # Blue (information)
}

# Font sizes
FONT_SIZES = {
    'title': 16,
    'subtitle': 14,
    'body': 12,
    'small': 10,
    'tiny': 8,
}

def create_shadow_effect(alpha=0.3, offset=(2, -2)):
    """Create a shadow effect for text."""
    return [pe.withStroke(linewidth=3, foreground='white', alpha=alpha),
            pe.SimplePatchShadow(offset=offset, alpha=alpha)]

def text(ax, x, y, text_str, size='body', weight='normal', ha='left', va='bottom', color=None, **kwargs):
    """Add text with consistent styling."""
    if color is None:
        color = COLORS['text']
    
    fontsize = FONT_SIZES.get(size, 12)
    
    ax.text(x, y, text_str, fontsize=fontsize, weight=weight, 
            ha=ha, va=va, color=color, **kwargs)

def neuron(ax, center, kind='E', radius=0.3, label=None, shadow_preset='soft'):
    """Draw a neuron with professional styling."""
    x, y = center
    
    # Color based on neuron type
    if kind == 'E':
        color = COLORS['excitatory']
    elif kind == 'I':
        color = COLORS['inhibitory']
    elif kind == 'S':
        color = COLORS['soma']
    else:
        color = COLORS['soma']
    
    # Draw main circle
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    
    # Add label if provided
    if label:
        text(ax, x, y, label, size='small', weight='bold', ha='center', va='center', color='white')
    
    return circle

def draw_dendrite_branch(ax, start_pos, length, angle, linewidth=3, shadow_preset='soft'):
    """Draw a dendritic branch."""
    angle_rad = np.radians(angle)
    end_pos = (start_pos[0] + length * np.cos(angle_rad),
               start_pos[1] + length * np.sin(angle_rad))
    
    line = ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   color=COLORS['dendrite'], linewidth=linewidth, solid_capstyle='round')
    
    return line, end_pos

def draw_synapse(ax, pos, synapse_type='E', size=0.15, shadow_preset='soft'):
    """Draw a synapse."""
    x, y = pos
    
    if synapse_type == 'E':
        color = COLORS['excitatory']
    else:
        color = COLORS['inhibitory']
    
    circle = Circle((x, y), size, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(circle)
    
    return circle

def professional_box(ax, center, width, height, facecolor=None, edgecolor=None, 
                    label=None, shadow_preset='soft'):
    """Create a professional box with styling."""
    if facecolor is None:
        facecolor = COLORS['bg_light']
    if edgecolor is None:
        edgecolor = COLORS['border']
    
    x, y = center
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    
    if label:
        text(ax, x, y, label, size='body', weight='bold', ha='center', va='center')
    
    return box

def save_figure(fig, filename, dpi=300):
    """Save figure in both PNG and SVG formats."""
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{filename}.svg", bbox_inches='tight', facecolor='white')

def create_clean_dendritic_neuron(output_file="clean_dendritic_diagram"):
    """
    Create a professional, anatomically correct dendritic neuron diagram.
    Perfect for first page or minimal presentation.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Central soma - positioned in the center
    soma_x, soma_y = 0, 0
    soma_radius = 0.5
    
    # Draw soma with gradient effect
    soma_circle = Circle((soma_x, soma_y), soma_radius, 
                        facecolor='#333333', edgecolor='black', linewidth=2)
    ax.add_patch(soma_circle)
    
    # Nucleus
    nucleus = Circle((soma_x, soma_y), 0.2, 
                    facecolor='#666666', 
                    edgecolor='black', 
                    linewidth=1)
    ax.add_patch(nucleus)
    
    # Nucleolus
    nucleolus = Circle((soma_x + 0.05, soma_y + 0.05), 0.08, 
                      facecolor='#999999', 
                      edgecolor='black', 
                      linewidth=0.5)
    ax.add_patch(nucleolus)
    
    # APICAL DENDRITES (top) - Main dendritic tree
    # Main apical dendrite trunk
    apical_start = (soma_x, soma_y + soma_radius)
    apical_mid = (soma_x, soma_y + 2.0)
    apical_end = (soma_x, soma_y + 3.5)
    
    # Draw main apical trunk with tapering
    points = np.array([apical_start, apical_mid, apical_end])
    widths = [5, 4, 3]
    for i in range(len(points)-1):
        ax.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 
               color=COLORS['dendrite'], linewidth=widths[i], solid_capstyle='round')
    
    # Apical tuft - branching at the top
    tuft_angles = [-45, -22.5, 0, 22.5, 45]
    for angle in tuft_angles:
        angle_rad = np.radians(angle)
        start = apical_end
        length = 1.5
        end = (start[0] + length * np.sin(angle_rad), 
               start[1] + length * np.cos(angle_rad))
        
        # Primary tuft branch
        ax.plot([start[0], end[0]], [start[1], end[1]], 
               color=COLORS['dendrite'], linewidth=2.5, solid_capstyle='round')
        
        # Secondary tuft branches
        for sub_angle_offset in [-15, 15]:
            sub_angle = angle + sub_angle_offset
            sub_angle_rad = np.radians(sub_angle)
            sub_length = 0.8
            sub_end = (end[0] + sub_length * np.sin(sub_angle_rad),
                      end[1] + sub_length * np.cos(sub_angle_rad))
            
            ax.plot([end[0], sub_end[0]], [end[1], sub_end[1]], 
                   color=COLORS['dendrite'], linewidth=1.5, solid_capstyle='round')
            
            # Add synapses on tuft branches
            if np.random.random() < 0.7:
                synapse_type = 'E' if np.random.random() < 0.8 else 'I'
                draw_synapse(ax, sub_end, synapse_type=synapse_type, 
                           size=0.08, shadow_preset='soft')
    
    # Oblique dendrites (branching from apical trunk)
    oblique_positions = [0.3, 0.5, 0.7]
    for pos_ratio in oblique_positions:
        branch_point = (soma_x, soma_y + soma_radius + pos_ratio * 2.5)
        for side in [-1, 1]:
            angle = side * np.random.uniform(30, 60)
            angle_rad = np.radians(angle)
            length = np.random.uniform(0.8, 1.2)
            end = (branch_point[0] + length * np.sin(angle_rad),
                  branch_point[1] + length * np.cos(angle_rad))
            
            ax.plot([branch_point[0], end[0]], [branch_point[1], end[1]], 
                   color=COLORS['dendrite'], linewidth=2, solid_capstyle='round')
            
            # Add synapse
            if np.random.random() < 0.6:
                synapse_type = 'E' if np.random.random() < 0.75 else 'I'
                draw_synapse(ax, end, synapse_type=synapse_type, 
                           size=0.08, shadow_preset='soft')
    
    # BASAL DENDRITES (radiating from soma sides)
    basal_angles = [135, 165, 195, 225]  # Lower hemisphere angles
    for angle in basal_angles:
        angle_rad = np.radians(angle)
        
        # Primary basal dendrite
        start_pos = (soma_x + soma_radius * np.cos(angle_rad), 
                    soma_y + soma_radius * np.sin(angle_rad))
        primary_length = 1.5
        primary_end = (start_pos[0] + primary_length * np.cos(angle_rad),
                      start_pos[1] + primary_length * np.sin(angle_rad))
        
        ax.plot([start_pos[0], primary_end[0]], [start_pos[1], primary_end[1]], 
               color=COLORS['dendrite'], linewidth=3, solid_capstyle='round')
        
        # Secondary basal branches
        for branch_ratio in [0.5, 0.8]:
            branch_point = (start_pos[0] + branch_ratio * primary_length * np.cos(angle_rad),
                           start_pos[1] + branch_ratio * primary_length * np.sin(angle_rad))
            
            for offset in [-20, 20]:
                branch_angle = angle + offset
                branch_angle_rad = np.radians(branch_angle)
                branch_length = 0.7
                branch_end = (branch_point[0] + branch_length * np.cos(branch_angle_rad),
                             branch_point[1] + branch_length * np.sin(branch_angle_rad))
                
                ax.plot([branch_point[0], branch_end[0]], [branch_point[1], branch_end[1]], 
                       color=COLORS['dendrite'], linewidth=1.8, solid_capstyle='round')
                
                # Add synapses
                if np.random.random() < 0.7:
                    synapse_type = 'E' if np.random.random() < 0.7 else 'I'
                    draw_synapse(ax, branch_end, synapse_type=synapse_type, 
                               size=0.08, shadow_preset='soft')
    
    # AXON (bottom) - Anatomically correct
    # Axon hillock
    hillock_start = (soma_x, soma_y - soma_radius)
    hillock_end = (soma_x, soma_y - soma_radius - 0.3)
    
    # Draw axon hillock (cone shape)
    hillock_width = 0.25
    hillock = patches.Polygon([
        (hillock_start[0] - hillock_width, hillock_start[1]),
        (hillock_start[0] + hillock_width, hillock_start[1]),
        (hillock_end[0], hillock_end[1])
    ], facecolor=COLORS['axon'], edgecolor='black', linewidth=1)
    ax.add_patch(hillock)
    
    # Initial segment
    initial_segment_end = (soma_x, hillock_end[1] - 0.5)
    ax.plot([hillock_end[0], initial_segment_end[0]], 
           [hillock_end[1], initial_segment_end[1]], 
           color=COLORS['axon'], linewidth=3, solid_capstyle='round')
    
    # Myelinated axon (with nodes of Ranvier)
    axon_length = 3.0
    num_segments = 5
    segment_length = axon_length / num_segments
    
    current_y = initial_segment_end[1]
    for i in range(num_segments):
        # Myelinated segment
        segment_start = (soma_x, current_y)
        segment_end = (soma_x, current_y - segment_length * 0.9)
        
        # Draw myelin sheath
        myelin_rect = Rectangle((soma_x - 0.08, current_y - segment_length * 0.9), 
                               0.16, segment_length * 0.9,
                               facecolor='#FFFFCC', edgecolor='black', linewidth=1)
        ax.add_patch(myelin_rect)
        
        # Draw axon inside
        ax.plot([segment_start[0], segment_end[0]], 
               [segment_start[1], segment_end[1]], 
               color=COLORS['axon'], linewidth=2, solid_capstyle='round')
        
        # Node of Ranvier
        current_y -= segment_length * 0.9
        node_end = current_y - segment_length * 0.1
        ax.plot([soma_x, soma_x], [current_y, node_end], 
               color=COLORS['axon'], linewidth=2, solid_capstyle='round')
        current_y = node_end
    
    # Axon terminals
    terminal_y = current_y - 0.5
    terminal_angles = [-30, -15, 0, 15, 30]
    
    for angle in terminal_angles:
        angle_rad = np.radians(angle)
        branch_end = (soma_x + 0.5 * np.sin(angle_rad), 
                     terminal_y + 0.3 * np.cos(angle_rad))
        
        # Axon branch
        ax.plot([soma_x, branch_end[0]], [current_y, branch_end[1]], 
               color=COLORS['axon'], linewidth=1.5, solid_capstyle='round')
        
        # Synaptic bouton
        bouton = Circle(branch_end, 0.06, 
                       facecolor=COLORS['synapse'], 
                       edgecolor='black', 
                       linewidth=1)
        ax.add_patch(bouton)
    
    # Set axis properties
    ax.set_xlim(-4, 4)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the figure
    save_figure(fig, output_file)
    plt.close(fig)
    
    print(f"✓ Professional dendritic diagram saved as {output_file}.png and {output_file}.svg")
    return output_file

def create_basic_dendritic_neuron(branching=(2, 3), syn_probs=None, output_file="dendritic_diagram"):
    """
    Create a clean dendritic neuron diagram with reduced clutter.
    
    Parameters:
    -----------
    branching : tuple
        (primary_branches, secondary_branches) counts
    syn_probs : dict or None
        Synapse probabilities {'E': 0.7, 'I': 0.3}
    output_file : str
        Base name for output files
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
        ax, (inset_x, inset_y), 
        inset_size, inset_size,
        facecolor=COLORS['bg_light'], 
        edgecolor=COLORS['border'],
        shadow_preset='soft'
    )
    
    # Draw magnified synapse detail
    synapse_detail_x, synapse_detail_y = inset_x, inset_y
    
    # Draw dendrite segment
    ax.plot([synapse_detail_x - 0.4, synapse_detail_x + 0.4], 
           [synapse_detail_y, synapse_detail_y], 
           color=COLORS['dendrite'], linewidth=6, solid_capstyle='round')
    
    # Draw synapses
    draw_synapse(ax, (synapse_detail_x - 0.2, synapse_detail_y + 0.15), 
                synapse_type='E', size=0.12)
    draw_synapse(ax, (synapse_detail_x + 0.2, synapse_detail_y - 0.15), 
                synapse_type='I', size=0.12)
    
    # Add labels
    text(ax, synapse_detail_x - 0.2, synapse_detail_y + 0.35, 'E', 
         size='small', weight='bold', ha='center', va='center')
    text(ax, synapse_detail_x + 0.2, synapse_detail_y - 0.35, 'I', 
         size='small', weight='bold', ha='center', va='center')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['dendrite'], lw=4, label='Dendrite'),
        plt.Line2D([0], [0], color=COLORS['axon'], lw=4, label='Axon'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['excitatory'], 
                  markersize=8, label='Excitatory Synapse'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['inhibitory'], 
                  markersize=8, label='Inhibitory Synapse'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['soma'], 
                  markersize=10, label='Soma')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
             frameon=True, fancybox=True, shadow=True)
    
    # Set axis properties
    ax.set_xlim(-6, 8)
    ax.set_ylim(-4, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    fig.suptitle('Dendritic Neuron Structure', 
                fontsize=FONT_SIZES['title'], 
                weight='bold', 
                y=0.95)
    
    # Save the figure
    save_figure(fig, output_file)
    plt.close(fig)
    
    print(f"✓ Dendritic diagram saved as {output_file}.png and {output_file}.svg")
    return output_file

def main():
    """Generate the original dendritic diagram."""
    print("Regenerating dendritic diagrams...")
    print("=" * 50)
    
    # Create the clean version for first page (no text)
    clean_file = create_clean_dendritic_neuron()
    
    # Create a comprehensive version with fewer branches
    comprehensive_file = create_basic_dendritic_neuron(
        branching=(4, 2), 
        syn_probs={'E': 0.7, 'I': 0.3},
        output_file="comprehensive_dendritic_diagram"
    )
    
    print("=" * 50)
    print("✓ Dendritic diagrams generated successfully!")
    print(f"✓ Clean version (no text): {clean_file}.png, {clean_file}.svg")
    print(f"✓ Comprehensive version: {comprehensive_file}.png, {comprehensive_file}.svg")
    print("\nTo copy to your original location:")
    print(f"cp {clean_file}.png docs/source/_static/dendritic_diagram.png")

if __name__ == "__main__":
    main() 