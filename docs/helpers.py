"""
Reusable drawing helpers for documentation figures.

This module provides common drawing utilities to eliminate code duplication
and ensure consistent visual appearance across all documentation figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches

try:
    from .styles import COLORS, FONT_SIZES, create_shadow_effect, text
except ImportError:
    from styles import COLORS, FONT_SIZES, create_shadow_effect, text

def arrow(ax, start, end, *, style='straight', color=None, lw=2, ms=20, **kwargs):
    """
    Create a professional arrow between two points.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add arrow to
    start : tuple
        (x, y) coordinates of arrow start
    end : tuple
        (x, y) coordinates of arrow end
    style : str
        Arrow style: 'straight', 'arc', 'arc_reverse'
    color : str or None
        Arrow color (defaults to arrow color from palette)
    lw : float
        Line width
    ms : float
        Mutation scale (arrowhead size)
    **kwargs : dict
        Additional FancyArrowPatch parameters
    """
    if color is None:
        color = COLORS['arrow']
    
    # Set connection style based on requested style
    if style == 'straight':
        connectionstyle = None
    elif style == 'arc':
        connectionstyle = "arc3,rad=0.2"
    elif style == 'arc_reverse':
        connectionstyle = "arc3,rad=-0.2"
    else:
        connectionstyle = None
    
    arrow_patch = FancyArrowPatch(
        start, end, 
        connectionstyle=connectionstyle,
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        **kwargs
    )
    ax.add_patch(arrow_patch)
    return arrow_patch

def neuron(ax, xy, *, kind='E', radius=0.3, label=None, shadow_preset='soft'):
    """
    Draw a neuron with consistent styling.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add neuron to
    xy : tuple
        (x, y) coordinates of neuron center
    kind : str
        Neuron type: 'E' (excitatory), 'I' (inhibitory), 'S' (soma)
    radius : float
        Neuron radius
    label : str or None
        Custom label (defaults to kind)
    shadow_preset : str
        Shadow style: 'hard', 'soft', 'none'
    """
    # Color based on neuron type
    if kind == 'E':
        color = COLORS['excitatory']
    elif kind == 'I':
        color = COLORS['inhibitory']
    elif kind == 'S':
        color = COLORS['soma']
    else:
        color = COLORS['soma']
    
    # Create circle
    circle = Circle(xy, radius, 
                   facecolor=color, 
                   edgecolor='black', 
                   linewidth=1,
                   path_effects=create_shadow_effect(shadow_preset))
    ax.add_patch(circle)
    
    # Add label
    if label is None:
        label = kind
    
    text(ax, xy[0], xy[1], label, 
         size='small', weight='bold', color='white',
         ha='center', va='center')
    
    return circle

def legend_synapses(ax, loc='upper right', **kwargs):
    """
    Create a standard synapse legend.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add legend to
    loc : str
        Legend location
    **kwargs : dict
        Additional legend parameters
    """
    exc_patch = mpatches.Patch(color=COLORS['excitatory'], label='Excitatory')
    inh_patch = mpatches.Patch(color=COLORS['inhibitory'], label='Inhibitory')
    
    return ax.legend(handles=[exc_patch, inh_patch], 
                    loc=loc, 
                    frameon=False, 
                    fontsize=FONT_SIZES['small'],
                    **kwargs)

def legend_neurons(ax, loc='upper right', **kwargs):
    """
    Create a standard neuron legend.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add legend to
    loc : str
        Legend location
    **kwargs : dict
        Additional legend parameters
    """
    exc_patch = mpatches.Patch(color=COLORS['excitatory'], label='Excitatory Neuron')
    inh_patch = mpatches.Patch(color=COLORS['inhibitory'], label='Inhibitory Neuron')
    
    return ax.legend(handles=[exc_patch, inh_patch], 
                    loc=loc, 
                    frameon=False, 
                    fontsize=FONT_SIZES['small'],
                    **kwargs)

def professional_box(ax, xy, width, height, *, 
                    facecolor=None, edgecolor=None, 
                    label=None, label_pos='center',
                    shadow_preset='soft', **kwargs):
    """
    Create a professional-looking box with optional label.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add box to
    xy : tuple
        (x, y) coordinates of box bottom-left corner
    width : float
        Box width
    height : float
        Box height
    facecolor : str or None
        Fill color
    edgecolor : str or None
        Border color
    label : str or None
        Box label
    label_pos : str
        Label position: 'center', 'top', 'bottom'
    shadow_preset : str
        Shadow style: 'hard', 'soft', 'none'
    **kwargs : dict
        Additional Rectangle parameters
    """
    if facecolor is None:
        facecolor = COLORS['bg_light']
    if edgecolor is None:
        edgecolor = COLORS['border']
    
    # Create box
    box = Rectangle(xy, width, height,
                   facecolor=facecolor,
                   edgecolor=edgecolor,
                   linewidth=1,
                   path_effects=create_shadow_effect(shadow_preset),
                   **kwargs)
    ax.add_patch(box)
    
    # Add label if provided
    if label is not None:
        if label_pos == 'center':
            label_x = xy[0] + width/2
            label_y = xy[1] + height/2
            va = 'center'
        elif label_pos == 'top':
            label_x = xy[0] + width/2
            label_y = xy[1] + height - 0.1
            va = 'top'
        elif label_pos == 'bottom':
            label_x = xy[0] + width/2
            label_y = xy[1] + 0.1
            va = 'bottom'
        else:
            label_x = xy[0] + width/2
            label_y = xy[1] + height/2
            va = 'center'
        
        text(ax, label_x, label_y, label,
             size='body', weight='bold', color=COLORS['text'],
             ha='center', va=va)
    
    return box

def draw_dendrite_branch(ax, start_xy, length, angle, *, 
                        color=None, linewidth=3, taper=True,
                        shadow_preset='soft'):
    """
    Draw a dendritic branch with optional tapering.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add branch to
    start_xy : tuple
        (x, y) coordinates of branch start
    length : float
        Branch length
    angle : float
        Branch angle in degrees
    color : str or None
        Branch color
    linewidth : float
        Branch thickness
    taper : bool
        Whether to taper the branch
    shadow_preset : str
        Shadow style: 'hard', 'soft', 'none'
    """
    if color is None:
        color = COLORS['dendrite']
    
    # Calculate end point
    angle_rad = np.radians(angle)
    end_x = start_xy[0] + length * np.cos(angle_rad)
    end_y = start_xy[1] + length * np.sin(angle_rad)
    
    # Create line
    line = plt.Line2D([start_xy[0], end_x], [start_xy[1], end_y],
                     color=color, linewidth=linewidth,
                     path_effects=create_shadow_effect(shadow_preset))
    ax.add_line(line)
    
    return line, (end_x, end_y)

def draw_synapse(ax, xy, *, synapse_type='E', size=0.15, shadow_preset='soft'):
    """
    Draw a synaptic terminal.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add synapse to
    xy : tuple
        (x, y) coordinates of synapse center
    synapse_type : str
        Synapse type: 'E' (excitatory), 'I' (inhibitory)
    size : float
        Synapse size
    shadow_preset : str
        Shadow style: 'hard', 'soft', 'none'
    """
    if synapse_type == 'E':
        color = COLORS['excitatory']
        marker = 'o'
    else:
        color = COLORS['inhibitory']
        marker = 's'
    
    # Draw synapse
    synapse = ax.scatter(xy[0], xy[1], s=size*1000, 
                        c=color, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        path_effects=create_shadow_effect(shadow_preset))
    
    return synapse

def draw_layer_box(ax, x, y, width, height, layer_name, 
                   neuron_positions=None, neuron_types=None,
                   shadow_preset='soft'):
    """
    Draw a neural network layer with neurons.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add layer to
    x, y : float
        Layer position
    width, height : float
        Layer dimensions
    layer_name : str
        Layer label
    neuron_positions : list of tuples or None
        Neuron positions within the layer
    neuron_types : list of str or None
        Neuron types corresponding to positions
    shadow_preset : str
        Shadow style: 'hard', 'soft', 'none'
    """
    # Draw layer background
    layer_box = professional_box(ax, (x, y), width, height,
                               facecolor=COLORS['bg_light'],
                               edgecolor=COLORS['border'],
                               label=layer_name,
                               label_pos='top',
                               shadow_preset=shadow_preset)
    
    # Draw neurons if provided
    if neuron_positions is not None:
        if neuron_types is None:
            neuron_types = ['E'] * len(neuron_positions)
        
        for pos, ntype in zip(neuron_positions, neuron_types):
            neuron(ax, (x + pos[0], y + pos[1]), 
                  kind=ntype, radius=0.2, shadow_preset='none')
    
    return layer_box

def add_formula_box(ax, x, y, formula, *, 
                   fontsize=None, facecolor=None, 
                   edgecolor=None, alpha=0.9,
                   shadow_preset='soft'):
    """
    Add a formula in a professional box.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add formula to
    x, y : float
        Formula position
    formula : str
        Mathematical formula (LaTeX)
    fontsize : float or None
        Font size for formula
    facecolor : str or None
        Background color
    edgecolor : str or None
        Border color
    alpha : float
        Transparency
    shadow_preset : str
        Shadow style: 'hard', 'soft', 'none'
    """
    if fontsize is None:
        fontsize = FONT_SIZES['body']
    if facecolor is None:
        facecolor = COLORS['bg']
    if edgecolor is None:
        edgecolor = COLORS['border']
    
    # Create text box
    bbox_props = dict(boxstyle='round,pad=0.3',
                     facecolor=facecolor,
                     edgecolor=edgecolor,
                     alpha=alpha,
                     path_effects=create_shadow_effect(shadow_preset))
    
    formula_text = ax.text(x, y, formula,
                          fontsize=fontsize,
                          ha='center', va='center',
                          bbox=bbox_props)
    
    return formula_text

def save_figure(fig, filename, dpi=300, formats=['png', 'svg']):
    """
    Save figure in multiple formats with consistent settings.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    dpi : int
        Resolution for raster formats
    formats : list of str
        Output formats ('png', 'svg', 'pdf')
    """
    for fmt in formats:
        if fmt == 'png':
            filepath = f"{filename}.png"
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        elif fmt == 'svg':
            filepath = f"{filename}.svg"
            fig.savefig(filepath, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        elif fmt == 'pdf':
            filepath = f"{filename}.pdf"
            fig.savefig(filepath, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        print(f"Saved: {filepath}")

def compute_layer_positions(layer_counts, layer_sep=3.0, neuron_sep=0.8):
    """
    Compute positions for a multi-layer network diagram.
    
    Parameters:
    -----------
    layer_counts : list of int
        Number of neurons in each layer
    layer_sep : float
        Separation between layers
    neuron_sep : float
        Separation between neurons within a layer
    
    Returns:
    --------
    dict : Layer positions and figure size
    """
    layers = {}
    max_neurons = max(layer_counts)
    
    for i, count in enumerate(layer_counts):
        x = i * layer_sep
        
        # Center neurons vertically
        total_height = (count - 1) * neuron_sep
        start_y = -total_height / 2
        
        positions = []
        for j in range(count):
            y = start_y + j * neuron_sep
            positions.append((x, y))
        
        layers[f'layer_{i}'] = {
            'positions': positions,
            'count': count,
            'x': x
        }
    
    # Compute figure size
    fig_width = (len(layer_counts) - 1) * layer_sep + 2
    fig_height = max_neurons * neuron_sep + 2
    
    return {
        'layers': layers,
        'figsize': (fig_width, fig_height),
        'xlim': (-1, fig_width - 1),
        'ylim': (-fig_height/2 + 1, fig_height/2 - 1)
    } 