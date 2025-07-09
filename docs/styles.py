"""
Professional styling module for documentation figures.

This module provides consistent colors, fonts, and styling utilities
for all documentation figures in the Dendritic Modeling Framework.
"""

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import matplotlib.patches as mpatches

# Professional color palette (Okabe-Ito color blind safe palette + extras)
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
    
    # Layer-specific colors
    'input_layer': '#E8F6F3',   # Light green tint
    'hidden_layer': '#EBF3FD',  # Light blue tint
    'output_layer': '#FDF2E9',  # Light orange tint
    'processing': '#F8F9FA',    # Very light gray
    
    # High contrast grays
    'gray1': '#333333',         # Dark gray
    'gray2': '#666666',         # Medium gray
    'gray3': '#999999',         # Light gray
    'gray4': '#CCCCCC',         # Very light gray
}

# Typography scale
FONT_SIZES = {
    'title': 16,
    'subtitle': 13,
    'body': 10,
    'small': 8,
    'caption': 7,
}

# Shadow presets
SHADOW_PRESETS = {
    'hard': {'offset': (2, -2), 'alpha': 0.4},
    'soft': {'offset': (1, -1), 'alpha': 0.2},
    'none': {'offset': (0, 0), 'alpha': 0.0},
}

def apply_professional_style():
    """Apply consistent professional styling to all matplotlib figures."""
    plt.style.use('default')
    plt.rcParams.update({
        # Figure settings
        'figure.facecolor': COLORS['bg'],
        'figure.edgecolor': COLORS['bg'],
        'savefig.facecolor': COLORS['bg'],
        'savefig.edgecolor': COLORS['bg'],
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'font.size': FONT_SIZES['body'],
        'font.weight': 'normal',
        
        # Axes settings
        'axes.facecolor': COLORS['bg'],
        'axes.edgecolor': COLORS['border'],
        'axes.linewidth': 1.0,
        'axes.titlesize': FONT_SIZES['title'],
        'axes.titleweight': 'bold',
        'axes.titlecolor': COLORS['text'],
        'axes.labelsize': FONT_SIZES['body'],
        'axes.labelcolor': COLORS['text'],
        'axes.labelweight': 'normal',
        
        # Grid settings
        'axes.grid': False,
        'grid.color': COLORS['gray4'],
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # Tick settings
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'xtick.labelsize': FONT_SIZES['small'],
        'ytick.labelsize': FONT_SIZES['small'],
        
        # Legend settings
        'legend.frameon': True,
        'legend.facecolor': COLORS['bg'],
        'legend.edgecolor': COLORS['border'],
        'legend.fontsize': FONT_SIZES['small'],
        'legend.title_fontsize': FONT_SIZES['body'],
        
        # Line settings
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        # Patch settings
        'patch.edgecolor': COLORS['border'],
        'patch.linewidth': 1.0,
    })

def create_shadow_effect(preset='soft'):
    """Create a shadow effect for matplotlib artists."""
    shadow_config = SHADOW_PRESETS.get(preset, SHADOW_PRESETS['soft'])
    if shadow_config['alpha'] == 0:
        return [pe.Normal()]
    
    return [
        pe.SimplePatchShadow(
            offset=shadow_config['offset'],
            shadow_rgbFace=COLORS['gray1'],
            alpha=shadow_config['alpha']
        ),
        pe.Normal()
    ]

def text(ax, x, y, label, *, size='body', weight='normal', color=None, **kwargs):
    """
    Unified text rendering function with consistent styling.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add text to
    x, y : float
        Text position
    label : str
        Text content
    size : str
        Font size key from FONT_SIZES
    weight : str
        Font weight ('normal', 'bold', etc.)
    color : str or None
        Text color (defaults to primary text color)
    **kwargs : dict
        Additional text parameters
    """
    if color is None:
        color = COLORS['text']
    
    fontsize = FONT_SIZES.get(size, FONT_SIZES['body'])
    
    return ax.text(x, y, label, 
                  fontsize=fontsize, 
                  weight=weight, 
                  color=color, 
                  **kwargs)

def get_color_palette():
    """Return the complete color palette for external use."""
    return COLORS.copy()

def get_font_sizes():
    """Return the font size scale for external use."""
    return FONT_SIZES.copy()

def validate_color_key(color_key):
    """Validate that a color key exists in the palette."""
    if color_key not in COLORS:
        available_keys = ', '.join(sorted(COLORS.keys()))
        raise ValueError(f"Color key '{color_key}' not found. Available keys: {available_keys}")
    return True 