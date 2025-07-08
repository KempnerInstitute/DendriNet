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
import matplotlib.patheffects as path_effects


# =====================================================================================
# Enhanced Visual Style
# =====================================================================================

# A more vibrant and professional color palette
COLORS = {
    'excitatory': '#ff6b6b',
    'inhibitory': '#48dbfb',
    'soma': '#485460',
    'dendrite': '#a29bfe',
    'background': '#f5f6fa',
    'text': '#2f3542',
    'arrow': '#57606f',
    'shadow': '#000000'
}

def apply_enhanced_style():
    """Apply a highly polished, modern visual style for all figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'savefig.pad_inches': 0.1,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.weight': 'normal',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#cccccc',
        'axes.labelcolor': COLORS['text'],
        'axes.labelweight': 'bold',
        'axes.titlecolor': COLORS['text'],
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.color': '#e1e1e1',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'legend.frameon': False,
        'legend.fontsize': 10
    })


def add_shadow(artist, dx=2, dy=-2, color='auto', alpha=0.3, **kwargs):
    """Add a drop shadow effect to a matplotlib artist."""
    if color == 'auto':
        shadow_color = COLORS['shadow']
    else:
        shadow_color = color

    shadow = path_effects.SimplePatchShadow(offset=(dx, dy), shadow_rgbFace=shadow_color, alpha=alpha)
    artist.set_path_effects([shadow, path_effects.Normal()])


# Apply the new style globally when the script is loaded
apply_enhanced_style()


# =====================================================================================
# Core Architectural Diagrams
# =====================================================================================

# Output directory
OUTPUT_DIR = "../docs/figures/architecture"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_basic_dendritic_neuron():
    """Create an improved dendritic neuron diagram with professional styling."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Central soma
    soma_x, soma_y = 0, 0
    soma = Circle((soma_x, soma_y), 0.8, color=COLORS['soma'], ec='black', lw=2.5, zorder=10)
    ax.add_patch(soma)
    add_shadow(soma)
    ax.text(soma_x, soma_y, 'Soma', ha='center', va='center', 
           color='white', fontsize=14, weight='bold')
    
    # Axon with terminals
    axon_points = [(0, -0.8), (0, -2.5), (-0.5, -3.5), (-1, -4)]
    for i in range(len(axon_points) - 1):
        ax.plot([axon_points[i][0], axon_points[i+1][0]], 
               [axon_points[i][1], axon_points[i+1][1]], 
               color='#2c3e50', lw=4, solid_capstyle='round')
    
    # Axon terminals
    terminal_points = [(-1.5, -4.2), (-0.8, -4.3), (-1.2, -3.8)]
    for tx, ty in terminal_points:
        ax.plot([axon_points[-1][0], tx], [axon_points[-1][1], ty], 
               color='#2c3e50', lw=2.5, solid_capstyle='round')
        terminal = Circle((tx, ty), 0.15, color='#34495e', ec='black', lw=1)
        ax.add_patch(terminal)
    
    ax.text(-1, -4.8, 'Axon', ha='center', fontsize=12, weight='bold', color='#2c3e50')
    
    # Dendrites with hierarchical branching
    np.random.seed(42)
    dendrite_data = [
        {"angle": 45, "color": "#9b59b6", "label": "Apical"},
        {"angle": 135, "color": "#3498db", "label": "Basal"},
        {"angle": 225, "color": "#3498db", "label": "Basal"},
        {"angle": 315, "color": "#3498db", "label": "Basal"},
    ]
    
    for dend in dendrite_data:
        angle_rad = np.radians(dend["angle"])
        
        # Primary dendrite
        start_x = soma_x + 0.8 * np.cos(angle_rad)
        start_y = soma_y + 0.8 * np.sin(angle_rad)
        end_x = soma_x + 2.5 * np.cos(angle_rad)
        end_y = soma_y + 2.5 * np.sin(angle_rad)
        
        ax.plot([start_x, end_x], [start_y, end_y], 
               color=dend["color"], lw=5, solid_capstyle='round', alpha=0.9)
        
        # Secondary branches
        for i in range(2):
            branch_start = 0.6 + i * 0.7
            bx1 = soma_x + (branch_start + 0.8) * np.cos(angle_rad)
            by1 = soma_y + (branch_start + 0.8) * np.sin(angle_rad)
            
            for offset in [-20, 20]:
                branch_angle = angle_rad + np.radians(offset)
                bx2 = bx1 + 1.2 * np.cos(branch_angle)
                by2 = by1 + 1.2 * np.sin(branch_angle)
                
                ax.plot([bx1, bx2], [by1, by2], 
                       color=dend["color"], lw=3, solid_capstyle='round', alpha=0.8)
                
                # Tertiary branches with synapses
                for j in range(2):
                    tbx1 = bx1 + (j + 1) * 0.4 * np.cos(branch_angle)
                    tby1 = by1 + (j + 1) * 0.4 * np.sin(branch_angle)
                    
                    for toffset in [-15, 15]:
                        tbranch_angle = branch_angle + np.radians(toffset)
                        tbx2 = tbx1 + 0.6 * np.cos(tbranch_angle)
                        tby2 = tby1 + 0.6 * np.sin(tbranch_angle)
                        
                        ax.plot([tbx1, tbx2], [tby1, tby2], 
                               color=dend["color"], lw=1.5, solid_capstyle='round', alpha=0.7)
                        
                        # Synapses
                        syn_color = COLORS['excitatory'] if np.random.random() > 0.3 else COLORS['inhibitory']
                        synapse = Circle((tbx2, tby2), 0.08, color=syn_color, ec='black', lw=0.5)
                        ax.add_patch(synapse)
    
    # Labels
    ax.text(2, 2, 'Apical\nDendrite', ha='center', fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(-2.5, 1.5, 'Basal\nDendrites', ha='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Legend
    elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['excitatory'], 
                  markersize=8, label='Excitatory synapse'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['inhibitory'], 
                  markersize=8, label='Inhibitory synapse'),
        plt.Line2D([0], [0], color='#9b59b6', lw=3, label='Apical dendrite'),
        plt.Line2D([0], [0], color='#3498db', lw=3, label='Basal dendrite'),
    ]
    ax.legend(handles=elements, loc='upper right', frameon=True, fancybox=True)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('Dendritic Neuron Structure', fontsize=18, weight='bold', pad=20)
    
    _save(fig, 'basic_dendritic_neuron.png')
    _save(fig, 'dendritic_diagram.png')  # Also save as dendritic_diagram.png


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
    """Multi-layer E/I network with accurate connectivity: I->E within layer, E->E/I to next layer."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    layers = [
        {"name": "Input", "e": 3, "i": 0},
        {"name": "Layer 1", "e": 4, "i": 2},
        {"name": "Layer 2", "e": 3, "i": 2},
        {"name": "Output", "e": 2, "i": 0},
    ]
    
    x_coords = np.arange(len(layers)) * 3.5 + 1.5
    
    all_neurons = []  # Store all neuron positions
    
    for idx, layer in enumerate(layers):
        x = x_coords[idx]
        
        # Layer background
        if layer["i"] > 0:  # Only draw background for layers with both E and I
            ax.add_patch(FancyBboxPatch((x - 1.2, 1.5), 2.4, 5.5, 
                                       boxstyle="round,pad=0.1", 
                                       facecolor="#f0f0f0", 
                                       ec="#cccccc", 
                                       alpha=0.3, 
                                       zorder=1))
        
        # Layer label
        ax.text(x, 7.5, layer["name"], ha="center", fontsize=14, weight="bold")
        
        # Excitatory neurons (top)
        e_count = layer["e"]
        if e_count > 0:
            e_y_start = 5.5 - (e_count - 1) * 0.4
            for i in range(e_count):
                y = e_y_start + i * 0.8
                neuron = Circle((x, y), 0.3, color=COLORS["excitatory"], ec="black", lw=1.5, zorder=5)
                ax.add_patch(neuron)
                ax.text(x, y, "E", ha="center", va="center", color="white", fontsize=10, weight="bold")
                all_neurons.append({"x": x, "y": y, "type": "E", "layer": idx})
        
        # Inhibitory neurons (bottom)
        i_count = layer["i"]
        if i_count > 0:
            i_y_start = 3.0 - (i_count - 1) * 0.4
            for i in range(i_count):
                y = i_y_start + i * 0.8
                neuron = Circle((x, y), 0.25, color=COLORS["inhibitory"], ec="black", lw=1.5, zorder=5)
                ax.add_patch(neuron)
                ax.text(x, y, "I", ha="center", va="center", color="white", fontsize=9, weight="bold")
                all_neurons.append({"x": x, "y": y, "type": "I", "layer": idx})
    
    # Draw connections
    # 1. Within-layer I->E connections (curved arrows)
    for neuron in all_neurons:
        if neuron["type"] == "I":
            # Find E neurons in same layer
            for target in all_neurons:
                if (target["type"] == "E" and 
                    target["layer"] == neuron["layer"]):
                    # Curved arrow from I to E within layer
                    arrow = FancyArrowPatch((neuron["x"] + 0.1, neuron["y"] + 0.25),
                                          (target["x"] - 0.1, target["y"] - 0.3),
                                          connectionstyle="arc3,rad=0.3",
                                          arrowstyle="-|>",
                                          mutation_scale=12,
                                          lw=1.5,
                                          color=COLORS["inhibitory"],
                                          alpha=0.7,
                                          zorder=3)
                    ax.add_patch(arrow)
    
    # 2. Between-layer E->E and E->I connections
    for neuron in all_neurons:
        if neuron["type"] == "E" and neuron["layer"] < len(layers) - 1:
            # Connect to both E and I neurons in next layer
            for target in all_neurons:
                if target["layer"] == neuron["layer"] + 1:
                    # Straight connections to next layer
                    color = COLORS["excitatory"] if target["type"] == "E" else COLORS["inhibitory"]
                    alpha = 0.4 if target["type"] == "E" else 0.3
                    ax.plot([neuron["x"] + 0.3, target["x"] - 0.3],
                           [neuron["y"], target["y"]],
                           color=color,
                           alpha=alpha,
                           lw=1.2,
                           zorder=2)
    
    # Input arrow
    arrow = FancyArrowPatch((0.2, 5), (1.0, 5),
                           arrowstyle="->",
                           mutation_scale=20,
                           lw=2.5,
                           color=COLORS["arrow"])
    ax.add_patch(arrow)
    ax.text(0, 5, "Input", ha="right", fontsize=12)
    
    # Add annotations
    ax.text(x_coords[1], 1.0, "Within-layer\nI→E inhibition", 
            ha="center", fontsize=10, style="italic", color="#666666")
    ax.text((x_coords[1] + x_coords[2])/2, 6.5, "Between-layer\nE→E/I propagation", 
            ha="center", fontsize=10, style="italic", color="#666666")
    
    # Legend
    exc_patch = patches.Patch(color=COLORS["excitatory"], label="Excitatory (E)")
    inh_patch = patches.Patch(color=COLORS["inhibitory"], label="Inhibitory (I)")
    ax.legend(handles=[exc_patch, inh_patch], loc="upper right", frameon=True)
    
    ax.set_xlim(-0.5, x_coords[-1] + 2)
    ax.set_ylim(0.5, 8)
    ax.axis("off")
    
    ax.set_title("Multi-Layer E/I Network Architecture", fontsize=16, weight="bold")
    
    _save(fig, "multilayer_ei_network.png")


def create_shunting_inhibition_diagram():
    """A polished diagram of the shunting inhibition mechanism."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis('off')

    # Central dendrite/soma
    soma = Circle((0.5, 0.5), 0.2, color=COLORS['soma'])
    ax.add_patch(soma)
    add_shadow(soma)
    ax.text(0.5, 0.5, 'Soma', ha='center', va='center', color='white', fontsize=12, weight='bold')

    # Input arrows with labels
    def draw_input(x, y, label, color, rad):
        arrow = FancyArrowPatch((x, y + 0.1), (0.5, 0.6), connectionstyle=f"arc3,rad={rad}",
                                arrowstyle='->,head_length=6,head_width=4', lw=2, color=color)
        ax.add_patch(arrow)
        add_shadow(arrow)
        text = ax.text(x, y + 0.2, label, ha='center', va='center', fontsize=12, weight='bold', color=color)
        add_shadow(text, dx=1, dy=-1, alpha=0.1)

    draw_input(0.15, 0.8, "Excitatory (E)", COLORS['excitatory'], -0.3)
    draw_input(0.85, 0.8, "Inhibitory (I)", COLORS['inhibitory'], 0.3)

    # Output arrow
    output_arrow = FancyArrowPatch((0.5, 0.3), (0.5, 0.1), arrowstyle='->,head_length=8,head_width=6',
                                   lw=2.5, color=COLORS['arrow'])
    ax.add_patch(output_arrow)
    add_shadow(output_arrow)
    ax.text(0.5, 0.0, "Output", ha='center', va='center', fontsize=12, weight='bold', color=COLORS['text'])

    # Equation
    eq_text = r'$V_{out} = \frac{\sum E}{\sum E + \sum I + \epsilon}$'
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='#cccccc', lw=0.5)
    text_bbox = ax.text(0.5, -0.25, eq_text, ha='center', va='center', fontsize=14, bbox=bbox_props)
    add_shadow(text_bbox.get_bbox_patch())

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 1.1)
    fig.suptitle("Shunting Inhibition Mechanism", fontsize=16, weight='bold')
    _save(fig, 'shunting_inhibition_diagram.png')


def create_network_architecture_overview():
    """A highly polished, professional overview of the network architecture."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    def draw_stage(x, y, w, h, label, sub_labels=None, color=COLORS['background']):
        """Draw a single stage of the pipeline with shadow and text."""
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=color, ec='#333333', lw=0.5)
        ax.add_patch(box)
        add_shadow(box)

        title_text = ax.text(x + w / 2, y + h - 0.5, label, ha='center', va='center', fontsize=12, weight='bold', color=COLORS['text'])
        add_shadow(title_text, dx=1, dy=-1, alpha=0.1)

        if sub_labels:
            for i, sub in enumerate(sub_labels):
                ax.text(x + w / 2, y + h - 1.2 - (i * 0.4), sub, ha='center', va='center', fontsize=9, color=COLORS['text'])

    # Pipeline stages
    draw_stage(1, 2, 2, 2, "Input", ["Images", "Time-Series"], color='#e0f7fa')
    draw_stage(4, 2, 2, 2, "Pre-Net", ["Feature Extraction"], color='#c5e1a5')
    draw_stage(7, 1.5, 3, 3, "Dendritic Core", ["E/I Separation", "Dendritic Processing", "Shunting Inhibition"], color=COLORS['dendrite'])
    draw_stage(11, 2, 2, 2, "Output", ["Classifier Head"], color='#ffe0b2')

    # Arrows
    def draw_arrow(start, end):
        arrow = FancyArrowPatch(start, end, arrowstyle='Simple,tail_width=0.7,head_width=4,head_length=8',
                                connectionstyle="arc3,rad=0.1", color=COLORS['arrow'],
                                shrinkA=10, shrinkB=10)
        add_shadow(arrow, dx=1, dy=-1)
        ax.add_patch(arrow)

    draw_arrow((3, 3), (4, 3))
    draw_arrow((6, 3), (7, 3))
    draw_arrow((10, 3), (11, 3))

    # E/I split visualization inside the core box
    ax.add_patch(Circle((8.2, 2.5), 0.3, color=COLORS['excitatory']))
    ax.add_patch(Circle((8.8, 2.5), 0.3, color=COLORS['inhibitory']))

    # Title
    fig.suptitle("Dendritic Modeling Framework Architecture", fontsize=18, weight='bold', color=COLORS['text'])

    _save(fig, 'network_architecture_overview.png')


def create_dendritic_computation_flow():
    """Create a compact dendritic computation flow diagram without excessive whitespace."""

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ---- Input icons ----
    exc_positions = [(0.8 + i * 0.8, 3.8) for i in range(4)]
    inh_positions = [(0.8 + i * 0.8, 2.2) for i in range(4)]

    for idx, (x, y) in enumerate(exc_positions):
        tri = plt.Polygon([(x, y), (x - 0.15, y + 0.25), (x + 0.15, y + 0.25)], 
                         color=COLORS["excitatory"], ec="k", linewidth=1)
        ax.add_patch(tri)
        ax.text(x, y - 0.35, f"E{idx + 1}", ha="center", fontsize=9)

    for idx, (x, y) in enumerate(inh_positions):
        circ = Circle((x, y), 0.12, color=COLORS["inhibitory"], ec="k", linewidth=1)
        ax.add_patch(circ)
        ax.text(x, y - 0.35, f"I{idx + 1}", ha="center", fontsize=9)

    # ---- Summation nodes ----
    sum_e = Circle((4.5, 3.8), 0.25, color="#ffffff", ec=COLORS["excitatory"], lw=2)
    sum_i = Circle((4.5, 2.2), 0.25, color="#ffffff", ec=COLORS["inhibitory"], lw=2)
    ax.add_patch(sum_e)
    ax.add_patch(sum_i)
    add_shadow(sum_e)
    add_shadow(sum_i)
    ax.text(4.5, 3.8, "Σ", ha="center", va="center", fontsize=12, weight="bold", color=COLORS["excitatory"])
    ax.text(4.5, 2.2, "Σ", ha="center", va="center", fontsize=12, weight="bold", color=COLORS["inhibitory"])

    # arrows from inputs to sums
    for (x, y) in exc_positions:
        ax.add_patch(FancyArrowPatch((x + 0.15, y), (4.25, 3.8), 
                                    arrowstyle="-", mutation_scale=10, lw=1, 
                                    color=COLORS["excitatory"], alpha=0.6))
    for (x, y) in inh_positions:
        ax.add_patch(FancyArrowPatch((x + 0.12, y), (4.25, 2.2), 
                                    arrowstyle="-", mutation_scale=10, lw=1, 
                                    color=COLORS["inhibitory"], alpha=0.6))

    # ---- Division node ----
    div_box = FancyBboxPatch((5.8, 2.5), 1.4, 1.0, 
                            boxstyle="round,pad=0.15", 
                            facecolor=COLORS["dendrite"], 
                            ec="#333333", lw=2)
    ax.add_patch(div_box)
    add_shadow(div_box)
    ax.text(6.5, 3.0, r"$\frac{E}{E + I + \varepsilon}$", ha="center", va="center", 
           fontsize=12, color="white", weight="bold")

    # arrows from sums to division
    ax.add_patch(FancyArrowPatch((4.75, 3.8), (5.8, 3.2), 
                                arrowstyle="-|>", mutation_scale=12, lw=1.5, 
                                color=COLORS["excitatory"]))
    ax.add_patch(FancyArrowPatch((4.75, 2.2), (5.8, 2.8), 
                                arrowstyle="-|>", mutation_scale=12, lw=1.5, 
                                color=COLORS["inhibitory"]))

    # ---- Output arrow ----
    ax.add_patch(FancyArrowPatch((7.2, 3.0), (8.5, 3.0), 
                                arrowstyle="-|>", mutation_scale=15, lw=2, 
                                color=COLORS["arrow"]))
    add_shadow(ax.patches[-1])
    ax.text(8.7, 3.0, "Vout", ha="left", va="center", fontsize=11, weight="bold")

    # ---- Title ----
    ax.text(5, 4.6, "Dendritic Computation Flow", ha="center", fontsize=16, weight="bold")

    # ---- Legend ----
    exc_patch = patches.Patch(color=COLORS["excitatory"], label="Excitatory (E)")
    inh_patch = patches.Patch(color=COLORS["inhibitory"], label="Inhibitory (I)")
    ax.legend(handles=[exc_patch, inh_patch], loc="lower right", frameon=True, fontsize=9)

    # ---- Annotations ----
    ax.text(2.5, 1.4, "Linear summation", ha="center", fontsize=8, style="italic", color="#666666")
    ax.text(6.5, 1.8, "Shunting inhibition", ha="center", fontsize=8, style="italic", color="#666666")

    _save(fig, "dendritic_computation_flow.png")


# =====================================================================================
# Bespoke diagrams for previously placeholder-only assets
# =====================================================================================


def _save(fig, name):
    """Helper to save fig at high DPI to OUTPUT_DIR."""
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_einet_architecture():
    """Create a clear EINet architecture diagram showing E/I separation and dendritic processing."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define components
    components = [
        {"name": "Input\nImage", "x": 1, "y": 4, "w": 1.5, "h": 1.5, "color": "#e0f7fa"},
        {"name": "Conv\nLayers", "x": 3.5, "y": 4, "w": 1.5, "h": 1.5, "color": "#c5e1a5"},
        {"name": "E/I\nSeparation", "x": 6, "y": 4, "w": 1.5, "h": 1.5, "color": "#ffccbc"},
    ]
    
    # Draw components
    for comp in components:
        box = FancyBboxPatch((comp["x"] - comp["w"]/2, comp["y"] - comp["h"]/2), 
                            comp["w"], comp["h"], 
                            boxstyle="round,pad=0.15", 
                            facecolor=comp["color"], 
                            ec="#333333", 
                            lw=2)
        ax.add_patch(box)
        add_shadow(box)
        ax.text(comp["x"], comp["y"], comp["name"], 
               ha="center", va="center", fontsize=12, weight="bold")
    
    # E/I input channels
    e_inputs = FancyBboxPatch((7.5, 4.5), 2, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=COLORS["excitatory"], 
                             ec="#333333", lw=2, alpha=0.8)
    i_inputs = FancyBboxPatch((7.5, 2.8), 2, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=COLORS["inhibitory"], 
                             ec="#333333", lw=2, alpha=0.8)
    ax.add_patch(e_inputs)
    ax.add_patch(i_inputs)
    add_shadow(e_inputs)
    add_shadow(i_inputs)
    
    ax.text(8.5, 5.1, "Excitatory\nInputs", ha="center", va="center", 
           fontsize=11, color="white", weight="bold")
    ax.text(8.5, 3.4, "Inhibitory\nInputs", ha="center", va="center", 
           fontsize=11, color="white", weight="bold")
    
    # Dendritic computation
    dendrite = FancyBboxPatch((10, 3), 2, 2.5, 
                             boxstyle="round,pad=0.15", 
                             facecolor=COLORS["dendrite"], 
                             ec="#333333", lw=2)
    ax.add_patch(dendrite)
    add_shadow(dendrite)
    
    # Division formula inside dendrite
    ax.text(11, 4.2, "Dendritic", ha="center", fontsize=12, weight="bold", color="white")
    ax.text(11, 3.7, "Computation", ha="center", fontsize=12, weight="bold", color="white")
    ax.text(11, 3.0, r"$\frac{E}{E+I+\epsilon}$", ha="center", fontsize=14, color="white")
    
    # Output
    output_box = FancyBboxPatch((12.5, 3.5), 1.5, 1.5, 
                               boxstyle="round,pad=0.15", 
                               facecolor="#ffe0b2", 
                               ec="#333333", lw=2)
    ax.add_patch(output_box)
    add_shadow(output_box)
    ax.text(13.25, 4.25, "Output", ha="center", fontsize=12, weight="bold")
    
    # Arrows
    arrows = [
        ((2.25, 4.25), (2.75, 4.25)),  # Input to Conv
        ((4.75, 4.25), (5.25, 4.25)),  # Conv to E/I
        ((6.75, 4.5), (7.5, 5.1)),     # To E inputs
        ((6.75, 3.5), (7.5, 3.4)),     # To I inputs
        ((9.5, 5.1), (10, 4.1)),       # E to dendrite
        ((9.5, 3.4), (10, 3.9)),       # I to dendrite
        ((12, 4.25), (12.5, 4.25)),    # Dendrite to output
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle="-|>", 
                               mutation_scale=15, 
                               lw=2, 
                               color=COLORS["arrow"])
        ax.add_patch(arrow)
        add_shadow(arrow)
    
    # Annotations
    ax.text(2.25, 3.0, "Raw pixels", ha="center", fontsize=9, style="italic", color="#666666")
    ax.text(4.75, 3.0, "Features", ha="center", fontsize=9, style="italic", color="#666666")
    ax.text(6.75, 2.2, "Separate\nchannels", ha="center", fontsize=9, style="italic", color="#666666")
    ax.text(11, 2.0, "Shunting\ninhibition", ha="center", fontsize=9, style="italic", color="#666666")
    
    ax.set_xlim(0, 14.5)
    ax.set_ylim(1.5, 6)
    ax.axis("off")
    
    ax.set_title("EINet Architecture: Excitatory/Inhibitory Network", fontsize=16, weight="bold")
    
    _save(fig, "einet_architecture.png")


def create_einet_classifier_dataflow():
    """Visualise the forward data-flow through an EINetClassifier."""

    fig, ax = plt.subplots(figsize=(14, 5))

    stages = [
        ("Input\nImage", 1),
        ("Input\nNetwork", 3),
        ("E/I\nSeparation", 5),
        ("Dendritic\nLayers", 7),
        ("Classifier\nHead", 9),
        ("Prediction", 11),
    ]

    for label, x in stages:
        box = FancyBboxPatch((x - 0.8, 1.5), 1.6, 1.6, boxstyle="round,pad=0.2", facecolor=COLORS["background"], edgecolor="#2c3e50", linewidth=2)
        ax.add_patch(box)
        ax.text(x, 2.3, label, ha="center", va="center", weight="bold", fontsize=11)

    # arrows
    for (label, x_start), (_, x_end) in zip(stages[:-1], stages[1:]):
        arrow = FancyArrowPatch((x_start + 0.8, 2.3), (x_end - 0.8, 2.3), arrowstyle="->", mutation_scale=20, lw=2, color=COLORS["arrow"])
        ax.add_patch(arrow)

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    ax.set_title("EINetClassifier Forward Data-flow", fontsize=16, weight="bold")

    _save(fig, "einet_classifier_dataflow.png")


def _draw_dendrinet(ax, center_x, layer_counts):
    """Helper to draw a simple multilayer dendrinet structure."""
    layer_spacing = 2.0
    cmap = plt.get_cmap("viridis")
    xs = np.arange(len(layer_counts)) * layer_spacing + center_x
    for idx, (n, x) in enumerate(zip(layer_counts, xs)):
        ys = np.linspace(-1, 1, n)
        color = cmap(idx / len(layer_counts))
        ax.scatter(np.full_like(ys, x), ys, s=300, color=color, edgecolor="k", zorder=3)
        # connections
        if idx > 0:
            prev_n = layer_counts[idx - 1]
            prev_x = xs[idx - 1]
            prev_ys = np.linspace(-1, 1, prev_n)
            for y0 in prev_ys:
                for y1 in ys:
                    ax.plot([prev_x, x], [y0, y1], color="#bbbbbb", alpha=0.3, lw=1)

    ax.axis("off")


def create_dendrinet_1():
    """Generate schematic for Dendrinet variant 1."""
    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_dendrinet(ax, center_x=0, layer_counts=[4, 6, 3])
    ax.set_title("Dendrinet-1 Architecture", fontsize=14, weight="bold")
    _save(fig, "dendrinet_1.png")


def create_dendrinet_2():
    """Generate schematic for Dendrinet variant 2."""
    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_dendrinet(ax, center_x=0, layer_counts=[6, 8, 6, 3])
    ax.set_title("Dendrinet-2 Architecture", fontsize=14, weight="bold")
    _save(fig, "dendrinet_2.png")


def _skip_experiment_tracking():
    """Diagram for experiment tracking pipeline (e.g., lightning + W&B)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    boxes = [
        ("Config YAML", 1, 4),
        ("Training Script", 4, 4),
        ("Trainer", 7, 4),
        ("W&B Logger", 10, 5.5),
        ("TensorBoard", 10, 2.5),
    ]

    for label, x, y in boxes:
        box = FancyBboxPatch((x - 1, y - 0.5), 2, 1, boxstyle="round,pad=0.2", facecolor=COLORS["background"], edgecolor="#2c3e50", linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", weight="bold")

    # arrows
    arrow_pairs = [((2, 4), (3, 4)), ((5, 4), (6, 4)), ((8, 4), (9, 5.5)), ((8, 4), (9, 2.5))]
    for start, end in arrow_pairs:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=15, lw=1.8, color=COLORS["arrow"]))

    ax.set_xlim(0, 12)
    ax.set_ylim(1.5, 7)
    ax.axis("off")
    ax.set_title("Experiment Tracking Workflow", fontsize=16, weight="bold")
    _save(fig, "experiment_tracking.png")


def _skip_configuration_system():
    """Diagram illustrating hierarchical Hydra/OMEGACONF config generation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = ["Defaults.yaml", "Dataset.yaml", "Model.yaml", "Optimizer.yaml", "Run Overrides"]
    y_positions = np.arange(len(levels))[::-1]
    for lvl, y in zip(levels, y_positions):
        rect = FancyBboxPatch((2, y), 6, 0.8, boxstyle="round,pad=0.15", facecolor=COLORS["background"], edgecolor="#2c3e50", linewidth=2)
        ax.add_patch(rect)
        ax.text(5, y + 0.4, lvl, ha="center", va="center", weight="bold")
    # arrow downward merge
    for y in y_positions[:-1]:
        ax.add_patch(FancyArrowPatch((5, y), (5, y - 0.2), arrowstyle="-|>", mutation_scale=15, lw=1.5, color=COLORS["arrow"]))
    ax.text(5, -0.8, "Final Config Object", ha="center", fontsize=12, weight="bold")
    ax.add_patch(FancyArrowPatch((5, -0.2), (5, -0.7), arrowstyle="->", mutation_scale=15, lw=1.5, color=COLORS["arrow"]))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, len(levels) + 1)
    ax.axis("off")
    ax.set_title("Configuration Composition", fontsize=16, weight="bold")
    _save(fig, "configuration_system.png")


def _skip_sweep_job_submission():
    """Diagram depicting distributed sweep submission to SLURM."""
    fig, ax = plt.subplots(figsize=(10, 5))
    # Boxes
    ax.add_patch(FancyBboxPatch((1, 3), 2, 1, boxstyle="round,pad=0.2", facecolor=COLORS["background"], edgecolor="#2c3e50", linewidth=2))
    ax.text(2, 3.5, "config_sweep.py", ha="center", va="center", weight="bold")

    ax.add_patch(FancyBboxPatch((4, 3), 2, 1, boxstyle="round,pad=0.2", facecolor=COLORS["background"], edgecolor="#2c3e50", linewidth=2))
    ax.text(5, 3.5, "Slurm Array Job", ha="center", va="center", weight="bold")

    # Cluster nodes
    for i in range(3):
        y = 1 - i * 1.2
        ax.add_patch(Rectangle((4 + i * 2, y), 1.5, 0.8, facecolor=COLORS["background"], edgecolor="#2c3e50"))
        ax.text(4 + i * 2 + 0.75, y + 0.4, f"GPU{i}", ha="center", va="center")

    # Arrows
    ax.add_patch(FancyArrowPatch((3, 3.5), (4, 3.5), arrowstyle="->", mutation_scale=15, lw=2, color=COLORS["arrow"]))
    for i in range(3):
        ax.add_patch(FancyArrowPatch((5, 3), (4 + i * 2 + 0.75, 1 + 0.8 - i * 1.2), arrowstyle="->", mutation_scale=15, lw=1.5, color=COLORS["arrow"]))

    ax.set_xlim(0, 9)
    ax.set_ylim(-2, 5)
    ax.axis("off")
    ax.set_title("Sweep Job Submission (SLURM)", fontsize=16, weight="bold")
    _save(fig, "sweep_job_submission.png")


# =====================================================================================
# Additional diagrams to replace remaining placeholders
# =====================================================================================


def _simple_flow(ax, labels, y):
    """Draw a horizontal flow of rounded boxes with arrows."""
    for idx, label in enumerate(labels):
        x = 1 + idx * 2
        box = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8, boxstyle="round,pad=0.2", facecolor=COLORS["background"], ec="#2c3e50", lw=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center")
        if idx < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.8, y), (x + 1.2, y), arrowstyle="->", mutation_scale=12, lw=1.2, color=COLORS["arrow"]))


def create_input_network_architecture():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    _simple_flow(ax, ["Raw Input", "Pre-Net", "Feature Map"], y=1.25)
    ax.set_xlim(0, 7)
    ax.set_ylim(0.2, 2.3)
    ax.axis("off")
    ax.set_title("Input Network Architecture", fontsize=12, weight="bold")
    _save(fig, "input_network_architecture.png")


def create_dendritic_processing():
    """Show dendritic processing: Neuron with multiple dendrites receiving synaptic inputs."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Neuron (soma)
    soma = Circle((2, 3), 0.6, color=COLORS["soma"], ec="black", lw=2.5)
    ax.add_patch(soma)
    add_shadow(soma)
    ax.text(2, 3, "Neuron\n(Soma)", ha="center", va="center", 
           color="white", fontsize=12, weight="bold")
    
    # Multiple dendrites extending from soma
    dendrite_data = [
        {"angle": 0, "label": "Dendrite 1", "y_offset": 0.5},
        {"angle": 20, "label": "Dendrite 2", "y_offset": 0},  
        {"angle": -20, "label": "Dendrite 3", "y_offset": -0.5},
    ]
    
    for i, dend in enumerate(dendrite_data):
        angle_rad = np.radians(dend["angle"])
        
        # Main dendrite
        start_x = 2 + 0.6 * np.cos(angle_rad)
        start_y = 3 + 0.6 * np.sin(angle_rad)
        end_x = 2 + 3.5 * np.cos(angle_rad)
        end_y = 3 + 3.5 * np.sin(angle_rad) + dend["y_offset"]
        
        # Draw dendrite as tapered shape
        dendrite = FancyBboxPatch((end_x - 0.3, end_y - 0.25), 2.5, 0.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLORS["dendrite"],
                                 ec="black", lw=1.5)
        ax.add_patch(dendrite)
        add_shadow(dendrite)
        
        # Connection from soma
        ax.plot([start_x, end_x - 0.3], [start_y, end_y], 
               color=COLORS["dendrite"], lw=3, alpha=0.8)
        
        # Label
        ax.text(end_x + 1, end_y, dend["label"], ha="center", va="center",
               fontsize=10, weight="bold")
        
        # Synapses along dendrite
        synapse_positions = np.linspace(0.3, 0.9, 5)
        for j, pos in enumerate(synapse_positions):
            sx = end_x + pos * 2
            sy = end_y + np.sin(pos * 10) * 0.1  # Slight variation
            
            # Alternate between excitatory and inhibitory
            if (i + j) % 2 == 0:
                # Excitatory synapse
                tri = plt.Polygon([(sx, sy), (sx-0.08, sy+0.12), (sx+0.08, sy+0.12)],
                                 color=COLORS["excitatory"], ec="black", lw=0.5)
                ax.add_patch(tri)
                # Input arrow
                ax.annotate("", xy=(sx, sy+0.12), xytext=(sx, sy+0.4),
                           arrowprops=dict(arrowstyle="->", color=COLORS["excitatory"], lw=1))
            else:
                # Inhibitory synapse
                circ = Circle((sx, sy), 0.06, color=COLORS["inhibitory"], ec="black", lw=0.5)
                ax.add_patch(circ)
                # Input arrow
                ax.annotate("", xy=(sx, sy+0.06), xytext=(sx, sy+0.3),
                           arrowprops=dict(arrowstyle="->", color=COLORS["inhibitory"], lw=1))
    
    # Axon
    ax.plot([2, 2, 1.5], [2.4, 1.5, 0.5], color="#34495e", lw=3)
    ax.text(1.5, 0.2, "Axon", ha="center", fontsize=10, weight="bold")
    
    # Legend
    exc_patch = patches.Patch(color=COLORS["excitatory"], label="Excitatory synapse")
    inh_patch = patches.Patch(color=COLORS["inhibitory"], label="Inhibitory synapse")
    ax.legend(handles=[exc_patch, inh_patch], loc="upper left", frameon=True)
    
    # Annotations
    ax.text(6.5, 4.5, "Synaptic inputs arrive\nalong dendrites", 
           ha="center", fontsize=10, style="italic", color="#666666",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Dendritic Processing", fontsize=16, weight="bold")
    
    _save(fig, "dendritic_processing.png")


def _skip_training_strategies():
    fig, ax = plt.subplots(figsize=(6, 3))
    strategies = ["Supervised", "Self-Supervised", "Contrastive"]
    ax.barh(strategies, [3, 2, 1], color=["#457b9d", "#e9c46a", "#e76f51"])
    ax.set_xlim(0, 3.5)
    ax.set_xlabel("Relative Usage")
    ax.set_title("Training Strategies")
    _save(fig, "training_strategies.png")  # preserved but disabled from auto-run


def _skip_sequential_vs_joint():
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot([0, 1, 2], [0.2, 0.8, 1], label="Joint", lw=2)
    ax.plot([0, 1, 2], [0.2, 0.4, 0.6], label="Sequential", lw=2, linestyle="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Start", "Mid", "End"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Sequential vs. Joint Training")
    ax.legend()
    _save(fig, "sequential_vs_joint.png")


def _skip_head_training_strategies():
    fig, ax = plt.subplots(figsize=(6, 3))
    heads = ["Linear", "MLP", "Dendritic"]
    colors = ["#2a9d8f", "#e9c46a", "#e63946"]
    ax.bar(heads, [0.7, 0.8, 0.9], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Head Training Strategies")
    _save(fig, "head_training_strategies.png")


def _skip_learning_strategies():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pie([40, 35, 25], labels=["Backprop", "Hebbian", "RL"], autopct="%1.0f%%", colors=["#264653", "#2a9d8f", "#e9c46a"])
    ax.set_title("Learning Strategies Share")
    _save(fig, "learning_strategies.png")


def _skip_ablation_visualization():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.errorbar([1, 2, 3], [0.8, 0.6, 0.4], yerr=[0.05, 0.04, 0.03], fmt="o-", capsize=4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Full", "No Inhib", "No Dendrite"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study")
    _save(fig, "ablation_visualization.png")


def _skip_computational_properties():
    fig, ax = plt.subplots(figsize=(6, 3))
    props = ["Add", "Multiply", "Normalize"]
    ax.barh(props, [1, 0.8, 0.9], color="#8ecae6")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Capability (relative)")
    ax.set_title("Computational Properties")
    _save(fig, "computational_properties.png")


def create_input_modes_complete():
    """Illustrate the three input modes defined in transfer.py and the effect of independent_pathways flag."""

    fig, ax = plt.subplots(figsize=(8, 3))

    modes = [
        ("Mode 0", "Raw pixel values"),
        ("Mode 1\n(independent)", "Separate E / I images"),
        ("Mode 1\n(shared)", "Shared channels"),
    ]

    total = len(modes)
    x_positions = np.linspace(1.3, 6.7, total)  # evenly spaced within figure

    for (x, (title, desc)) in zip(x_positions, modes):
        box = FancyBboxPatch((x - 1.1, 0.55), 2.2, 1.3, boxstyle="round,pad=0.25", facecolor=COLORS["background"], ec="#2c3e50", lw=1.0)
        ax.add_patch(box)
        ax.text(x, 1.35, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x, 0.9, desc, ha="center", va="center", fontsize=8.5)

    ax.set_xlim(0, 8)
    ax.set_ylim(0.4, 2.0)
    ax.axis("off")
    ax.set_title("Input Modes Supported by transfer.py", fontsize=12, weight="bold")

    _save(fig, "input_modes_complete.png")


def _skip_installation_workflow():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    _simple_flow(ax, ["Clone Repo", "Install Deps", "Run Tests", "Build Docs"], y=1.25)
    ax.set_xlim(0, 9)
    ax.set_ylim(0.2, 2.3)
    ax.axis("off")
    ax.set_title("Installation Workflow", fontsize=12, weight="bold")
    _save(fig, "installation_workflow.png")


def create_dendritic_computation():
    """Create a clear diagram showing the dendritic computation mathematics."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Title
    ax.text(5, 6.5, "Dendritic Computation", ha="center", fontsize=18, weight="bold")
    
    # Input signals
    exc_x = [1, 2, 3]
    inh_x = [1.5, 2.5]
    
    for i, x in enumerate(exc_x):
        # Excitatory inputs
        tri = plt.Polygon([(x, 5), (x-0.2, 5.3), (x+0.2, 5.3)], 
                         color=COLORS["excitatory"], ec="black", lw=1.5)
        ax.add_patch(tri)
        add_shadow(tri)
        ax.text(x, 4.6, f"E{i+1}", ha="center", fontsize=10, weight="bold")
    
    for i, x in enumerate(inh_x):
        # Inhibitory inputs
        circ = Circle((x, 3.5), 0.15, color=COLORS["inhibitory"], ec="black", lw=1.5)
        ax.add_patch(circ)
        add_shadow(circ)
        ax.text(x, 3.0, f"I{i+1}", ha="center", fontsize=10, weight="bold")
    
    # Summation boxes
    sum_e_box = FancyBboxPatch((4, 4.5), 1.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor="white", 
                              ec=COLORS["excitatory"], lw=2)
    sum_i_box = FancyBboxPatch((4, 3), 1.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor="white", 
                              ec=COLORS["inhibitory"], lw=2)
    ax.add_patch(sum_e_box)
    ax.add_patch(sum_i_box)
    add_shadow(sum_e_box)
    add_shadow(sum_i_box)
    
    ax.text(4.75, 5, r"$\sum E_i$", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(4.75, 3.5, r"$\sum I_j$", ha="center", va="center", fontsize=14, weight="bold")
    
    # Arrows to summation
    for x in exc_x:
        arrow = FancyArrowPatch((x, 5), (4, 5), 
                               connectionstyle="arc3,rad=0.2", 
                               arrowstyle="-", 
                               lw=1.5, 
                               color=COLORS["excitatory"], 
                               alpha=0.7)
        ax.add_patch(arrow)
    
    for x in inh_x:
        arrow = FancyArrowPatch((x, 3.5), (4, 3.5), 
                               connectionstyle="arc3,rad=-0.2", 
                               arrowstyle="-", 
                               lw=1.5, 
                               color=COLORS["inhibitory"], 
                               alpha=0.7)
        ax.add_patch(arrow)
    
    # Division operation
    div_box = FancyBboxPatch((6.5, 3), 2.5, 2, 
                            boxstyle="round,pad=0.2", 
                            facecolor=COLORS["dendrite"], 
                            ec="black", lw=2)
    ax.add_patch(div_box)
    add_shadow(div_box)
    
    # Division formula
    formula = r"$V_{out} = \frac{\sum E_i}{\sum E_i + \sum I_j + \epsilon}$"
    ax.text(7.75, 4, formula, ha="center", va="center", 
           fontsize=16, color="white", weight="bold")
    
    # Arrows to division
    arrow1 = FancyArrowPatch((5.5, 5), (6.5, 4.2), 
                            arrowstyle="-|>", 
                            mutation_scale=15, 
                            lw=2, 
                            color=COLORS["excitatory"])
    arrow2 = FancyArrowPatch((5.5, 3.5), (6.5, 3.8), 
                            arrowstyle="-|>", 
                            mutation_scale=15, 
                            lw=2, 
                            color=COLORS["inhibitory"])
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    add_shadow(arrow1)
    add_shadow(arrow2)
    
    # Output
    output_arrow = FancyArrowPatch((9, 4), (10, 4), 
                                  arrowstyle="-|>", 
                                  mutation_scale=20, 
                                  lw=2.5, 
                                  color=COLORS["arrow"])
    ax.add_patch(output_arrow)
    add_shadow(output_arrow)
    
    ax.text(10.2, 4, r"$V_{out} \in [0, 1]$", ha="left", va="center", 
           fontsize=12, weight="bold")
    
    # Annotations
    ax.text(2, 2.2, "Excitatory inputs\nsum linearly", 
           ha="center", fontsize=9, style="italic", color="#666666")
    ax.text(2, 1.5, "Inhibitory inputs\nsum linearly", 
           ha="center", fontsize=9, style="italic", color="#666666")
    ax.text(7.75, 2.2, "Shunting division\nnormalizes output", 
           ha="center", fontsize=9, style="italic", color="#666666")
    
    # Epsilon note
    ax.text(5, 1, r"$\epsilon$ = small constant to prevent division by zero", 
           ha="center", fontsize=10, style="italic", color="#666666",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0.5, 7)
    ax.axis("off")
    
    _save(fig, "dendritic_computation.png")


# Register new creators in __main__ if script run standalone
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
    
    create_einet_architecture()
    print("✓ EINet architecture diagram created")

    create_einet_classifier_dataflow()
    print("✓ EINetClassifier dataflow diagram created")

    create_dendrinet_1()
    print("✓ Dendrinet-1 architecture diagram created")

    create_dendrinet_2()
    print("✓ Dendrinet-2 architecture diagram created")

    create_input_network_architecture(); print("✓ Input network architecture")
    create_dendritic_processing(); print("✓ Dendritic processing")
    create_dendritic_computation(); print("✓ Dendritic computation")
    # Skipped result-based plots
    create_input_modes_complete(); print("✓ Input modes complete")
    
    print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}") 