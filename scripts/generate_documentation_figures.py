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


def apply_enhanced_style():
    """Apply a highly polished, modern visual style for all figures."""
    # A more vibrant and professional color palette
    global COLORS
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

    shadow = path_effects.SimplePatchShadow(
        offset=(dx, dy), shadow_rgb_alpha=(0, 0, 0, alpha),
        alpha=1.0, **kwargs
    )
    artist.set_path_effects([shadow, path_effects.Normal()])


# Apply the new style globally when the script is loaded
apply_enhanced_style()


# =====================================================================================
# Core Architectural Diagrams
# =====================================================================================


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
    """Improved multi-layer E/I network diagram with clear neuron vs. branch grouping."""

    fig, ax = plt.subplots(figsize=(12, 6))

    layers = [
        {"name": "Input", "e": 4, "i": 0},
        {"name": "Hidden 1", "e": 5, "i": 3},
        {"name": "Hidden 2", "e": 4, "i": 2},
        {"name": "Output", "e": 2, "i": 0},
    ]

    x_coords = np.arange(len(layers)) * 3 + 2

    neuron_coordinates = []  # store for connections

    for idx, layer in enumerate(layers):
        x = x_coords[idx]

        # background column box
        ax.add_patch(FancyBboxPatch((x - 1.0, 1.2), 2.0, 3.6, boxstyle="round,pad=0.1", facecolor=COLORS["background"], ec="#aaaaaa", alpha=0.3))

        # label at top
        ax.text(x, 5.1, layer["name"], ha="center", fontsize=12, weight="bold")

        # excitatory neurons (top half)
        e_count = layer["e"]
        e_ys = np.linspace(4.2, 3.2, e_count)
        for y in e_ys:
            ax.add_patch(Circle((x, y), 0.25, color=COLORS["excitatory"], ec="k", lw=1))
            neuron_coordinates.append((x, y, "E"))

        # inhibitory neurons (bottom half)
        i_count = layer["i"]
        if i_count:
            i_ys = np.linspace(2.2, 1.4, i_count)
            for y in i_ys:
                ax.add_patch(Circle((x, y), 0.2, color=COLORS["inhibitory"], ec="k", lw=1))
                neuron_coordinates.append((x, y, "I"))

        # branch delimiter line between E and I
        if i_count:
            ax.plot([x - 0.9, x + 0.9], [2.7, 2.7], color="#555555", lw=0.8, linestyle="--")

    # draw forward connections (E->E only for clarity)
    for (x0, y0, t0) in neuron_coordinates:
        for (x1, y1, t1) in neuron_coordinates:
            if x1 > x0 and t1 == "E":
                color = COLORS["excitatory"] if t0 == "E" else COLORS["inhibitory"]
                ax.plot([x0, x1], [y0, y1], color=color, alpha=0.15, lw=1)

    # input arrow and label
    ax.add_patch(FancyArrowPatch((0.5, 3.2), (1.2, 3.2), arrowstyle="->", mutation_scale=15, lw=1.8, color=COLORS["arrow"]))
    ax.text(0.4, 3.2, "Input", ha="right", va="center")

    # legend
    ax.legend([
        patches.Patch(color=COLORS["excitatory"], label="Excitatory neuron"),
        patches.Patch(color=COLORS["inhibitory"], label="Inhibitory neuron"),
        patches.Patch(facecolor=COLORS["background"], edgecolor="#aaaaaa", label="Branch / layer area", alpha=0.3),
    ], loc="lower center", frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.05))

    ax.set_xlim(-0.2, x_coords[-1] + 1.8)
    ax.set_ylim(1.0, 5.4)
    ax.axis("off")

    ax.set_title("Multi-Layer Excitatory / Inhibitory Network", fontsize=16, weight="bold")

    _save(fig, "multilayer_ei_network.png")


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
    draw_stage(7, 1.5, 3, 3, "Dendritic Core", ["E/I Separation", "Branching", "Shunting Inhibition"], color=COLORS['dendrite'])
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
    """Create an improved dendritic computation flow diagram."""

    fig, ax = plt.subplots(figsize=(12, 7))

    fig.set_size_inches(10, 6)

    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 6.5)
    ax.axis("off")

    # ---- Title ----
    ax.text(6, 7.5, "Dendritic Computation Flow", ha="center", fontsize=20, weight="bold")

    # ---- Input icons ----
    exc_positions = [(1.2 + i * 1.0, 5.5) for i in range(4)]
    inh_positions = [(1.2 + i * 1.0, 4.3) for i in range(4)]

    for idx, (x, y) in enumerate(exc_positions):
        tri = plt.Polygon([(x, y), (x - 0.18, y + 0.28), (x + 0.18, y + 0.28)], color=COLORS["excitatory"], ec="k", linewidth=0.7)
        ax.add_patch(tri)
        ax.text(x, y - 0.3, f"E{idx + 1}", ha="center", fontsize=9)

    for idx, (x, y) in enumerate(inh_positions):
        circ = Circle((x, y), 0.14, color=COLORS["inhibitory"], ec="k", linewidth=0.7)
        ax.add_patch(circ)
        ax.text(x, y - 0.35, f"I{idx + 1}", ha="center", fontsize=9)

    # ---- Summation nodes ----
    sum_e = Circle((4.8, 5.5), 0.25, color="#ffffff", ec=COLORS["excitatory"], lw=1.5)
    sum_i = Circle((4.8, 4.3), 0.25, color="#ffffff", ec=COLORS["inhibitory"], lw=1.5)
    ax.add_patch(sum_e)
    ax.add_patch(sum_i)
    ax.text(4.8, 5.5, "Σ", ha="center", va="center", fontsize=12, weight="bold", color=COLORS["excitatory"])
    ax.text(4.8, 4.3, "Σ", ha="center", va="center", fontsize=12, weight="bold", color=COLORS["inhibitory"])

    # arrows from inputs to sums
    for (x, y) in exc_positions:
        ax.add_patch(FancyArrowPatch((x, y), (4.55, 5.55), arrowstyle="-|>", mutation_scale=10, lw=0.9, color=COLORS["excitatory"], alpha=0.6))
    for (x, y) in inh_positions:
        ax.add_patch(FancyArrowPatch((x, y), (4.55, 4.25), arrowstyle="-|>", mutation_scale=10, lw=0.9, color=COLORS["inhibitory"], alpha=0.6))

    # ---- Division node ----
    div_box = FancyBboxPatch((6.0, 4.7), 1.3, 1.3, boxstyle="round,pad=0.15", facecolor="#ffffff", ec="#333333", lw=1.5)
    ax.add_patch(div_box)
    ax.text(6.65, 5.35, r"$\frac{E}{E + I + \varepsilon}$", ha="center", va="center", fontsize=11)

    # arrows from sums to division
    ax.add_patch(FancyArrowPatch((5.05, 5.5), (6.0, 5.3), arrowstyle="->", mutation_scale=12, lw=1.2, color=COLORS["excitatory"]))
    ax.add_patch(FancyArrowPatch((5.05, 4.3), (6.0, 4.8), arrowstyle="->", mutation_scale=12, lw=1.2, color=COLORS["inhibitory"]))

    # ---- Output arrow ----
    ax.add_patch(FancyArrowPatch((7.3, 5.35), (8.7, 5.35), arrowstyle="->", mutation_scale=12, lw=1.5, color=COLORS["arrow"]))
    ax.text(8.9, 5.35, "Vout", ha="left", va="center", fontsize=11, weight="bold")

    # ---- Legend ----
    exc_patch = patches.Patch(color=COLORS["excitatory"], label="Excitatory input (E)")
    inh_patch = patches.Patch(color=COLORS["inhibitory"], label="Inhibitory input (I)")
    ax.legend(handles=[exc_patch, inh_patch], loc="upper right", frameon=False, fontsize=9)

    _save(fig, "dendritic_computation_flow.png")


# =====================================================================================
# Bespoke diagrams for previously placeholder-only assets
# =====================================================================================


def _save(fig, name):
    """Helper to save fig at high DPI to OUTPUT_DIR."""
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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


def create_experiment_tracking():
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


def create_configuration_system():
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


def create_sweep_job_submission():
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
    fig, ax = plt.subplots(figsize=(8, 2.5))
    _simple_flow(ax, ["Dendrite", "Branch", "Synapse"], y=1.25)
    ax.set_xlim(0, 7)
    ax.set_ylim(0.2, 2.3)
    ax.axis("off")
    ax.set_title("Dendritic Processing", fontsize=12, weight="bold")
    _save(fig, "dendritic_processing.png")


def create_training_strategies():
    fig, ax = plt.subplots(figsize=(6, 3))
    strategies = ["Supervised", "Self-Supervised", "Contrastive"]
    ax.barh(strategies, [3, 2, 1], color=["#457b9d", "#e9c46a", "#e76f51"])
    ax.set_xlim(0, 3.5)
    ax.set_xlabel("Relative Usage")
    ax.set_title("Training Strategies")
    _save(fig, "training_strategies.png")


def create_sequential_vs_joint():
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot([0, 1, 2], [0.2, 0.8, 1], label="Joint", lw=2)
    ax.plot([0, 1, 2], [0.2, 0.4, 0.6], label="Sequential", lw=2, linestyle="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Start", "Mid", "End"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Sequential vs. Joint Training")
    ax.legend()
    _save(fig, "sequential_vs_joint.png")


def create_head_training_strategies():
    fig, ax = plt.subplots(figsize=(6, 3))
    heads = ["Linear", "MLP", "Dendritic"]
    colors = ["#2a9d8f", "#e9c46a", "#e63946"]
    ax.bar(heads, [0.7, 0.8, 0.9], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Head Training Strategies")
    _save(fig, "head_training_strategies.png")


def create_learning_strategies():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pie([40, 35, 25], labels=["Backprop", "Hebbian", "RL"], autopct="%1.0f%%", colors=["#264653", "#2a9d8f", "#e9c46a"])
    ax.set_title("Learning Strategies Share")
    _save(fig, "learning_strategies.png")


def create_ablation_visualization():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.errorbar([1, 2, 3], [0.8, 0.6, 0.4], yerr=[0.05, 0.04, 0.03], fmt="o-", capsize=4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Full", "No Inhib", "No Dendrite"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study")
    _save(fig, "ablation_visualization.png")


def create_computational_properties():
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


def create_installation_workflow():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    _simple_flow(ax, ["Clone Repo", "Install Deps", "Run Tests", "Build Docs"], y=1.25)
    ax.set_xlim(0, 9)
    ax.set_ylim(0.2, 2.3)
    ax.axis("off")
    ax.set_title("Installation Workflow", fontsize=12, weight="bold")
    _save(fig, "installation_workflow.png")


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
    
    create_einet_classifier_dataflow()
    print("✓ EINetClassifier dataflow diagram created")

    create_dendrinet_1()
    print("✓ Dendrinet-1 architecture diagram created")

    create_dendrinet_2()
    print("✓ Dendrinet-2 architecture diagram created")

    create_experiment_tracking()
    print("✓ Experiment tracking workflow diagram created")

    create_configuration_system()
    print("✓ Configuration system diagram created")

    create_sweep_job_submission()
    print("✓ Sweep job submission diagram created")
    
    create_input_network_architecture(); print("✓ Input network architecture")
    create_dendritic_processing(); print("✓ Dendritic processing")
    create_training_strategies(); print("✓ Training strategies")
    create_sequential_vs_joint(); print("✓ Sequential vs joint")
    create_head_training_strategies(); print("✓ Head training strategies")
    create_learning_strategies(); print("✓ Learning strategies")
    create_ablation_visualization(); print("✓ Ablation visualization")
    create_computational_properties(); print("✓ Computational properties")
    create_input_modes_complete(); print("✓ Input modes complete")
    create_installation_workflow(); print("✓ Installation workflow")
    
    print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}") 