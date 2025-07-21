"""
Final figure generation script for dendritic modeling documentation.
Generates high-quality, professional figures with improved designs.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import numpy as np
import os

# Create directories if they don't exist
output_dirs = ['figures/architecture', 'source/_static']
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Define colorblind-safe color palette (Okabe & Ito)
colors = {
    'vermillion': '#D55E00',  # Excitatory
    'sky_blue': '#56B4E9',    # Inhibitory
    'bluish_green': '#009E73', # Dendrites
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'orange': '#E69F00',
    'reddish_purple': '#CC79A7',
    'dark_gray': '#333333',
    'light_gray': '#CCCCCC',
    'medium_gray': '#999999',  # Lighter soma color
    'very_light_gray': '#E5E5E5',
}

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

def save_figure(fig, name, dpi=600):
    """Save figure in both locations with multiple formats."""
    for dir_path in output_dirs:
        fig.savefig(f'{dir_path}/{name}.png', dpi=dpi, transparent=True, bbox_inches='tight')
        fig.savefig(f'{dir_path}/{name}.svg', format='svg', transparent=True, bbox_inches='tight')
    print(f"Saved: {name}")

def create_improved_dendritic_neuron():
    """Create an improved dendritic neuron with better structure."""
    fig, ax = plt.subplots(figsize=(10, 12))

    # Draw soma (much lighter color)
    soma_center = (0, 0)
    soma_radius = 0.8
    soma = Circle(soma_center, soma_radius,
                  facecolor=colors['very_light_gray'],
                  edgecolor=colors['medium_gray'],
                  linewidth=2.5,
                  zorder=10)
    ax.add_patch(soma)

    # Draw nucleus (subtle)
    nucleus = Circle(soma_center, soma_radius * 0.35,
                    facecolor=colors['light_gray'],
                    alpha=0.5,
                    zorder=11)
    ax.add_patch(nucleus)

    # Helper function to draw a dendritic branch with tapering
    def draw_dendrite_branch(start, angle, length, width, n_segments=20, taper_rate=0.7, curvature=0.2):
        """Draw a curved dendritic branch with tapering."""
        points = []
        widths = []

        for i in range(n_segments + 1):
            t = i / n_segments
            # Add some natural curvature
            current_angle = angle + curvature * np.sin(3 * np.pi * t)
            x = start[0] + length * t * np.cos(current_angle)
            y = start[1] + length * t * np.sin(current_angle)
            points.append([x, y])
            widths.append(width * (1 - t * (1 - taper_rate)))

        return np.array(points), widths

    all_dendrite_points = []  # Store all dendrite points for synapse placement

    # Apical dendrites (top part of soma - main dendritic tree)
    apical_main_angle = -np.pi/2
    apical_start_x = soma_center[0] + soma_radius * 0.8 * np.cos(apical_main_angle)
    apical_start_y = soma_center[1] + soma_radius * 0.8 * np.sin(apical_main_angle)

    # Main apical trunk
    trunk_points, trunk_widths = draw_dendrite_branch(
        (apical_start_x, apical_start_y), apical_main_angle, 2.5, 0.2, curvature=0.1
    )
    all_dendrite_points.append(trunk_points)

    # Draw trunk
    segments = [[trunk_points[i], trunk_points[i + 1]] for i in range(len(trunk_points) - 1)]
    lc = LineCollection(segments, linewidths=np.array(trunk_widths[:-1]) * 100,
                       colors=colors['bluish_green'], capstyle='round', zorder=5)
    ax.add_collection(lc)

    # Apical tuft (branches from trunk)
    tuft_angles = [-np.pi/2 - 0.4, -np.pi/2 - 0.2, -np.pi/2, -np.pi/2 + 0.2, -np.pi/2 + 0.4]
    for angle_offset in tuft_angles:
        branch_start = trunk_points[-5]  # Branch from near the end
        branch_points, branch_widths = draw_dendrite_branch(
            branch_start, apical_main_angle + angle_offset, 2.0, 0.1, curvature=0.3
        )
        all_dendrite_points.append(branch_points)

        segments = [[branch_points[i], branch_points[i + 1]] for i in range(len(branch_points) - 1)]
        lc = LineCollection(segments, linewidths=np.array(branch_widths[:-1]) * 100,
                           colors=colors['bluish_green'], capstyle='round', zorder=5)
        ax.add_collection(lc)

        # Secondary branches
        if abs(angle_offset) < 0.3:
            for secondary_offset in [-0.2, 0.2]:
                sec_start = branch_points[len(branch_points)//2]
                sec_points, sec_widths = draw_dendrite_branch(
                    sec_start, apical_main_angle + angle_offset + secondary_offset, 1.0, 0.05
                )
                all_dendrite_points.append(sec_points)

                segments = [[sec_points[i], sec_points[i + 1]] for i in range(len(sec_points) - 1)]
                lc = LineCollection(segments, linewidths=np.array(sec_widths[:-1]) * 100,
                                   colors=colors['bluish_green'], capstyle='round', zorder=5)
                ax.add_collection(lc)

    # Oblique dendrites (side branches from apical trunk)
    for height_ratio in [0.3, 0.6]:
        trunk_idx = int(len(trunk_points) * height_ratio)
        for side in [-1, 1]:
            oblique_angle = apical_main_angle + side * np.pi/3
            oblique_points, oblique_widths = draw_dendrite_branch(
                trunk_points[trunk_idx], oblique_angle, 1.5, 0.08, curvature=0.2
            )
            all_dendrite_points.append(oblique_points)

            segments = [[oblique_points[i], oblique_points[i + 1]] for i in range(len(oblique_points) - 1)]
            lc = LineCollection(segments, linewidths=np.array(oblique_widths[:-1]) * 100,
                               colors=colors['bluish_green'], capstyle='round', zorder=5)
            ax.add_collection(lc)

    # Basal dendrites (distributed around lower part of soma)
    basal_angles = []
    # Left side
    for angle in np.linspace(np.pi - np.pi/4, np.pi/2 + np.pi/6, 3):
        basal_angles.append(angle)
    # Right side
    for angle in np.linspace(np.pi/2 - np.pi/6, np.pi/4, 3):
        basal_angles.append(angle)
    # Bottom
    for angle in np.linspace(np.pi/4, np.pi - np.pi/4, 4):
        basal_angles.append(angle)

    for angle in basal_angles:
        # Start from soma surface
        start_x = soma_center[0] + soma_radius * 0.9 * np.cos(angle)
        start_y = soma_center[1] + soma_radius * 0.9 * np.sin(angle)

        # Main basal dendrite
        length = np.random.uniform(1.8, 2.5)
        points, widths = draw_dendrite_branch(
            (start_x, start_y), angle, length, 0.12, curvature=0.15
        )
        all_dendrite_points.append(points)

        segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
        lc = LineCollection(segments, linewidths=np.array(widths[:-1]) * 100,
                           colors=colors['bluish_green'], capstyle='round', zorder=5)
        ax.add_collection(lc)

        # Add 1-2 branches
        n_branches = np.random.randint(1, 3)
        for _ in range(n_branches):
            branch_idx = np.random.randint(len(points)//3, 2*len(points)//3)
            branch_angle = angle + np.random.uniform(-0.4, 0.4)
            branch_length = np.random.uniform(0.8, 1.2)
            branch_points, branch_widths = draw_dendrite_branch(
                points[branch_idx], branch_angle, branch_length, 0.06
            )
            all_dendrite_points.append(branch_points)

            segments = [[branch_points[i], branch_points[i + 1]] for i in range(len(branch_points) - 1)]
            lc = LineCollection(segments, linewidths=np.array(branch_widths[:-1]) * 100,
                               colors=colors['bluish_green'], capstyle='round', zorder=5)
            ax.add_collection(lc)

    # Add synapses (smaller and distributed along dendrites)
    np.random.seed(42)
    synapse_size = 0.03  # Much smaller synapses

    # Place synapses along dendrite branches
    for dendrite_points in all_dendrite_points:
        if len(dendrite_points) > 5:  # Only on longer branches
            n_synapses = np.random.randint(2, 6)
            for _ in range(n_synapses):
                # Pick a point along the dendrite (avoiding the base)
                idx = np.random.randint(3, len(dendrite_points) - 1)

                # Add small offset perpendicular to dendrite
                if idx > 0:
                    direction = dendrite_points[idx] - dendrite_points[idx-1]
                    direction = direction / np.linalg.norm(direction)
                    perpendicular = np.array([-direction[1], direction[0]])
                    offset = perpendicular * np.random.uniform(-0.05, 0.05)
                else:
                    offset = np.array([0, 0])

                syn_x = dendrite_points[idx][0] + offset[0]
                syn_y = dendrite_points[idx][1] + offset[1]

                # Determine synapse type
                if np.random.random() < 0.65:
                    color = colors['vermillion']  # Excitatory
                    marker = 'o'
                else:
                    color = colors['sky_blue']    # Inhibitory
                    marker = 's'

                ax.scatter(syn_x, syn_y, s=synapse_size*1000, c=color,
                          marker=marker, edgecolor='none',
                          zorder=8, alpha=0.9)

    # Draw axon (from bottom of soma)
    axon_start_angle = np.pi/2  # Bottom
    axon_start_x = soma_center[0] + soma_radius * 0.9 * np.cos(axon_start_angle)
    axon_start_y = soma_center[1] + soma_radius * 0.9 * np.sin(axon_start_angle)

    # Axon initial segment
    axon_points = [(axon_start_x, axon_start_y)]
    for i in range(15):
        axon_points.append((axon_start_x, axon_start_y - (i+1)*0.15))

    ax.plot([p[0] for p in axon_points], [p[1] for p in axon_points],
           color=colors['dark_gray'], linewidth=3.5, zorder=3)

    # Axon branches
    branch_point = axon_points[8]
    for side in [-1, 1]:
        branch_x = [branch_point[0]]
        branch_y = [branch_point[1]]
        for j in range(12):
            branch_x.append(branch_x[-1] + side * 0.1)
            branch_y.append(branch_y[-1] - 0.08)
        ax.plot(branch_x, branch_y, color=colors['dark_gray'],
               linewidth=2.5, zorder=3)

        # Terminal with synaptic boutons
        for k in range(3):
            terminal_x = branch_x[-1] + side * k * 0.15
            terminal_y = branch_y[-1] - k * 0.1
            ax.scatter(terminal_x, terminal_y, s=120, c=colors['vermillion'],
                      marker='o', edgecolor=colors['dark_gray'], linewidth=0.5, zorder=9)

    # Add subtle labels
    ax.text(0, 0, 'Soma', fontsize=11, ha='center', va='center',
            color=colors['dark_gray'], zorder=12)
    ax.text(0, 5.2, 'Apical Tuft', fontsize=10, ha='center', va='bottom',
            style='italic', color=colors['dark_gray'])
    ax.text(-2.5, -0.5, 'Basal\nDendrites', fontsize=10, ha='center', va='center',
            style='italic', color=colors['dark_gray'])
    ax.text(1.5, -3.2, 'Axon', fontsize=10, ha='left', va='center',
            style='italic', color=colors['dark_gray'])

    # Legend
    exc_patch = mpatches.Patch(color=colors['vermillion'], label='Excitatory Synapse')
    inh_patch = mpatches.Patch(color=colors['sky_blue'], label='Inhibitory Synapse')
    ax.legend(handles=[exc_patch, inh_patch],
              loc='upper right', frameon=True, fancybox=True,
              framealpha=0.9, shadow=True, borderpad=0.5, fontsize=12)

    # Style
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Pyramidal Neuron with Dendritic Structure',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    save_figure(fig, 'dendritic_neuron_improved')
    plt.close()

def create_accurate_block_linear():
    """Create accurate block linear architecture showing dendrites connecting to matrix blocks."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Parameters
    n_inputs = 12
    n_outputs = 3
    block_size = n_inputs // n_outputs  # 4 inputs per output

    # Positions
    input_x = 1
    matrix_center_x = 5.5
    output_x = 10

    # Draw input layer
    input_y_start = 6
    input_spacing = 0.6
    input_positions = []

    for i in range(n_inputs):
        y = input_y_start - i * input_spacing
        # Input neuron
        circle = Circle((input_x, y), 0.15,
                       facecolor=colors['very_light_gray'],
                       edgecolor=colors['dark_gray'],
                       linewidth=1.5)
        ax.add_patch(circle)
        input_positions.append((input_x, y))
        ax.text(input_x - 0.5, y, f'x_{i}', fontsize=9, ha='right', va='center')

    # Draw weight matrix
    matrix_width = 3
    matrix_height = 5
    matrix_left = matrix_center_x - matrix_width/2
    matrix_bottom = 1

    # Matrix background
    matrix_bg = Rectangle((matrix_left, matrix_bottom), matrix_width, matrix_height,
                         facecolor='white', edgecolor=colors['dark_gray'],
                         linewidth=2)
    ax.add_patch(matrix_bg)

    # Draw the block diagonal structure
    block_colors = [colors['vermillion'], colors['bluish_green'], colors['orange']]

    for block_idx in range(n_outputs):
        # Calculate block position in matrix
        block_x = matrix_left + (block_idx * matrix_width / n_inputs) * block_size
        block_y = matrix_bottom + (n_outputs - 1 - block_idx) * matrix_height / n_outputs
        block_w = matrix_width / n_inputs * block_size
        block_h = matrix_height / n_outputs

        # Draw block
        block = Rectangle((block_x, block_y), block_w, block_h,
                         facecolor=block_colors[block_idx], alpha=0.3,
                         edgecolor=block_colors[block_idx], linewidth=2)
        ax.add_patch(block)

        # Add weight indicators (dots)
        for i in range(block_size):
            for j in range(1):
                dot_x = block_x + (i + 0.5) * block_w / block_size
                dot_y = block_y + block_h / 2
                ax.scatter(dot_x, dot_y, s=60, c=block_colors[block_idx],
                          alpha=0.8, edgecolor='none', zorder=10)

    # Draw output neurons with dendrites
    output_y_start = 5.5
    output_spacing = 2

    for out_idx in range(n_outputs):
        y = output_y_start - out_idx * output_spacing

        # Draw soma
        soma = Circle((output_x, y), 0.25,
                     facecolor=colors['very_light_gray'],
                     edgecolor=colors['dark_gray'],
                     linewidth=2)
        ax.add_patch(soma)
        ax.text(output_x + 0.5, y, f'y_{out_idx}', fontsize=11, ha='left', va='center',
               fontweight='bold')

        # Draw main dendrite extending left
        main_dendrite_end = output_x - 2
        ax.plot([output_x - 0.25, main_dendrite_end], [y, y],
               color=block_colors[out_idx], linewidth=4, solid_capstyle='round', zorder=5)

        # Draw dendritic branches (one for each input in the block)
        branch_positions = []
        branch_spread = 0.8
        for i in range(block_size):
            # Calculate branch position
            branch_y = y + (i - (block_size-1)/2) * branch_spread / (block_size-1)
            branch_end_x = main_dendrite_end - 0.8

            # Draw branch
            ax.plot([main_dendrite_end, branch_end_x], [y, branch_y],
                   color=block_colors[out_idx], linewidth=2.5, solid_capstyle='round', zorder=5)

            # Add synapse at branch end
            ax.scatter(branch_end_x, branch_y, s=80, c=block_colors[out_idx],
                      marker='o', edgecolor=colors['dark_gray'], linewidth=0.5, zorder=11)

            branch_positions.append((branch_end_x, branch_y))

        # Connect inputs to matrix and matrix to dendrites
        for i in range(block_size):
            input_idx = out_idx * block_size + i
            if input_idx < n_inputs:
                # Input to matrix connection
                input_pos = input_positions[input_idx]

                # Matrix position
                block_x = matrix_left + (out_idx * matrix_width / n_inputs) * block_size
                block_y = matrix_bottom + (n_outputs - 1 - out_idx) * matrix_height / n_outputs
                matrix_x = block_x + (i + 0.5) * (matrix_width / n_inputs * block_size) / block_size
                matrix_y = block_y + matrix_height / n_outputs / 2

                # Draw connection from input to matrix
                ax.plot([input_pos[0] + 0.15, matrix_x], [input_pos[1], matrix_y],
                       color=colors['medium_gray'], linewidth=1, alpha=0.5, linestyle='--', zorder=1)

                # Draw connection from matrix to dendrite branch
                branch_pos = branch_positions[i]
                ax.plot([matrix_x, branch_pos[0]], [matrix_y, branch_pos[1]],
                       color=block_colors[out_idx], linewidth=1.5, alpha=0.7, zorder=2)

    # Add annotations
    ax.text(input_x, 7, 'Input Layer', fontsize=13, ha='center', fontweight='bold')
    ax.text(matrix_center_x, 7, 'Block Diagonal\nWeight Matrix', fontsize=13, ha='center', fontweight='bold')
    ax.text(output_x, 7, 'Output Neurons\nwith Dendrites', fontsize=13, ha='center', fontweight='bold')

    # Add mathematical notation
    ax.text(matrix_center_x, 0.2,
           r'$W \in \mathbb{R}^{n_{out} \times n_{in}}$' + '\n' +
           f'{n_outputs} blocks × {block_size} weights each\n' +
           'Each dendrite receives inputs from one block',
           fontsize=10, ha='center', style='italic',
           bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['very_light_gray'],
                    edgecolor=colors['medium_gray'], linewidth=1))

    # Add block structure explanation
    for i in range(n_outputs):
        ax.text(matrix_left - 0.3,
               matrix_bottom + (n_outputs - 0.5 - i) * matrix_height / n_outputs,
               f'Block {i}', fontsize=9, ha='right', va='center',
               color=block_colors[i], fontweight='bold')

    # Style
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    ax.set_title('Block Linear Layer: Dendrites Connect to Diagonal Weight Blocks',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    save_figure(fig, 'block_linear_accurate')
    plt.close()

# Additional figures from the original script
def create_network_architecture():
    """Create the E/I network architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Layer positions
    layers = [
        {'name': 'Input', 'y': 0.5, 'neurons': 6},
        {'name': 'E/I Layer 1', 'y': 2.5, 'exc': 4, 'inh': 2},
        {'name': 'E/I Layer 2', 'y': 4.5, 'exc': 4, 'inh': 2},
        {'name': 'Output', 'y': 6.5, 'neurons': 3}
    ]

    # Draw layers
    all_neurons = []

    for layer in layers:
        if 'neurons' in layer:  # Input/Output layers
            x_positions = np.linspace(2, 8, layer['neurons'])
            for x in x_positions:
                neuron = Circle((x, layer['y']), 0.15,
                              facecolor=colors['light_gray'],
                              edgecolor=colors['dark_gray'],
                              linewidth=2)
                ax.add_patch(neuron)
                all_neurons.append({'pos': (x, layer['y']), 'type': 'neutral', 'layer': layer['name']})
        else:  # E/I layers
            # Excitatory neurons
            exc_x = np.linspace(2, 5.5, layer['exc'])
            for x in exc_x:
                neuron = Circle((x, layer['y']), 0.18,
                              facecolor=colors['vermillion'],
                              edgecolor=colors['dark_gray'],
                              linewidth=2, alpha=0.8)
                ax.add_patch(neuron)
                all_neurons.append({'pos': (x, layer['y']), 'type': 'exc', 'layer': layer['name']})

            # Inhibitory neurons
            inh_x = np.linspace(6.5, 8, layer['inh'])
            for x in inh_x:
                neuron = Circle((x, layer['y']), 0.18,
                              facecolor=colors['sky_blue'],
                              edgecolor=colors['dark_gray'],
                              linewidth=2, alpha=0.8)
                ax.add_patch(neuron)
                all_neurons.append({'pos': (x, layer['y']), 'type': 'inh', 'layer': layer['name']})

    # Draw connections (simplified for clarity)
    np.random.seed(42)

    # Connect layers
    for i, layer in enumerate(layers[:-1]):
        next_layer = layers[i + 1]

        # Get neurons in current and next layer
        current_neurons = [n for n in all_neurons if n['layer'] == layer['name']]
        next_neurons = [n for n in all_neurons if n['layer'] == next_layer['name']]

        # Sample connections
        for curr in current_neurons[::2]:  # Every other neuron for clarity
            # Connect to some neurons in next layer
            targets = np.random.choice(next_neurons, size=min(3, len(next_neurons)), replace=False)

            for target in targets:
                # Determine connection type
                if curr['type'] == 'inh' or target['type'] == 'inh':
                    color = colors['sky_blue']
                    style = '-'
                    alpha = 0.4
                else:
                    color = colors['vermillion']
                    style = '-'
                    alpha = 0.3

                # Draw connection
                ax.plot([curr['pos'][0], target['pos'][0]],
                       [curr['pos'][1] + 0.18, target['pos'][1] - 0.18],
                       color=color, linestyle=style, linewidth=1.5, alpha=alpha)

    # Add labels
    for layer in layers:
        ax.text(10, layer['y'], layer['name'], fontsize=14, va='center', fontweight='bold')

    # Add legend
    exc_patch = mpatches.Patch(color=colors['vermillion'], label='Excitatory', alpha=1.0)
    inh_patch = mpatches.Patch(color=colors['sky_blue'], label='Inhibitory', alpha=1.0)
    ax.legend(handles=[exc_patch, inh_patch], loc='upper left',
              frameon=True, fancybox=True, shadow=True)

    # Add title and labels
    ax.set_title('Excitation-Inhibition Network Architecture', fontsize=18, fontweight='bold', pad=20)
    ax.text(5, -0.5, 'Information Flow →', fontsize=12, ha='center', style='italic')

    # Style
    ax.set_xlim(0, 11)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    plt.tight_layout()
    save_figure(fig, 'network_architecture')
    plt.close()

def create_shunting_inhibition():
    """Create shunting inhibition mechanism diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Circuit diagram
    ax1.set_title('Shunting Inhibition Circuit', fontsize=14, fontweight='bold')

    # Dendrite
    dendrite_x = [2, 5]
    dendrite_y = [3, 3]
    ax1.plot(dendrite_x, dendrite_y, color=colors['bluish_green'], linewidth=8, solid_capstyle='round')

    # Excitatory synapse
    exc_x, exc_y = 3, 3.5
    ax1.scatter(exc_x, exc_y, s=200, c=colors['vermillion'], marker='o',
               edgecolor='black', linewidth=2, zorder=10)
    ax1.plot([exc_x, exc_x], [exc_y, 3], color='black', linewidth=2)
    ax1.text(exc_x - 0.3, exc_y + 0.5, 'Excitatory\nInput', fontsize=10, ha='center')

    # Inhibitory synapse
    inh_x, inh_y = 4, 2.5
    ax1.scatter(inh_x, inh_y, s=200, c=colors['sky_blue'], marker='s',
               edgecolor='black', linewidth=2, zorder=10)
    ax1.plot([inh_x, inh_x], [inh_y, 3], color='black', linewidth=2)
    ax1.text(inh_x + 0.3, inh_y - 0.5, 'Inhibitory\nInput', fontsize=10, ha='center')

    # Soma
    soma = Circle((6, 3), 0.5, facecolor=colors['light_gray'],
                 edgecolor=colors['dark_gray'], linewidth=2)
    ax1.add_patch(soma)
    ax1.text(6, 3, 'Soma', fontsize=10, ha='center', va='center')

    # Current flow arrows
    ax1.annotate('', xy=(5, 3), xytext=(3.2, 3),
                arrowprops=dict(arrowstyle='->', color=colors['vermillion'], lw=2))
    ax1.text(4, 3.2, 'I_exc', fontsize=10, ha='center', color=colors['vermillion'])

    ax1.set_xlim(1, 7)
    ax1.set_ylim(1, 5)
    ax1.axis('off')

    # Right: Voltage dynamics
    ax2.set_title('Effect on Membrane Potential', fontsize=14, fontweight='bold')

    t = np.linspace(0, 100, 1000)

    # Without inhibition
    v_no_inh = -70 + 20 * np.exp(-(t - 30)**2 / 100)

    # With shunting inhibition
    v_with_inh = -70 + 10 * np.exp(-(t - 30)**2 / 100)

    ax2.plot(t, v_no_inh, color=colors['vermillion'], linewidth=2.5,
            label='Without inhibition')
    ax2.plot(t, v_with_inh, color=colors['blue'], linewidth=2.5,
            label='With shunting inhibition')

    # Threshold
    ax2.axhline(y=-50, color='black', linestyle='--', alpha=0.5)
    ax2.text(80, -48, 'Threshold', fontsize=10)

    # Annotations
    ax2.annotate('Reduced amplitude', xy=(30, -60), xytext=(50, -55),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))

    ax2.set_xlabel('Time (ms)', fontsize=14)
    ax2.set_ylabel('Membrane Potential (mV)', fontsize=14)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-75, -45)
    ax2.legend(loc='upper right', frameon=True, fancybox=True,
           framealpha=0.9, shadow=True, fontsize=12)
    ax2.grid(True, alpha=0.4, linestyle='--')

    plt.tight_layout()
    save_figure(fig, 'shunting_inhibition')
    plt.close()

def main():
    """Generate all figures."""
    print("Generating improved figures...")

    # Generate the improved figures
    create_improved_dendritic_neuron()
    create_accurate_block_linear()

    # Generate other figures
    create_network_architecture()
    create_shunting_inhibition()

    print("\nAll figures generated successfully!")
    print(f"Figures saved to: {', '.join(output_dirs)}")

if __name__ == "__main__":
    main()
