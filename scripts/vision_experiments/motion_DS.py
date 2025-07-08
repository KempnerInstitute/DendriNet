"""Demo: Motion direction selectivity using spatiotemporal E/I patterns.

Shows how dendritic neurons can implement Reichardt-like motion detectors
through asymmetric E/I organization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d
from dendritic_modeling.networks.factories.vision_units import create_motion_unit
from dendritic_modeling.datasets.stimuli.bars import create_moving_bar
from dendritic_modeling.utils.visual_metrics import direction_selectivity_index


def test_direction_selectivity(neuron: DendriteConv2d, directions: np.ndarray) -> np.ndarray:
    """Test neuron response to bars moving in different directions.
    
    Parameters
    ----------
    neuron : DendriteConv2d
        Motion-sensitive neuron
    directions : np.ndarray
        Motion directions to test (degrees)
        
    Returns
    -------
    responses : np.ndarray
        Peak response for each direction
    """
    responses = []
    
    for direction in directions:
        # Create moving bar stimulus
        movie = create_moving_bar(
            size=64,
            bar_width=4,
            bar_length=32,
            direction=direction,
            speed=2.0,
            n_frames=20
        )
        
        # Process each frame and get max response
        frame_responses = []
        for frame in movie:
            frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                response = neuron(frame_tensor)
                frame_responses.append(response.max().item())
                
        responses.append(np.max(frame_responses))
        
    return np.array(responses)


def main():
    """Run motion direction selectivity demo."""
    # Create motion detectors for different preferred directions
    preferred_directions = ["rightward", "leftward", "upward", "downward"]
    direction_angles = [0, 180, 270, 90]  # Corresponding angles
    
    neurons = []
    for direction in preferred_directions:
        neuron = create_motion_unit(direction=direction)
        neurons.append(neuron)
    
    # Test all directions
    test_directions = np.linspace(0, 360, 37)[:-1]  # 0-350 degrees
    
    # Get direction tuning curves
    tuning_curves = []
    for neuron in neurons:
        responses = test_direction_selectivity(neuron, test_directions)
        tuning_curves.append(responses)
        
    tuning_curves = np.array(tuning_curves)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Show receptive fields
    for i, (neuron, direction) in enumerate(zip(neurons, preferred_directions)):
        with torch.no_grad():
            E = neuron.weight_E[0, 0].numpy()
            I = neuron.weight_I[0, 0].numpy()
            rf = E - I
            
        im = axes[0, i].imshow(rf, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, i].set_title(f'{direction.capitalize()} detector')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)
    
    # Plot direction tuning curves
    ax = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(preferred_directions)))
    
    for i, (direction, curve, color, angle) in enumerate(
        zip(preferred_directions, tuning_curves, colors, direction_angles)):
        ax.plot(test_directions, curve, 'o-', color=color,
                label=f'{direction} ({angle}°)', linewidth=2, markersize=4)
        
        # Mark preferred direction
        max_idx = np.argmax(curve)
        ax.axvline(test_directions[max_idx], color=color, linestyle='--', alpha=0.5)
        
    ax.set_xlabel('Motion Direction (degrees)', fontsize=12)
    ax.set_ylabel('Peak Response', fontsize=12)
    ax.set_title('Direction Tuning Curves', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 370)
    
    # Polar plot
    ax_polar = plt.subplot(2, 4, 6, projection='polar')
    theta = np.deg2rad(np.concatenate([test_directions, [360]]))
    
    for i, (direction, curve, color) in enumerate(
        zip(preferred_directions, tuning_curves, colors)):
        # Normalize and complete circle
        curve_norm = curve / curve.max()
        curve_complete = np.concatenate([curve_norm, [curve_norm[0]]])
        
        ax_polar.plot(theta, curve_complete, 'o-', color=color,
                     label=direction, linewidth=2, markersize=3)
        ax_polar.fill(theta, curve_complete, color=color, alpha=0.2)
        
    ax_polar.set_title('Polar Direction Plot', fontsize=14, pad=20)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(-1)
    
    # Calculate and display DSI
    ax_text = axes[1, 2]
    ax_text.axis('off')
    
    dsi_text = "Direction Selectivity Index (DSI):\n\n"
    for i, (direction, curve) in enumerate(zip(preferred_directions, tuning_curves)):
        dsi = direction_selectivity_index(curve, test_directions)
        dsi_text += f"{direction:10s}: DSI = {dsi:.3f}\n"
        
    ax_text.text(0.1, 0.5, dsi_text, fontsize=12, verticalalignment='center',
                 family='monospace')
    
    # Show temporal dynamics
    ax_temporal = axes[1, 3]
    
    # Create rightward moving bar
    movie = create_moving_bar(
        size=64,
        bar_width=4,
        bar_length=32,
        direction=0,  # Rightward
        speed=2.0,
        n_frames=30
    )
    
    # Test with rightward and leftward detectors
    rightward_neuron = neurons[0]
    leftward_neuron = neurons[1]
    
    rightward_responses = []
    leftward_responses = []
    
    for frame in movie:
        frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            r_resp = rightward_neuron(frame_tensor)
            l_resp = leftward_neuron(frame_tensor)
            rightward_responses.append(r_resp.max().item())
            leftward_responses.append(l_resp.max().item())
            
    time = np.arange(len(rightward_responses))
    ax_temporal.plot(time, rightward_responses, 'b-', label='Rightward detector', linewidth=2)
    ax_temporal.plot(time, leftward_responses, 'r-', label='Leftward detector', linewidth=2)
    ax_temporal.set_xlabel('Time (frames)', fontsize=11)
    ax_temporal.set_ylabel('Response', fontsize=11)
    ax_temporal.set_title('Response to Rightward Motion', fontsize=12)
    ax_temporal.legend()
    ax_temporal.grid(True, alpha=0.3)
    
    plt.suptitle('Motion Direction Selectivity via Spatiotemporal E/I Patterns', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Demonstrate speed tuning
    print("\nTesting speed tuning...")
    speeds = np.linspace(0.5, 5, 10)
    speed_responses = []
    
    test_neuron = neurons[0]  # Rightward detector
    for speed in speeds:
        movie = create_moving_bar(
            size=64,
            bar_width=4,
            bar_length=32,
            direction=0,
            speed=speed,
            n_frames=20
        )
        
        frame_responses = []
        for frame in movie:
            frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                response = test_neuron(frame_tensor)
                frame_responses.append(response.max().item())
                
        speed_responses.append(np.max(frame_responses))
        
    # Plot speed tuning
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(speeds, speed_responses, 'ko-', linewidth=2, markersize=8)
    ax.set_xlabel('Speed (pixels/frame)', fontsize=12)
    ax.set_ylabel('Peak Response', fontsize=12)
    ax.set_title('Speed Tuning of Motion Detector', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark optimal speed
    opt_idx = np.argmax(speed_responses)
    ax.axvline(speeds[opt_idx], color='r', linestyle='--', alpha=0.7)
    ax.text(speeds[opt_idx] + 0.1, speed_responses[opt_idx] * 0.9,
            f'Optimal: {speeds[opt_idx]:.1f} px/frame', fontsize=10)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 