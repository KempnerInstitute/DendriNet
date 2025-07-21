"""Demo: Gabor orientation filter using dendritic E/I patterns.

Shows orientation selectivity emerging from push-pull E/I organization,
similar to V1 simple cells.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.vision_factories.vision_units import create_gabor_unit
from dendritic_modeling.datasets.stimuli.gratings import create_static_grating
from dendritic_modeling.analysis.metrics.visual_metrics import orientation_selectivity_index


def test_orientation_tuning(neuron: DendriteConv2d, test_orientations: np.ndarray) -> np.ndarray:
    """Test neuron response to gratings at different orientations.
    
    Parameters
    ----------
    neuron : DendriteConv2d
        Dendritic neuron to test
    test_orientations : np.ndarray
        Orientations to test (degrees)
        
    Returns
    -------
    responses : np.ndarray
        Response magnitude for each orientation
    """
    responses = []

    for ori in test_orientations:
        # Create test grating
        grating = create_static_grating(
            size=64,
            orientation=ori,
            spatial_frequency=8,
            contrast=1.0
        )

        # Convert to tensor
        grating_tensor = torch.tensor(grating, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Get response
        with torch.no_grad():
            response = neuron(grating_tensor)
            # Use mean absolute response as magnitude
            responses.append(response.abs().mean().item())

    return np.array(responses)


def main():
    """Run Gabor orientation tuning demo."""
    # Create Gabor neurons with different preferred orientations
    preferred_orientations = [0, 45, 90, 135]
    neurons = []

    for ori in preferred_orientations:
        neuron = create_gabor_unit(
            orientation=ori,
            wavelength=8,
            sigma=3.0
        )
        neurons.append(neuron)

    # Test orientations (0-180 degrees in 5-degree steps)
    test_angles = np.linspace(0, 180, 37)

    # Get tuning curves
    tuning_curves = []
    for neuron in neurons:
        responses = test_orientation_tuning(neuron, test_angles)
        tuning_curves.append(responses)

    tuning_curves = np.array(tuning_curves)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Plot receptive fields
    for i, (neuron, ori) in enumerate(zip(neurons[:3], preferred_orientations[:3])):
        with torch.no_grad():
            E = neuron.weight_E[0, 0].numpy()
            I = neuron.weight_I[0, 0].numpy()
            rf = E - I

        im = axes[0, i].imshow(rf, cmap='RdBu_r', vmin=-rf.max(), vmax=rf.max())
        axes[0, i].set_title(f'{ori}° Gabor RF')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

    # Plot tuning curves
    ax = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(preferred_orientations)))

    for i, (ori, curve, color) in enumerate(zip(preferred_orientations, tuning_curves, colors)):
        ax.plot(test_angles, curve, 'o-', color=color,
                label=f'{ori}° preferred', linewidth=2, markersize=6)

        # Mark preferred orientation
        max_idx = np.argmax(curve)
        ax.axvline(test_angles[max_idx], color=color, linestyle='--', alpha=0.5)

    ax.set_xlabel('Stimulus Orientation (degrees)', fontsize=12)
    ax.set_ylabel('Response Magnitude', fontsize=12)
    ax.set_title('Orientation Tuning Curves', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 185)

    # Polar plot
    ax_polar = plt.subplot(2, 3, 5, projection='polar')
    theta = np.deg2rad(np.concatenate([test_angles, [180]]))

    for i, (ori, curve, color) in enumerate(zip(preferred_orientations, tuning_curves, colors)):
        # Normalize and complete circle
        curve_norm = curve / curve.max()
        curve_complete = np.concatenate([curve_norm, [curve_norm[0]]])

        ax_polar.plot(theta, curve_complete, 'o-', color=color,
                     label=f'{ori}°', linewidth=2, markersize=4)
        ax_polar.fill(theta, curve_complete, color=color, alpha=0.2)

    ax_polar.set_title('Polar Tuning Plot', fontsize=14, pad=20)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(-1)

    # Calculate and display OSI
    ax_text = axes[1, 2]
    ax_text.axis('off')

    osi_text = "Orientation Selectivity Index (OSI):\n\n"
    for i, (ori, curve) in enumerate(zip(preferred_orientations, tuning_curves)):
        osi = orientation_selectivity_index(curve, test_angles)
        osi_text += f"Neuron {ori}°: OSI = {osi:.3f}\n"

    ax_text.text(0.1, 0.5, osi_text, fontsize=12, verticalalignment='center',
                 family='monospace')

    plt.suptitle('Gabor Orientation Selectivity via E/I Patterns', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Show phase sensitivity (simple cell property)
    print("\nTesting phase sensitivity...")
    phases = np.linspace(0, 2*np.pi, 9)
    phase_responses = []

    test_neuron = neurons[0]  # 0° neuron
    for phase in phases:
        grating = create_static_grating(
            size=64,
            orientation=0,
            spatial_frequency=8,
            phase=phase,
            contrast=1.0
        )

        grating_tensor = torch.tensor(grating, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            response = test_neuron(grating_tensor)
            # For simple cells, use mean (not absolute) to show phase sensitivity
            phase_responses.append(response.mean().item())

    # Plot phase sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(phases * 180/np.pi, phase_responses, 'ko-', linewidth=2, markersize=8)
    ax1.set_xlabel('Grating Phase (degrees)')
    ax1.set_ylabel('Response')
    ax1.set_title('Phase Sensitivity (Simple Cell Property)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 370)

    # Show F1/F0 ratio
    f1_f0 = np.abs(np.max(phase_responses) - np.min(phase_responses)) / np.mean(np.abs(phase_responses))
    ax2.text(0.1, 0.5, f'F1/F0 ratio: {f1_f0:.2f}\n\n(>1 indicates simple cell)',
             fontsize=14, transform=ax2.transAxes)
    ax2.axis('off')

    plt.suptitle('Simple Cell Phase Sensitivity', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
