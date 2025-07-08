"""Test that hand-crafted dendritic neurons match expected visual properties."""

import numpy as np
import torch
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.vision_factories.neuron_factory import (
    build_neuron_from_yaml,
    get_stimulus_params,
    get_criterion,
    load_properties_config,
)
from dendritic_modeling.datasets.stimuli.gratings import create_static_grating
from dendritic_modeling.datasets.stimuli.noise import create_white_noise
from dendritic_modeling.datasets.stimuli.bars import create_moving_bar, create_length_sweep_bars
from dendritic_modeling.utils.visual_metrics import (
    orientation_selectivity_index,
    direction_selectivity_index,
    size_tuning_index,
    length_selectivity_index,
)


def compute_dog_fit(receptive_field: np.ndarray) -> float:
    """Compute R² for DoG fit."""
    from scipy.optimize import curve_fit
    
    H, W = receptive_field.shape
    y = np.arange(H) - H/2
    x = np.arange(W) - W/2
    Y, X = np.meshgrid(y, x)
    R = np.sqrt(X**2 + Y**2)
    
    # Flatten for fitting
    r_flat = R.flatten()
    rf_flat = receptive_field.flatten()
    
    # DoG function
    def dog(r, A_c, sigma_c, A_s, sigma_s):
        center = A_c * np.exp(-r**2 / (2 * sigma_c**2))
        surround = A_s * np.exp(-r**2 / (2 * sigma_s**2))
        return center - surround
    
    try:
        # Initial guess
        p0 = [rf_flat.max(), 3.0, rf_flat.max()*0.5, 6.0]
        popt, _ = curve_fit(dog, r_flat, rf_flat, p0=p0, maxfev=5000)
        
        # Calculate R²
        fitted = dog(r_flat, *popt)
        ss_res = np.sum((rf_flat - fitted)**2)
        ss_tot = np.sum((rf_flat - rf_flat.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        
        return max(0, r2)  # Ensure non-negative
    except:
        return 0.0


def get_config():
    """Load properties configuration."""
    return load_properties_config()


class TestVisualProperties:
    """Test each visual property template."""
    
    def test_retina_dog(self, config=None):
        """Test center-surround DoG property."""
        if config is None:
            config = get_config()
        # Build neuron
        neuron = build_neuron_from_yaml("retina_dog", config)
        
        # Get stimulus params
        stim_params = get_stimulus_params("retina_dog", config)
        
        # Generate white noise stimulus
        noise = create_white_noise(
            size=stim_params["params"]["size"],
            n_frames=100,  # Reduced for testing
            std=stim_params["params"]["std"]
        )
        
        # Compute spike-triggered average (simplified)
        sta = np.zeros((neuron.kernel_size[0], neuron.kernel_size[1]))
        
        # Get receptive field directly (E - I)
        with torch.no_grad():
            E = neuron.weight_E[0, 0].numpy()
            I = neuron.weight_I[0, 0].numpy()
            sta = E - I
            
        # Compute DoG fit
        r2 = compute_dog_fit(sta)
        
        # Check criterion
        criterion = get_criterion("retina_dog", config)
        assert r2 >= criterion["threshold"], f"DoG R² = {r2:.3f} < {criterion['threshold']}"
        
    def test_v1_osi(self, config=None):
        """Test orientation selectivity."""
        if config is None:
            config = get_config()
        # Build neuron
        neuron = build_neuron_from_yaml("V1_OSI", config)
        
        # Test orientations
        orientations = np.linspace(0, 180, 37)
        responses = []
        
        for ori in orientations:
            grating = create_static_grating(64, ori, 8)
            grating_tensor = torch.tensor(grating).float().unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                resp = neuron(grating_tensor)
                responses.append(resp.abs().mean().item())
                
        # Calculate OSI
        osi = orientation_selectivity_index(responses, orientations)
        
        # Check criterion
        criterion = get_criterion("V1_OSI", config)
        assert osi >= criterion["threshold"], f"OSI = {osi:.3f} < {criterion['threshold']}"
        
    def test_v1_dsi(self, config=None):
        """Test direction selectivity."""
        if config is None:
            config = get_config()
        # Build neuron
        neuron = build_neuron_from_yaml("V1_DSI", config)
        
        # Test directions
        directions = np.linspace(0, 360, 9)[:-1]
        responses = []
        
        for direction in directions:
            # Create moving bar
            movie = create_moving_bar(64, 4, direction=direction, n_frames=10)
            
            # Get max response across frames
            frame_responses = []
            for frame in movie:
                frame_tensor = torch.tensor(frame).float().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    resp = neuron(frame_tensor)
                    frame_responses.append(resp.max().item())
                    
            responses.append(max(frame_responses))
            
        # Calculate DSI
        dsi = direction_selectivity_index(responses, directions)
        
        # Check criterion
        criterion = get_criterion("V1_DSI", config)
        assert dsi >= criterion["threshold"], f"DSI = {dsi:.3f} < {criterion['threshold']}"
        
    def test_v1_endstop(self, config=None):
        """Test end-stopping (length tuning)."""
        if config is None:
            config = get_config()
        # Build neuron
        neuron = build_neuron_from_yaml("V1_endstop", config)
        
        # Get stimulus params
        stim_params = get_stimulus_params("V1_endstop", config)
        lengths = stim_params["params"]["lengths"]
        
        # Create bars of different lengths
        bars = create_length_sweep_bars(
            64, 
            stim_params["params"]["bar_width"],
            lengths=lengths,
            orientation=stim_params["params"]["orientation"]
        )
        
        responses = []
        for bar in bars:
            bar_tensor = torch.tensor(bar).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                resp = neuron(bar_tensor)
                responses.append(resp.max().item())
                
        responses = np.array(responses)
        
        # Check for peak-then-fall pattern
        peak_idx = np.argmax(responses)
        
        # Should peak in middle lengths
        assert 0 < peak_idx < len(responses) - 1, "No clear peak in length tuning"
        
        # Response should decrease for longest bars
        assert responses[-1] < responses[peak_idx] * 0.8, "No end-stopping observed"
        
        # Calculate length selectivity index
        lsi = length_selectivity_index(responses)
        assert lsi > 0.2, f"Weak end-stopping: LSI = {lsi:.3f}"


def test_all_properties():
    """Run all property tests."""
    config = load_properties_config()
    tester = TestVisualProperties()
    
    print("\nTesting visual property templates:")
    print("-" * 50)
    
    # Test each property
    for prop_name in ["retina_dog", "V1_OSI", "V1_DSI", "V1_endstop"]:
        if prop_name not in config:
            continue
            
        try:
            if prop_name == "retina_dog":
                tester.test_retina_dog(config)
            elif prop_name == "V1_OSI":
                tester.test_v1_osi(config)
            elif prop_name == "V1_DSI":
                tester.test_v1_dsi(config)
            elif prop_name == "V1_endstop":
                tester.test_v1_endstop(config)
                
            print(f"{prop_name}: PASSED")
            
        except AssertionError as e:
            print(f"✗ {prop_name}: FAILED - {str(e)}")
        except Exception as e:
            print(f"✗ {prop_name}: ERROR - {str(e)}")
            
    print("-" * 50)


if __name__ == "__main__":
    test_all_properties() 