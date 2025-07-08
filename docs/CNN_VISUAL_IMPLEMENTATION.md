# Dendritic CNN and Visual Neuron Implementation

This module implements dendritic neurons that can function as CNN filters and reproduce visual neuron properties through excitatory/inhibitory (E/I) synaptic patterns.

## Overview

The implementation demonstrates that dendritic neurons with shunting inhibition can:
1. **Replace CNN filters** - Any linear CNN filter can be decomposed into E/I patterns
2. **Reproduce visual neuron properties** - Orientation selectivity, motion detection, center-surround, end-stopping
3. **Learn via gradient descent** - Training with mixed task + biological losses

## Key Components

### 1. Stimulus Generators (`src/dendritic_modeling/datasets/stimuli/`)
- **gratings.py**: Drifting/static gratings for orientation tuning
- **noise.py**: White/colored noise for receptive field mapping  
- **bars.py**: Moving bars for motion/length tuning
- **textures.py**: Complex textures for higher-order properties

### 2. Visual Neuron Factories (`src/dendritic_modeling/networks/factories/vision_units.py`)
Pre-configured dendritic neurons that implement:
- Gabor filters (V1 simple cells)
- Center-surround (retinal/LGN cells)
- Motion detectors (direction selective cells)
- End-stopped cells (hypercomplex cells)

### 3. Demo Scripts (`scripts/`)
- **cnn_equiv_edge.py**: Shows Sobel edge detector via E/I patterns
- **gabor_orientation.py**: Orientation tuning curves and phase sensitivity
- **motion_DS.py**: Direction selectivity via spatiotemporal E/I

### 4. Biological Loss Functions (`src/dendritic_modeling/training/bio_losses.py`)
Differentiable losses for training:
- `loss_DoG`: Difference of Gaussians fit for center-surround
- `loss_OSI`: Orientation selectivity index
- `loss_DSI`: Direction selectivity index  
- `loss_phase_invariance`: F1/F0 ratio for complex cells
- `loss_endstop`: Length tuning curves
- `loss_sparsity`: Activity sparsity

### 5. Visual Metrics (`src/dendritic_modeling/metrics/visual.py`)
Standard neuroscience metrics:
- Orientation Selectivity Index (OSI)
- Direction Selectivity Index (DSI)
- Size/length tuning indices

### 6. Property Configuration (`configs/properties.yaml`)
Defines visual properties with:
- Weight initialization templates
- Stimulus parameters
- Success criteria/thresholds

## Usage Examples

### Creating Visual Neurons

```python
from dendritic_modeling.networks.factories.vision_units import (
    create_gabor_unit, create_center_surround_unit
)

# V1 simple cell (45° Gabor)
simple_cell = create_gabor_unit(orientation=45, wavelength=8, sigma=3)

# Retinal ON-center cell
on_center = create_center_surround_unit(kind="on", center_size=5, surround_size=11)
```

### Testing Visual Properties

```python
from dendritic_modeling.stimuli.gratings import create_static_grating
from dendritic_modeling.metrics.visual import orientation_selectivity_index

# Test orientation tuning
orientations = np.linspace(0, 180, 37)
responses = []

for ori in orientations:
    grating = create_static_grating(64, ori, spatial_frequency=8)
    grating_tensor = torch.tensor(grating).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        resp = simple_cell(grating_tensor)
        responses.append(resp.abs().mean().item())

# Calculate OSI
osi = orientation_selectivity_index(responses, orientations)
print(f"OSI = {osi:.3f}")  # Should be > 0.8 for good selectivity
```

### Converting CNN to Dendritic

```python
from dendritic_modeling.networks.convert.conv_to_dendrite import conv2d_to_dendrite

# Convert existing Conv2d layer
conv_layer = nn.Conv2d(3, 64, 7)
dendrite_layer = conv2d_to_dendrite(conv_layer)

# Now dendrite_layer has E/I weights implementing the same computation
```

### Training with Biological Constraints

```python
from dendritic_modeling.training.bio_losses import compute_bio_losses

# During training loop
model_responses = {
    'gratings': grating_responses,  # (batch, n_orientations)
    'motion': motion_responses,      # (batch, n_directions)
}

loss_weights = {
    'osi': 0.05,
    'dsi': 0.05,
    'sparse': 0.01
}

bio_loss, loss_dict = compute_bio_losses(model_responses, loss_weights, probe_metadata)
total_loss = task_loss + bio_loss
```

## Key Findings

1. **Push-pull E/I = Linear filters**: Any CNN kernel can be decomposed as E-I
2. **Shunting adds normalization**: Divisive form provides automatic gain control
3. **Biological constraints satisfied**: No negative weights needed
4. **Rich dynamics possible**: Motion, end-stopping emerge from E/I patterns

## Running Tests

```bash
# Test visual properties
python -m dendritic_modeling.tests.test_property_match

# Run demos
python src/dendritic_modeling/demos/gabor_orientation.py
python src/dendritic_modeling/demos/motion_DS.py
```

## References

The implementation is based on neuroscience literature showing:
- V1 simple cells have Gabor-like receptive fields
- Push-pull inhibition implements linear operations
- Shunting inhibition provides divisive normalization
- Complex cells pool simple cells with different phases 