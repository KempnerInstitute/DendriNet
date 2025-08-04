# Documentation Figures

This directory contains network schematics and architectural diagrams for the Dendritic Modeling Framework documentation.

## Network Schematics (in `architecture/`)

### 1. **dendritic_neuron_schematic.png**
Clean schematic of a dendritic neuron showing:
- Anatomically-inspired neuron morphology
- Excitatory synapses (red triangles)
- Inhibitory synapses (blue circles)
- Apical and basal dendrites
- Soma and axon
- Clear, professional visualization

### 2. **shunting_inhibition_mechanism.png**
Shunting inhibition principle:
- Circuit diagram showing E/(E+I+ε) computation
- Clean mathematical representation
- Key properties: divisive normalization, gain control

### 3. **network_architecture.png**
Multi-layer E/I network architecture:
- Three-layer organization
- Excitatory (E) and inhibitory (I) populations
- Connection patterns:
  - E→E between layers
  - E→I between layers
  - I→E within layers
  - I→I between layers

### 4. **dendritic_computation_flow.png**
Sequential dendritic processing:
- Three dendrites (distal to proximal)
- Mathematical formulas for each stage
- Integration at soma
- Clear information flow

### 5. **sparse_connectivity.png**
Connectivity strategies comparison:
- Dense connectivity (fully connected)
- TopK sparse (K connections per neuron)
- DEEPST dynamic (adaptive rewiring)

## Figure Generation

All figures are generated using the `generate_figures.py` script:

```bash
cd docs
python generate_figures.py
```

This script creates:
- High-quality network schematics at 300 DPI
- Clean, professional visualizations
- Consistent styling and color scheme
- No simulated data or results

## Figure Locations

- **Main figures**: `docs/figures/architecture/`
- **Static copies**: `docs/source/_static/` (for Sphinx)
- **README figure**: Repository root (`dendritic_neuron.png`)

## Design Principles

- **Clarity**: Clean, uncluttered designs
- **Accuracy**: Scientifically correct representations
- **Consistency**: Uniform style and color scheme
- **Professional**: Publication-quality graphics

## Color Scheme

- **Excitatory**: Orange (#D55E00)
- **Inhibitory**: Royal blue (#4169E1)
- **Dendrites**: Dim gray (#696969)
- **Soma**: Dark slate gray (#2F4F4F)
- **Axon**: Slate gray (#708090)
- **Background**: White (#FFFFFF)
- **Text**: Black (#000000) 