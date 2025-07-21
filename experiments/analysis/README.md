# Analysis Tools

Comprehensive analysis tools for dendritic modeling experiments and results.

## Scripts Overview

### `analyze_pixel_information.py` (33KB, 841 lines)
**Purpose**: Comprehensive pixel-level information analysis using mutual information and partial information decomposition (PID).

**Key Features**:
- Single pixel mutual information analysis
- Pairwise pixel PID analysis (synergy, redundancy, unique information)
- Support for MNIST and CIFAR-10 datasets
- Parallel processing for large-scale analysis
- Clustering analysis of redundancy patterns
- Both sequential and parallel execution modes

**Usage**:
```bash
# Basic pixel analysis for MNIST
python analyze_pixel_information.py --dataset mnist --config configs/analysis.yaml

# Advanced PID analysis with parallel processing
python analyze_pixel_information.py --dataset cifar10 --pid-mode all_pairs --workers 8

# Random sampling of pixel pairs
python analyze_pixel_information.py --dataset mnist --pid-mode random_sample --num-pairs 1000

# Custom output directory and discretization
python analyze_pixel_information.py --dataset mnist --output-dir custom_output --num-bins 5
```

### `run_pixel_analysis.sh` (4.3KB, 113 lines)
**Purpose**: Shell script wrapper for running pixel analysis with different configurations.

**Key Features**:
- Automated dataset processing
- Resource management for cluster environments
- Parallel execution control
- Log management and error handling

**Usage**:
```bash
# Basic analysis for MNIST
bash run_pixel_analysis.sh mnist

# Parallel analysis for CIFAR-10
bash run_pixel_analysis.sh cifar10 --parallel

# Custom parameters
bash run_pixel_analysis.sh mnist --workers 4 --bins 10
```

## Dependencies

### Required Packages
```bash
# Information theory packages
pip install dit
pip install broja

# Scientific computing
pip install numpy scipy pandas

# Visualization
pip install matplotlib seaborn

# Parallel processing
pip install tqdm multiprocessing

# Machine learning
pip install scikit-learn torch torchvision
```

### Optional Dependencies
```bash
# Advanced clustering
pip install scikit-learn[extra]

# High-performance computing
pip install numba
```

## Analysis Types

### 1. Single Pixel Mutual Information

Computes `I(Pixel Value; Class Label)` for each pixel position:

```python
# Example output: Heatmap showing information content per pixel
MI_map[row, col] = mutual_info_score(labels, discretized_pixel_values)
```

**Output**: Heatmap visualization saved as `{dataset}_single_pixel_mi_bins{num_bins}.png`

### 2. Pairwise Pixel PID Analysis

Computes Partial Information Decomposition for pixel pairs:

- **Synergy (SI)**: Information unique to the pair `I(P1,P2; Y) - I(P1; Y) - I(P2; Y)`
- **Redundancy (CI)**: Shared information `I(P1; Y) + I(P2; Y) - I(P1,P2; Y)`
- **Unique Information**: Information unique to each pixel

**Modes**:
- `all_pairs`: Analyze all possible pixel pairs (computationally intensive)
- `random_sample`: Random sampling of pixel pairs (recommended for large images)

### 3. Redundancy Clustering

Groups pixels based on redundancy patterns:

```bash
python analyze_pixel_information.py --dataset mnist --cluster-redundancy --num-clusters 5
```

## Usage Examples

### Complete Information Analysis Pipeline

```bash
# 1. Single pixel analysis
python analyze_pixel_information.py \
    --dataset mnist \
    --config configs/mnist_analysis.yaml \
    --output-dir results/mnist_info

# 2. Pairwise analysis with sampling
python analyze_pixel_information.py \
    --dataset mnist \
    --pid-mode random_sample \
    --num-pairs 5000 \
    --workers 4 \
    --output-dir results/mnist_info

# 3. Clustering analysis
python analyze_pixel_information.py \
    --dataset mnist \
    --cluster-redundancy \
    --num-clusters 8 \
    --output-dir results/mnist_info
```

### Large-Scale Analysis (CIFAR-10)

```bash
# Use shell script for resource management
bash run_pixel_analysis.sh cifar10 \
    --workers 16 \
    --memory 64GB \
    --time 12:00:00 \
    --partition gpu
```

### Custom Configuration File

Create `configs/analysis.yaml`:

```yaml
analysis:
  num_bins_pixel: 5
  num_pairs_to_analyze: 10000
  pid_analysis_mode: "random_sample"
  num_workers: 8
  num_redundancy_clusters: 6

datasets:
  mnist:
    flatten: false
    normalize: false
  cifar10:
    flatten: false
    normalize: false
    
output:
  save_matrices: true
  save_plots: true
  plot_format: "png"
  dpi: 300
```

## Output Structure

```
output/
└── pixel_analysis/
    ├── mnist/
    │   ├── mnist_single_pixel_mi_bins5.png
    │   ├── mnist_pairwise_pid_bins5_pairs5000.txt
    │   ├── pid_matrices/
    │   │   ├── mnist_pid_matrix_synergy_bins5.npy
    │   │   ├── mnist_pid_matrix_redundancy_bins5.npy
    │   │   └── mnist_pid_matrix_unique_info_*.npy
    │   ├── clustering/
    │   │   ├── redundancy_clusters_8.png
    │   │   └── cluster_analysis.json
    │   └── heatmaps/
    │       ├── synergy_heatmap.png
    │       ├── redundancy_heatmap.png
    │       └── unique_info_heatmaps.png
    └── cifar10/
        └── ... (similar structure)
```

## Advanced Features

### Parallel Processing Configuration

```python
# Configure multiprocessing
import multiprocessing as mp

# Set number of workers based on system
num_workers = min(mp.cpu_count(), 16)

# Memory-efficient processing for large datasets
chunk_size = 1000  # Process 1000 pairs at a time
```

### Custom Discretization

```python
# Adaptive binning based on data distribution
def adaptive_discretize(pixel_data, target_bins=5):
    # Use quantile-based binning for improved distribution
    quantiles = np.linspace(0, 1, target_bins + 1)
    bin_edges = np.quantile(pixel_data, quantiles)
    return np.digitize(pixel_data, bin_edges[:-1])
```

### Information Theory Metrics

Available metrics computed by the scripts:

1. **Mutual Information**: `I(X; Y)`
2. **Conditional Mutual Information**: `I(X; Y | Z)`
3. **Synergy**: `I(X,Y; Z) - I(X; Z) - I(Y; Z)`
4. **Redundancy**: `I(X; Z) + I(Y; Z) - I(X,Y; Z)`
5. **Unique Information**: `I(X; Z | Y)` and `I(Y; Z | X)`

## Troubleshooting

### Common Issues

1. **Memory Issues**:
   ```bash
   # Reduce number of pairs or workers
   python analyze_pixel_information.py --num-pairs 1000 --workers 2
   ```

2. **BROJA Import Errors**:
   ```bash
   # Install specific version
   pip install git+https://github.com/Abzinger/BROJA-2PID.git
   ```

3. **Long Computation Times**:
   ```bash
   # Use random sampling instead of all pairs
   python analyze_pixel_information.py --pid-mode random_sample --num-pairs 5000
   ```

4. **Disk Space Issues**:
   ```bash
   # Disable matrix saving for large analyses
   python analyze_pixel_information.py --no-save-matrices
   ```

### Performance Optimization

- **Use appropriate number of workers**: `min(cpu_count(), 16)`
- **Sample pixel pairs for large images**: Use `random_sample` mode
- **Adjust discretization bins**: Lower bins = faster computation
- **Monitor memory usage**: Use `htop` or `nvidia-smi` during execution

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add progress tracking
from tqdm import tqdm
```

## Related Theory

### Partial Information Decomposition (PID)

The Williams-Beer PID framework decomposes the information that two sources provide about a target into four components:

- **Redundancy**: Information provided by both sources
- **Unique Information**: Information provided by only one source
- **Synergy**: Information provided only by both sources together
- **Misinformation**: Negative interaction (rare in practice)

### Applications in Dendritic Modeling

1. **Pixel Correlation Analysis**: Understanding spatial correlations in input data
2. **Information Routing**: Analyzing how information flows through dendritic branches
3. **Feature Decomposition**: Identifying redundant vs. synergistic features
4. **Network Architecture**: Informing dendritic branch organization

---

*Compatible with dendritic-modeling framework v2.0+* 