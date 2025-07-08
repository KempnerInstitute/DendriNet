"""Demo: Dendritic neuron as Sobel edge detector.

Shows that carefully designed E/I weights can exactly reproduce
standard CNN kernels like Sobel filters.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d


def init_push_pull_edge(layer: DendriteConv2d, edge_type: str = "sobel_x"):
    """Initialize dendritic layer as edge detector.
    
    Parameters
    ----------
    layer : DendriteConv2d
        Layer to initialize
    edge_type : str
        'sobel_x', 'sobel_y', or 'laplacian'
    """
    assert layer.kernel_size == (3, 3), "This demo requires 3x3 kernels"
    
    if edge_type == "sobel_x":
        # Horizontal edge detector (Sobel X)
        kernel = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=np.float32)
    elif edge_type == "sobel_y":
        # Vertical edge detector (Sobel Y)
        kernel = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=np.float32)
    elif edge_type == "laplacian":
        # Laplacian edge detector
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")
    
    # Push-pull decomposition
    W_E = np.maximum(kernel, 0) + 0.01  # Excitatory parts
    W_I = np.maximum(-kernel, 0) + 0.01  # Inhibitory parts
    
    # Set weights
    with torch.no_grad():
        layer.weight_E_raw.data[0, 0] = torch.log(torch.tensor(W_E))
        layer.weight_I_raw.data[0, 0] = torch.log(torch.tensor(W_I))


def create_test_image(size: int = 64) -> np.ndarray:
    """Create test image with edges."""
    img = np.zeros((size, size))
    
    # Diagonal edge
    for i in range(size//4, 3*size//4):
        img[i, i] = 1.0
        img[i, i+1] = 1.0
        
    # Vertical edge
    img[:, size//2:size//2+2] = 0.8
    
    # Horizontal edge
    img[3*size//4:3*size//4+2, :] = 0.6
    
    # Add some noise
    img += np.random.randn(size, size) * 0.1
    
    return np.clip(img, 0, 1)


def main():
    """Run edge detection demo."""
    # Create dendritic edge detector
    layer = DendriteConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding=1,
        bias=False
    )
    
    # Initialize as Sobel X
    init_push_pull_edge(layer, "sobel_x")
    
    # Create test image
    test_img = create_test_image()
    
    # Process
    img_tensor = torch.tensor(test_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = layer(img_tensor)
        
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # E and I weights
    with torch.no_grad():
        E = layer.weight_E[0, 0].numpy()
        I = layer.weight_I[0, 0].numpy()
        
    im = axes[0, 1].imshow(E, cmap='hot')
    axes[0, 1].set_title('E weights')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    im = axes[0, 2].imshow(I, cmap='hot')
    axes[0, 2].set_title('I weights')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Effective kernel (E - I)
    effective = E - I
    im = axes[1, 0].imshow(effective, cmap='RdBu_r', vmin=-3, vmax=3)
    axes[1, 0].set_title('Effective kernel (E - I)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Output
    im = axes[1, 1].imshow(output[0, 0].numpy(), cmap='gray')
    axes[1, 1].set_title('Dendritic Output')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Compare with standard Sobel
    from scipy import ndimage
    sobel_output = ndimage.sobel(test_img, axis=1)
    
    im = axes[1, 2].imshow(sobel_output, cmap='gray')
    axes[1, 2].set_title('Standard Sobel')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.suptitle('Dendritic Neuron as Sobel Edge Detector', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("\nComparison with standard Sobel:")
    print(f"Correlation: {np.corrcoef(output[0, 0].numpy().ravel(), sobel_output.ravel())[0, 1]:.4f}")
    print(f"RMSE: {np.sqrt(np.mean((output[0, 0].numpy() - sobel_output)**2)):.4f}")
    
    # Show other edge types
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, edge_type in enumerate(['sobel_x', 'sobel_y', 'laplacian']):
        layer_type = DendriteConv2d(1, 1, 3, padding=1, bias=False)
        init_push_pull_edge(layer_type, edge_type)
        
        with torch.no_grad():
            E = layer_type.weight_E[0, 0].numpy()
            I = layer_type.weight_I[0, 0].numpy()
            kernel = E - I
            
        im = axes[i].imshow(kernel, cmap='RdBu_r', vmin=-4, vmax=4)
        axes[i].set_title(f'{edge_type.title()} kernel')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
        
    plt.suptitle('Different Edge Detectors via E/I Patterns', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 