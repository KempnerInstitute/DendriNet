"""
Test suite for Dendritic Convolution Layer.

This implements the test plan stages for validating dendritic convolution
as a drop-in replacement for standard convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytest
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d, DendriteAttention


class TestStage1ReceptiveField:
    """Stage 1: Unit-test single-patch receptive field check."""
    
    def test_oriented_edge_detector(self):
        """Test that a single dendritic neuron behaves like a simple cell."""
        # Create layer with single output channel
        layer = DendriteConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            epsilon=1e-4
        )
        
        # Freeze weights to form oriented edge (Sobel-like)
        edge = torch.tensor([[1, 1, 1], 
                           [-1, -1, -1], 
                           [0, 0, 0]], dtype=torch.float32)
        
        # Set excitatory weights
        with torch.no_grad():
            layer.weight_E_raw.data.copy_(
                torch.log(torch.exp(edge[None, None, :, :]) - 1 + 1e-6)
            )
            # Set inhibitory weights (anti-phase)
            layer.weight_I_raw.data.copy_(
                torch.log(torch.exp(-edge[None, None, :, :] + 2) - 1 + 1e-6)
            )
        
        # Create test images with moving bar
        batch_size = 8
        angles = torch.linspace(0, np.pi, batch_size)
        test_images = []
        
        for angle in angles:
            # Create image with oriented bar
            x = torch.linspace(-1, 1, 28)
            y = torch.linspace(-1, 1, 28)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Oriented bar
            bar = torch.cos(angle) * X + torch.sin(angle) * Y
            img = (bar > -0.1) & (bar < 0.1)
            test_images.append(img.float())
        
        test_batch = torch.stack(test_images).unsqueeze(1)  # (batch, 1, 28, 28)
        
        # Forward pass
        with torch.no_grad():
            output = layer(test_batch)
        
        # Verify response characteristics
        max_response = output.max(dim=(2, 3), keepdim=False)[0].squeeze()
        min_response = output.min(dim=(2, 3), keepdim=False)[0].squeeze()
        
        # Pass criterion: max - min >= 0.7
        response_range = (max_response - min_response).mean().item()
        assert response_range >= 0.7, f"Response range {response_range} < 0.7"
        
        # Peak should occur when bar aligns with excitatory field
        peak_idx = torch.argmax(max_response)
        expected_angle = 0  # Horizontal edge detector
        assert abs(angles[peak_idx] - expected_angle) < 0.5, "Peak response at wrong orientation"
        
        print(f"✓ Stage 1 passed: Response range = {response_range:.3f}")
        
        # Visualize if requested
        self._visualize_receptive_field(layer, test_batch, output)
    
    def _visualize_receptive_field(self, layer, inputs, outputs, save_path="results/stage1_rf.png"):
        """Visualize receptive field test results."""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        # Show first 4 test images
        for i in range(4):
            axes[0, i].imshow(inputs[i, 0].numpy(), cmap='gray')
            axes[0, i].set_title(f'Input {i}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(outputs[i, 0].detach().numpy(), cmap='RdBu_r')
            axes[1, i].set_title(f'Response {i}')
            axes[1, i].axis('off')
        
        plt.suptitle('Stage 1: Receptive Field Test')
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()


class ToyNet(nn.Module):
    """Simple network for Stage 2 testing."""
    
    def __init__(self, dendrite=True, num_classes=10):
        super(ToyNet, self).__init__()
        Conv = DendriteConv2d if dendrite else nn.Conv2d
        
        # For standard Conv2d, we need to handle the signature difference
        if dendrite:
            self.conv1 = Conv(3, 32, kernel_size=5, stride=1, padding=2)
            self.conv2 = Conv(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = Conv(64, 128, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = Conv(3, 32, kernel_size=5, stride=1, padding=2)
            self.conv2 = Conv(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = Conv(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.dendrite = dendrite
        
        # Calculate feature size
        # CIFAR-10: 32x32 -> 32x32 -> 16x16 -> 8x8
        # MNIST: 28x28 -> 28x28 -> 14x14 -> 7x7
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 6 * 6, num_classes)
        
    def forward(self, x):
        # Conv1
        if self.dendrite:
            x = F.relu(self.conv1(x))
        else:
            x = F.relu(self.conv1(x))
        
        # Conv2
        if self.dendrite:
            x = F.relu(self.conv2(x))
        else:
            x = F.relu(self.conv2(x))
        
        # Conv3
        if self.dendrite:
            x = F.relu(self.conv3(x))
        else:
            x = F.relu(self.conv3(x))
        
        # Pooling and classification
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


class TestStage2FeatureExtraction:
    """Stage 2: Layer-level benchmark on image classification."""
    
    def test_mnist_classification(self):
        """Test on MNIST dataset."""
        from torchvision import datasets, transforms
        
        # Setup data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Create small subset for testing
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        # Use subset for quick testing
        train_subset = torch.utils.data.Subset(train_dataset, range(1000))
        test_subset = torch.utils.data.Subset(test_dataset, range(100))
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)
        
        # Train both models
        results = {}
        
        for use_dendrite in [True, False]:
            model_name = "Dendritic" if use_dendrite else "Standard"
            print(f"\nTraining {model_name} Conv model...")
            
            # Adjust input channels for MNIST (1 channel)
            model = self._create_mnist_model(use_dendrite)
            
            # Train
            accuracy = self._train_model(model, train_loader, test_loader, epochs=5)
            results[model_name] = accuracy
            
            print(f"{model_name} accuracy: {accuracy:.2f}%")
        
        # Check that dendritic performs comparably
        assert results["Dendritic"] > results["Standard"] - 5, \
            f"Dendritic ({results['Dendritic']:.1f}%) significantly worse than Standard ({results['Standard']:.1f}%)"
        
        print(f"\n✓ Stage 2 MNIST passed: Dendritic {results['Dendritic']:.1f}% vs Standard {results['Standard']:.1f}%")
    
    def _create_mnist_model(self, dendrite=True):
        """Create model for MNIST (1 input channel)."""
        class MNISTNet(nn.Module):
            def __init__(self, dendrite=True):
                super().__init__()
                Conv = DendriteConv2d if dendrite else nn.Conv2d
                
                self.conv1 = Conv(1, 32, kernel_size=5, stride=1, padding=2)
                self.conv2 = Conv(32, 64, kernel_size=3, stride=2, padding=1)
                self.conv3 = Conv(64, 128, kernel_size=3, stride=2, padding=1)
                
                self.dendrite = dendrite
                self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(128 * 6 * 6, 10)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = self.adaptive_pool(x)
                x = self.flatten(x)
                x = self.fc(x)
                return x
        
        return MNISTNet(dendrite)
    
    def _train_model(self, model, train_loader, test_loader, epochs=5):
        """Train model and return test accuracy."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy


class TestStage3RepresentationAnalysis:
    """Stage 3: Representation analysis using CKA."""
    
    def test_cka_similarity(self):
        """Test CKA similarity between dendritic and standard features."""
        # Create both models
        model_dendrite = ToyNet(dendrite=True)
        model_standard = ToyNet(dendrite=False)
        
        # Generate random data
        batch_size = 100
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Extract features from both models
        features_dendrite = self._extract_features(model_dendrite, x)
        features_standard = self._extract_features(model_standard, x)
        
        # Compute CKA for each layer pair
        cka_scores = []
        
        for i, (feat_d, feat_s) in enumerate(zip(features_dendrite, features_standard)):
            cka = self._compute_cka(feat_d, feat_s)
            cka_scores.append(cka)
            print(f"Layer {i+1} CKA: {cka:.3f}")
        
        # Pass criterion: mean CKA > 0.6
        mean_cka = np.mean(cka_scores)
        assert mean_cka > 0.6, f"Mean CKA {mean_cka:.3f} < 0.6"
        
        print(f"\n✓ Stage 3 passed: Mean CKA = {mean_cka:.3f}")
        
        # Visualize CKA matrix
        self._visualize_cka(cka_scores)
    
    def _extract_features(self, model, x):
        """Extract intermediate features from model."""
        features = []
        
        # Hook to capture features
        def hook(module, input, output):
            features.append(output.detach())
        
        # Register hooks
        handles = []
        handles.append(model.conv1.register_forward_hook(hook))
        handles.append(model.conv2.register_forward_hook(hook))
        handles.append(model.conv3.register_forward_hook(hook))
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Flatten features for CKA
        features_flat = []
        for feat in features:
            feat_flat = feat.reshape(feat.size(0), -1)
            features_flat.append(feat_flat)
        
        return features_flat
    
    def _compute_cka(self, X, Y):
        """Compute Centered Kernel Alignment (CKA) between feature matrices."""
        # Center the features
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        K_X = torch.mm(X, X.t())
        K_Y = torch.mm(Y, Y.t())
        
        # Center Gram matrices
        n = K_X.size(0)
        H = torch.eye(n) - torch.ones(n, n) / n
        K_X = torch.mm(torch.mm(H, K_X), H)
        K_Y = torch.mm(torch.mm(H, K_Y), H)
        
        # Compute CKA
        hsic_xy = torch.trace(torch.mm(K_X, K_Y))
        hsic_xx = torch.trace(torch.mm(K_X, K_X))
        hsic_yy = torch.trace(torch.mm(K_Y, K_Y))
        
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        
        return cka.item()
    
    def _visualize_cka(self, cka_scores, save_path="results/stage3_cka.png"):
        """Visualize CKA scores."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create matrix (for now just diagonal)
        n_layers = len(cka_scores)
        cka_matrix = np.eye(n_layers)
        for i in range(n_layers):
            cka_matrix[i, i] = cka_scores[i]
        
        im = ax.imshow(cka_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('Standard Conv Layers')
        ax.set_ylabel('Dendritic Conv Layers')
        ax.set_title('CKA Similarity Matrix')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(n_layers):
            ax.text(i, i, f'{cka_scores[i]:.2f}', ha='center', va='center', color='white')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()


class TestStage4Attention:
    """Stage 4: Attention-level benchmark."""
    
    def test_attention_sparsity(self):
        """Test that dendritic attention produces sparser weights."""
        embed_dim = 128
        num_heads = 4
        seq_len = 16
        batch_size = 8
        
        # Create both attention mechanisms
        dendrite_attn = DendriteAttention(embed_dim, num_heads)
        standard_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Generate random input
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Forward pass through dendritic attention
        output_d, attn_weights_d = dendrite_attn(x, x, x)
        
        # Forward pass through standard attention
        with torch.no_grad():
            output_s, attn_weights_s = standard_attn(x, x, x, need_weights=True)
        
        # Compute sparsity (fraction of weights > 0.1)
        sparsity_d = (attn_weights_d > 0.1).float().mean().item()
        sparsity_s = (attn_weights_s > 0.1).float().mean().item()
        
        print(f"Dendritic attention sparsity: {sparsity_d:.3f}")
        print(f"Standard attention sparsity: {sparsity_s:.3f}")
        
        # Dendritic should be sparser
        assert sparsity_d < sparsity_s, "Dendritic attention not sparser than standard"
        
        # Check energy conservation: sum of attention weights should be <= 1
        attn_sum = attn_weights_d.sum(dim=-1)
        assert (attn_sum <= 1.01).all(), "Attention weights sum > 1"
        
        print(f"\n✓ Stage 4 passed: Dendritic sparsity {sparsity_d:.3f} < Standard {sparsity_s:.3f}")
        
        # Visualize attention patterns
        self._visualize_attention(attn_weights_d, attn_weights_s)
    
    def _visualize_attention(self, attn_d, attn_s, save_path="results/stage4_attention.png"):
        """Visualize attention weight patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Show first head of first batch
        axes[0, 0].imshow(attn_d[0, 0].detach().numpy(), cmap='hot', aspect='auto')
        axes[0, 0].set_title('Dendritic Attention (Head 1)')
        axes[0, 0].set_xlabel('Keys')
        axes[0, 0].set_ylabel('Queries')
        
        axes[0, 1].imshow(attn_s[0, 0].detach().numpy(), cmap='hot', aspect='auto')
        axes[0, 1].set_title('Standard Attention (Head 1)')
        axes[0, 1].set_xlabel('Keys')
        axes[0, 1].set_ylabel('Queries')
        
        # Show attention sum histograms
        attn_sum_d = attn_d.sum(dim=-1).flatten().detach().numpy()
        attn_sum_s = attn_s.sum(dim=-1).flatten().detach().numpy()
        
        axes[1, 0].hist(attn_sum_d, bins=50, alpha=0.7, label='Dendritic')
        axes[1, 0].axvline(x=1.0, color='red', linestyle='--', label='Sum=1')
        axes[1, 0].set_xlabel('Attention Sum')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Dendritic Attention Sum Distribution')
        axes[1, 0].legend()
        
        axes[1, 1].hist(attn_sum_s, bins=50, alpha=0.7, label='Standard')
        axes[1, 1].axvline(x=1.0, color='red', linestyle='--', label='Sum=1')
        axes[1, 1].set_xlabel('Attention Sum')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Standard Attention Sum Distribution')
        axes[1, 1].legend()
        
        plt.suptitle('Stage 4: Attention Analysis')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()


class TestVisualizationDiagnostics:
    """Stage 5: Visual diagnostics."""
    
    def test_learned_kernels(self):
        """Visualize learned E and I kernels."""
        # Create and initialize model
        model = ToyNet(dendrite=True)
        
        # Extract first layer kernels
        conv1 = model.conv1
        E_kernels = conv1.weight_E.detach()[:8]  # First 8 filters
        I_kernels = conv1.weight_I.detach()[:8]
        
        # Visualize
        self._plot_kernels(E_kernels, I_kernels)
        
        print("✓ Stage 5: Kernel visualization completed")
    
    def test_contrast_response(self):
        """Test response curves vs contrast."""
        layer = DendriteConv2d(1, 1, kernel_size=3, padding=1)
        
        # Create edge detector weights
        with torch.no_grad():
            edge = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32)
            layer.weight_E_raw.data.copy_(
                torch.log(torch.exp(edge[None, None, :, :]) - 1 + 1e-6)
            )
            layer.weight_I_raw.data.copy_(
                torch.log(torch.exp(torch.ones_like(edge)[None, None, :, :] * 0.5) - 1 + 1e-6)
            )
        
        # Test different contrasts
        contrasts = torch.linspace(0, 2, 20)
        responses = []
        
        for contrast in contrasts:
            # Create test image with edge at given contrast
            img = torch.zeros(1, 1, 10, 10)
            img[:, :, :, :5] = contrast
            img[:, :, :, 5:] = -contrast
            
            with torch.no_grad():
                output = layer(img)
                response = output.abs().max().item()
                responses.append(response)
        
        # Plot response curve
        self._plot_contrast_response(contrasts.numpy(), np.array(responses))
        
        print("✓ Stage 5: Contrast response analysis completed")
    
    def _plot_kernels(self, E_kernels, I_kernels, save_path="results/stage5_kernels.png"):
        """Plot E and I kernel pairs."""
        n_filters = len(E_kernels)
        fig, axes = plt.subplots(2, n_filters, figsize=(n_filters * 2, 4))
        
        for i in range(n_filters):
            # Plot E kernel
            e_kernel = E_kernels[i, 0].numpy()  # First channel
            axes[0, i].imshow(e_kernel, cmap='RdBu_r', aspect='auto')
            axes[0, i].set_title(f'E{i}')
            axes[0, i].axis('off')
            
            # Plot I kernel
            i_kernel = I_kernels[i, 0].numpy()
            axes[1, i].imshow(i_kernel, cmap='RdBu_r', aspect='auto')
            axes[1, i].set_title(f'I{i}')
            axes[1, i].axis('off')
        
        plt.suptitle('Learned E/I Kernel Pairs (First Layer)')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    
    def _plot_contrast_response(self, contrasts, responses, save_path="results/stage5_contrast.png"):
        """Plot contrast response curve."""
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.plot(contrasts, responses, 'o-', label='Dendritic response')
        
        # Fit hyperbolic ratio function: R = c / (c + s)
        from scipy.optimize import curve_fit
        
        def hyperbolic(c, s):
            return c / (c + s + 0.1)
        
        try:
            popt, _ = curve_fit(hyperbolic, contrasts[5:], responses[5:])
            fitted = hyperbolic(contrasts, *popt)
            ax.plot(contrasts, fitted, '--', label=f'Hyperbolic fit (s={popt[0]:.2f})')
        except:
            pass
        
        ax.set_xlabel('Input Contrast')
        ax.set_ylabel('Response')
        ax.set_title('Contrast Response Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def run_all_tests():
    """Run all test stages in sequence."""
    print("="*60)
    print("DENDRITIC CONVOLUTION TEST SUITE")
    print("="*60)
    
    # Stage 1: Receptive field
    print("\n[Stage 1] Single-patch receptive field test...")
    test1 = TestStage1ReceptiveField()
    test1.test_oriented_edge_detector()
    
    # Stage 2: Feature extraction
    print("\n[Stage 2] Layer-level feature extraction...")
    test2 = TestStage2FeatureExtraction()
    test2.test_mnist_classification()
    
    # Stage 3: Representation analysis
    print("\n[Stage 3] Representation similarity analysis...")
    test3 = TestStage3RepresentationAnalysis()
    test3.test_cka_similarity()
    
    # Stage 4: Attention
    print("\n[Stage 4] Attention mechanism test...")
    test4 = TestStage4Attention()
    test4.test_attention_sparsity()
    
    # Stage 5: Visualizations
    print("\n[Stage 5] Visual diagnostics...")
    test5 = TestVisualizationDiagnostics()
    test5.test_learned_kernels()
    test5.test_contrast_response()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests() 