import unittest
import torch
import numpy as np
import os
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.synthetic_datasets import MultiXORDataset
from dendritic_modeling.visualization import (
    plot_branch_activations,
    plot_network_weights,
    plot_activation_distributions,
    plot_training_history,
    visualize_network_architecture
)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Create a small model for testing
        self.model = ExcitationInhibitionNetwork(
            input_dim=4, 
            hidden_layer_sizes=[8, 8], 
            branch_factors=[2, 2],
            output_dim=2,
            return_branch_outputs=True
        )
        
        # Create a small dataset for testing
        self.dataset = MultiXORDataset(n_samples=20, n_bits=4)
        
        # Sample input for testing
        self.input_data = torch.randint(0, 2, (10, 4), dtype=torch.float32)
        
        # Forward pass to get activations
        with torch.no_grad():
            _, self.branch_outputs = self.model(self.input_data)
        
        # Create a temporary directory for saving outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_plot_branch_activations(self):
        """Test plotting branch activations."""
        # Generate a plot
        save_path = os.path.join(self.temp_dir, 'branch_activations.png')
        fig = plot_branch_activations(self.branch_outputs, save_path=save_path)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(save_path))
        
        # Basic check that the file is a valid image
        self.assertGreater(os.path.getsize(save_path), 0)
    
    def test_plot_network_weights(self):
        """Test plotting network weights."""
        # Generate a plot
        save_path = os.path.join(self.temp_dir, 'network_weights.png')
        fig = plot_network_weights(self.model, save_path=save_path)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(save_path))
        
        # Basic check that the file is a valid image
        self.assertGreater(os.path.getsize(save_path), 0)
    
    def test_plot_activation_distributions(self):
        """Test plotting activation distributions."""
        # Generate a plot
        save_path = os.path.join(self.temp_dir, 'activation_distributions.png')
        fig = plot_activation_distributions(self.branch_outputs, save_path=save_path)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(save_path))
        
        # Basic check that the file is a valid image
        self.assertGreater(os.path.getsize(save_path), 0)
    
    def test_plot_training_history(self):
        """Test plotting training history."""
        # Create mock training history
        history = {
            'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
            'valid_loss': [0.9, 0.7, 0.5, 0.4, 0.3],
            'valid_acc': [0.5, 0.6, 0.7, 0.8, 0.85]
        }
        
        # Generate a plot
        save_path = os.path.join(self.temp_dir, 'training_history.png')
        fig = plot_training_history(history, save_path=save_path)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(save_path))
        
        # Basic check that the file is a valid image
        self.assertGreater(os.path.getsize(save_path), 0)
    
    def test_visualize_network_architecture(self):
        """Test network architecture visualization."""
        # Generate a plot
        save_path = os.path.join(self.temp_dir, 'network_architecture.png')
        fig = visualize_network_architecture(self.model, save_path=save_path)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        
        # Check that the file was saved
        self.assertTrue(os.path.exists(save_path))
        
        # Basic check that the file is a valid image
        self.assertGreater(os.path.getsize(save_path), 0)
    
    def test_visualization_without_saving(self):
        """Test visualization functions without saving."""
        # Plot without saving
        fig1 = plot_branch_activations(self.branch_outputs)
        fig2 = plot_network_weights(self.model)
        fig3 = plot_activation_distributions(self.branch_outputs)
        
        # Check that figures were created
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(fig3)
        
        # Close figures to free memory
        import matplotlib.pyplot as plt
        plt.close('all')

if __name__ == '__main__':
    unittest.main() 