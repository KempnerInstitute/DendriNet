import unittest
import torch
import os
import tempfile
import re
from torch.utils.data import TensorDataset

from dendritic_modeling.utils import analyze_branch_ablation, analyze_layerwise_ablation, ablate_branch
from dendritic_modeling.networks import ExcitationInhibitionNetwork

class TestAblationFunctions(unittest.TestCase):
    def setUp(self):
        # Create a small test model and dataset
        self.model = ExcitationInhibitionNetwork(
            input_dim=10,
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[4],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            reactivate=True,
            enable_branch_outputs=True,
            output_dim=2
        )
        
        # Create a simple binary classification dataset
        inputs = torch.randn(20, 10)
        labels = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(inputs, labels)
    
    def test_ablate_branch(self):
        """Test the basic ablate_branch function."""
        # Make a copy of the model to test with
        model = self.model
        
        # Store original weights
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.detach().clone()
        
        # Test ablating a specific branch in layer 0
        ablate_branch(model, layer_idx=0, branch_ids=[0], mode='direct')
        
        # Verify that the branch's weights are zeroed out using a more flexible approach
        found_ablated_weights = False
        for name, param in model.named_parameters():
            # Use regex to extract layer and branch indices
            layer_match = re.search(r'branch_layers\.(\d+)', name)
            branch_match = re.search(r'branch_excitation.*?(\d+)', name)
            
            if (layer_match and branch_match and 'pre_w' in name):
                layer_idx = int(layer_match.group(1))
                branch_idx = int(branch_match.group(1))
                
                # Check if this is the branch we ablated (layer 0, branch 0)
                if layer_idx == 0 and branch_idx == 0:
                    # Check if the weights have been zeroed
                    if torch.all(param[0, :] == 0).item():
                        found_ablated_weights = True
                        break
        
        self.assertTrue(found_ablated_weights, "Branch weights were not properly ablated")
        
        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])
    
    def test_analyze_branch_ablation_basic(self):
        """Test that analyze_branch_ablation returns expected structure."""
        baseline_acc, results = analyze_branch_ablation(
            model=self.model,
            dataset=self.dataset,
            layer_idx=0,
            device='cpu'
        )
        
        # Check return values
        self.assertIsInstance(baseline_acc, float)
        self.assertTrue(0 <= baseline_acc <= 1)
        self.assertIsInstance(results, dict)
        self.assertIn("all_branches", results)
        
        # Make sure the results for all_branches is a tuple with (accuracy, delta)
        self.assertIsInstance(results["all_branches"], tuple)
        self.assertEqual(len(results["all_branches"]), 2)
        
        # Ablated accuracy should be a float between 0 and 1
        ablated_acc, delta = results["all_branches"]
        self.assertTrue(0 <= ablated_acc <= 1)
        
        # Delta should be the difference between baseline and ablated accuracy
        self.assertAlmostEqual(delta, baseline_acc - ablated_acc)
    
    def test_analyze_branch_ablation_specific_branches(self):
        """Test ablating specific branches."""
        baseline_acc, results = analyze_branch_ablation(
            model=self.model,
            dataset=self.dataset,
            layer_idx=0,
            branch_ids=[0, 1],
            device='cpu'
        )
        
        # Check that results include entries for each branch
        self.assertIn(0, results)
        self.assertIn(1, results)
        
        # Check result format for each branch
        for branch_id in [0, 1]:
            ablated_acc, delta = results[branch_id]
            self.assertIsInstance(ablated_acc, float)
            self.assertIsInstance(delta, float)
            
            # Accuracy should be in valid range
            self.assertTrue(0 <= ablated_acc <= 1)
            
            # Delta should match difference from baseline
            self.assertAlmostEqual(delta, baseline_acc - ablated_acc)
    
    def test_analyze_branch_ablation_dendritic_order(self):
        """Test ablation by dendritic order."""
        # Create a model with multiple dendritic orders (layers)
        model = ExcitationInhibitionNetwork(
            input_dim=10,
            excitatory_layer_sizes=[4, 4],  # Two layers for two orders
            inhibitory_layer_sizes=[4, 4],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            reactivate=True,
            enable_branch_outputs=True,
            output_dim=2
        )
        
        results = analyze_branch_ablation(
            model=model,
            dataset=self.dataset,
            layer_idx=None,
            branch_ids=None,
            device='cpu'
        )
        
        # Check that results contain the expected keys for order analysis
        self.assertIn("order", results)
        self.assertIn("exc_drop", results)
        self.assertIn("inh_drop", results)
        
        # Check lengths match
        self.assertEqual(len(results["order"]), len(results["exc_drop"]))
        self.assertEqual(len(results["order"]), len(results["inh_drop"]))
        
        # Order should be a list of consecutive integers starting from 1
        self.assertEqual(results["order"], list(range(1, len(results["order"]) + 1)))
        
        # Drops should be non-negative floats
        for drop in results["exc_drop"]:
            self.assertGreaterEqual(drop, 0)
            self.assertIsInstance(drop, float)
            
        for drop in results["inh_drop"]:
            self.assertGreaterEqual(drop, 0)
            self.assertIsInstance(drop, float)
    
    def test_analyze_layerwise_ablation(self):
        """Test layerwise ablation analysis."""
        # Create a model with multiple dendritic orders (layers)
        model = ExcitationInhibitionNetwork(
            input_dim=10,
            excitatory_layer_sizes=[4, 4],  # Two layers for two orders
            inhibitory_layer_sizes=[4, 4],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            reactivate=True,
            enable_branch_outputs=True,
            output_dim=2
        )
        
        # Create a temporary directory for saving results
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Test with the signature from combined_dendritic_modeling.py
            results = analyze_layerwise_ablation(
                model=model,
                dataset=self.dataset,
                train_data=self.dataset,
                valid_data=self.dataset,
                test_data=self.dataset,
                save_path=tmpdirname,
                device='cpu'
            )
            
            # Check that results contains expected keys
            self.assertIn('layer_idx', results)
            self.assertIn('baseline_acc', results)
            self.assertIn('inh_ablated_acc', results)
            self.assertIn('exc_ablated_acc', results)
            
            # Check result shapes
            self.assertEqual(len(results['layer_idx']), len(results['baseline_acc']))
            self.assertEqual(len(results['layer_idx']), len(results['inh_ablated_acc']))
            self.assertEqual(len(results['layer_idx']), len(results['exc_ablated_acc']))
            
            # Check that files were saved to the temporary directory
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "layerwise_ablation")))
            
            # Note: We're not checking plot files because the plot call might be skipped 
            # in a test environment without display

if __name__ == '__main__':
    unittest.main() 