import unittest
import torch
import os
import tempfile
import shutil
from pathlib import Path

from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.synthetic_datasets import MultiXORDataset
from dendritic_modeling.trainers import (
    ClassificationTrainer, 
    PruningTrainer,
    get_dataloaders
)

class TestTrainers(unittest.TestCase):
    def setUp(self):
        # Create a small dataset for testing
        self.dataset = MultiXORDataset(n_samples=100, n_bits=4)
        
        # Create a small model for testing
        self.model = ExcitationInhibitionNetwork(
            input_dim=4, 
            hidden_layer_sizes=[8, 8], 
            branch_factors=[2, 2],
            output_dim=2
        )
        
        # Create a temporary directory for saving outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_get_dataloaders(self):
        """Test the get_dataloaders function."""
        # Split the dataset
        train_loader, valid_loader = get_dataloaders(
            self.dataset, 
            batch_size=16, 
            train_split=0.8,
            shuffle=True
        )
        
        # Check that the loaders have the expected number of samples
        self.assertEqual(len(train_loader.dataset), 80)
        self.assertEqual(len(valid_loader.dataset), 20)
        
        # Check that the batches have the expected shapes
        x, y = next(iter(train_loader))
        self.assertEqual(x.shape, (16, 4))  # Batch size x input dim
        self.assertEqual(y.shape, (16,))    # Batch size
        
        # Test with no validation split
        train_loader_only = get_dataloaders(
            self.dataset, 
            batch_size=16, 
            train_split=1.0,
            shuffle=True
        )
        
        # Should only return a single loader
        self.assertIsInstance(train_loader_only, torch.utils.data.DataLoader)
        self.assertEqual(len(train_loader_only.dataset), 100)  # All samples
    
    def test_classification_trainer_init(self):
        """Test ClassificationTrainer initialization."""
        trainer = ClassificationTrainer(
            model=self.model,
            dataset=self.dataset,
            batch_size=16,
            lr=0.001,
            epochs=2,
            device='cpu',
            save_path=self.temp_dir
        )
        
        # Check that the trainer has the expected attributes
        self.assertEqual(trainer.batch_size, 16)
        self.assertEqual(trainer.lr, 0.001)
        self.assertEqual(trainer.epochs, 2)
        self.assertEqual(trainer.device, 'cpu')
        self.assertEqual(trainer.save_path, self.temp_dir)
        
        # Check that the model was moved to the correct device
        self.assertEqual(next(self.model.parameters()).device.type, 'cpu')
        
        # Check that the dataloaders were created
        self.assertIsNotNone(trainer.train_loader)
        self.assertIsNotNone(trainer.valid_loader)
    
    def test_classification_trainer_train_step(self):
        """Test a single training step of ClassificationTrainer."""
        trainer = ClassificationTrainer(
            model=self.model,
            dataset=self.dataset,
            batch_size=16,
            lr=0.001,
            epochs=1,
            device='cpu',
            save_path=self.temp_dir
        )
        
        # Get a batch from the data loader
        inputs, targets = next(iter(trainer.train_loader))
        
        # Run a single training step
        loss = trainer._train_step(inputs, targets)
        
        # Check that loss is a scalar and not NaN
        self.assertIsInstance(loss, float)
        self.assertFalse(torch.isnan(torch.tensor(loss)))
    
    def test_classification_trainer_train_epoch(self):
        """Test a single training epoch of ClassificationTrainer."""
        trainer = ClassificationTrainer(
            model=self.model,
            dataset=self.dataset,
            batch_size=16,
            lr=0.001,
            epochs=1,
            device='cpu',
            save_path=self.temp_dir
        )
        
        # Run a single training epoch
        train_loss = trainer._train_epoch()
        
        # Check that loss is a scalar and not NaN
        self.assertIsInstance(train_loss, float)
        self.assertFalse(torch.isnan(torch.tensor(train_loss)))
    
    def test_classification_trainer_evaluate(self):
        """Test evaluation with ClassificationTrainer."""
        trainer = ClassificationTrainer(
            model=self.model,
            dataset=self.dataset,
            batch_size=16,
            lr=0.001,
            epochs=1,
            device='cpu',
            save_path=self.temp_dir
        )
        
        # Set model to eval mode
        self.model.eval()
        
        # Evaluate on the validation set
        val_loss, val_acc = trainer.evaluate(trainer.valid_loader)
        
        # Check that metrics are scalars and not NaN
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertFalse(torch.isnan(torch.tensor(val_loss)))
        self.assertFalse(torch.isnan(torch.tensor(val_acc)))
        
        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)
    
    def test_classification_trainer_train(self):
        """Test the full training loop of ClassificationTrainer."""
        trainer = ClassificationTrainer(
            model=self.model,
            dataset=self.dataset,
            batch_size=16,
            lr=0.001,
            epochs=2,  # Use a small number of epochs for testing
            device='cpu',
            save_path=self.temp_dir
        )
        
        # Train the model (only 2 epochs)
        history = trainer.train()
        
        # Check that history has the expected metrics
        self.assertIn('train_loss', history)
        self.assertIn('valid_loss', history)
        self.assertIn('valid_acc', history)
        
        # Check that metrics are lists with the expected length
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['valid_loss']), 2)
        self.assertEqual(len(history['valid_acc']), 2)
        
        # Check that a checkpoint was saved
        checkpoint_path = os.path.join(self.temp_dir, 'best_model.pt')
        self.assertTrue(os.path.exists(checkpoint_path))
    
    def test_pruning_trainer(self):
        """Basic test for PruningTrainer initialization and methods."""
        trainer = PruningTrainer(
            model=self.model,
            dataset=self.dataset,
            batch_size=16,
            lr=0.001,
            epochs=2,
            device='cpu',
            save_path=self.temp_dir,
            prune_rate=0.3,
            prune_interval=1
        )
        
        # Check pruning-specific attributes
        self.assertEqual(trainer.prune_rate, 0.3)
        self.assertEqual(trainer.prune_interval, 1)
        
        # Train for a single epoch to trigger pruning
        history = trainer.train(verbose=False)
        
        # Check that history has pruning-related metrics
        self.assertIn('train_loss', history)
        self.assertIn('valid_loss', history)
        self.assertIn('valid_acc', history)
        
        # Check that a pruned checkpoint was saved
        pruned_checkpoint_path = os.path.join(self.temp_dir, 'pruned_model.pt')
        self.assertTrue(os.path.exists(pruned_checkpoint_path))

if __name__ == '__main__':
    unittest.main() 