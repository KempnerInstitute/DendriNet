import unittest

import numpy as np
import torch

from dendritic_modeling.synthetic_datasets import (CenterSurroundDataset,
                                                   LineDataset,
                                                   MultiXORDataset,
                                                   OrientationBarsDataset)


class TestSyntheticDatasets(unittest.TestCase):
    def test_multi_xor_dataset(self):
        """Test MultiXORDataset generation and properties."""
        n_samples = 100
        n_bits = 8
        dataset = MultiXORDataset(n_samples=n_samples, n_bits=n_bits)

        # Check dataset size
        self.assertEqual(len(dataset), n_samples)

        # Check item format
        x, y = dataset[0]
        self.assertEqual(x.shape, (n_bits,))  # n_bits input bits
        self.assertIn(y.item(), [0, 1])  # Binary target

        # Check a few samples manually to verify XOR logic
        for i in range(min(10, n_samples)):
            x, y = dataset[i]
            x_np = x.numpy()
            # Verify that XOR logic is applied: odd number of 1s should result in y=1
            num_ones = np.sum(x_np == 1)
            expected_y = 1 if num_ones % 2 == 1 else 0
            self.assertEqual(y.item(), expected_y, f"XOR logic failed for sample {i}")

    def test_line_dataset(self):
        """Test LineDataset generation and properties."""
        n_samples = 100
        image_size = 16
        n_classes = 2
        dataset = LineDataset(
            n_samples=n_samples, image_size=image_size, n_classes=n_classes
        )

        # Check dataset size
        self.assertEqual(len(dataset), n_samples)

        # Check item format
        x, y = dataset[0]
        self.assertEqual(
            x.shape, (1, image_size, image_size)
        )  # 1-channel image of specified size
        self.assertLess(y.item(), n_classes)  # Class index should be within range

        # Check that images are binary (contain only 0s and 1s)
        for i in range(min(10, n_samples)):
            x, _ = dataset[i]
            unique_vals = torch.unique(x)
            self.assertTrue(
                torch.all(torch.isin(unique_vals, torch.tensor([0.0, 1.0])))
            )

    def test_center_surround_dataset(self):
        """Test CenterSurroundDataset generation and properties."""
        n_samples = 100
        image_size = 16
        n_classes = 4
        dataset = CenterSurroundDataset(
            n_samples=n_samples, image_size=image_size, n_classes=n_classes
        )

        # Check dataset size
        self.assertEqual(len(dataset), n_samples)

        # Check item format
        x, y = dataset[0]
        self.assertEqual(
            x.shape, (1, image_size, image_size)
        )  # 1-channel image of specified size
        self.assertLess(y.item(), n_classes)  # Class index should be within range

        # Verify that images have the center-surround pattern
        # (should have some values that are not 0 or 1, since it creates a gradient)
        has_gradient = False
        for i in range(min(10, n_samples)):
            x, _ = dataset[i]
            if len(torch.unique(x)) > 2:
                has_gradient = True
                break

        self.assertTrue(
            has_gradient, "CenterSurroundDataset should create gradient images"
        )

    def test_orientation_bars_dataset(self):
        """Test OrientationBarsDataset generation and properties."""
        n_samples = 100
        image_size = 16
        n_orient_classes = 8
        dataset = OrientationBarsDataset(
            n_samples=n_samples,
            image_size=image_size,
            n_orient_classes=n_orient_classes,
        )

        # Check dataset size
        self.assertEqual(len(dataset), n_samples)

        # Check item format
        x, y = dataset[0]
        self.assertEqual(
            x.shape, (1, image_size, image_size)
        )  # 1-channel image of specified size
        self.assertLess(
            y.item(), n_orient_classes
        )  # Class index should be within range

        # Check that the orientation is encoded in the class
        # (difficult to verify exactly, but let's ensure samples have varying classes)
        classes = set()
        for i in range(min(20, n_samples)):
            _, y = dataset[i]
            classes.add(y.item())

        # There should be multiple classes
        self.assertGreater(
            len(classes),
            1,
            "OrientationBarsDataset should have multiple orientation classes",
        )

    def test_dataset_option_consistency(self):
        """Test that dataset parameters are respected consistently."""
        # Test that changing the image size changes the output
        ds1 = LineDataset(n_samples=10, image_size=16)
        ds2 = LineDataset(n_samples=10, image_size=32)

        # Images should have different sizes
        self.assertEqual(ds1[0][0].shape, (1, 16, 16))
        self.assertEqual(ds2[0][0].shape, (1, 32, 32))

        # Test that changing the number of classes changes the output range
        ds1 = OrientationBarsDataset(n_samples=20, n_orient_classes=4)
        ds2 = OrientationBarsDataset(n_samples=20, n_orient_classes=8)

        # Collect all classes from each dataset
        classes1 = set()
        classes2 = set()
        for i in range(20):
            classes1.add(ds1[i][1].item())
            classes2.add(ds2[i][1].item())

        # Check class ranges
        self.assertTrue(all(c < 4 for c in classes1))
        self.assertTrue(any(c >= 4 for c in classes2))

    def test_deterministic_with_seed(self):
        """Test that datasets are deterministic with a fixed seed."""
        seed = 42

        # Create datasets with the same seed
        ds1 = MultiXORDataset(n_samples=10, n_bits=8, seed=seed)
        ds2 = MultiXORDataset(n_samples=10, n_bits=8, seed=seed)

        # Datasets with the same seed should have identical data
        for i in range(10):
            x1, y1 = ds1[i]
            x2, y2 = ds2[i]
            self.assertTrue(
                torch.all(x1 == x2), f"Sample {i} inputs differ despite same seed"
            )
            self.assertEqual(
                y1.item(), y2.item(), f"Sample {i} labels differ despite same seed"
            )

        # Create a dataset with a different seed
        ds3 = MultiXORDataset(n_samples=10, n_bits=8, seed=seed + 1)

        # At least some samples should differ
        any_different = False
        for i in range(10):
            x1, _ = ds1[i]
            x3, _ = ds3[i]
            if not torch.all(x1 == x3):
                any_different = True
                break

        self.assertTrue(
            any_different, "Different seeds should produce different datasets"
        )


if __name__ == "__main__":
    unittest.main()
