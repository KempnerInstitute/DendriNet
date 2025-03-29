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
        self.assertIn(
            y.item(), [0, 1, 2, 3]
        )  # Binary target (2 bits = 4 possibilities)

        # Check a few samples manually to verify XOR logic
        for i in range(min(10, n_samples)):
            x, y = dataset[i]
            x_np = x.numpy()
            # Verify that XOR logic is applied correctly:
            # y1 = bits[0] ^ bits[1]
            # y2 = bits[2] ^ bits[3] ^ bits[4]
            # label = (y1 << 1) + y2
            y1 = int(x_np[0]) ^ int(x_np[1])
            y2 = int(x_np[2]) ^ int(x_np[3]) ^ int(x_np[4])
            expected_y = (y1 << 1) + y2
            self.assertEqual(y.item(), expected_y, f"XOR logic failed for sample {i}")

    def test_line_dataset(self):
        """Test LineDataset generation and properties."""
        n_samples = 100
        image_size = 16
        dataset = LineDataset(n_samples=n_samples, image_size=image_size)

        # Check dataset size
        self.assertEqual(len(dataset), n_samples)

        # Check item format
        x, y = dataset[0]
        self.assertEqual(x.shape, (image_size * image_size,))  # Flattened image
        self.assertIn(y.item(), [0, 1, 2])  # Class index should be 0, 1, or 2

        # Check that images have expected format (zeros and ones)
        for i in range(min(10, n_samples)):
            x, y = dataset[i]
            x_reshaped = x.reshape(image_size, image_size)

            # For class 1, there should be a horizontal line (a row of ones)
            if y.item() == 1:
                # Find the row with sum == image_size (all ones)
                row_sums = x_reshaped.sum(dim=1)
                self.assertTrue(
                    any(row_sums == image_size),
                    f"No horizontal line found for class 1 sample {i}",
                )

            # For class 2, there should be a vertical line (a column of ones)
            if y.item() == 2:
                # Find the column with sum == image_size (all ones)
                col_sums = x_reshaped.sum(dim=0)
                self.assertTrue(
                    any(col_sums == image_size),
                    f"No vertical line found for class 2 sample {i}",
                )

    def test_center_surround_dataset(self):
        """Test CenterSurroundDataset generation and properties."""
        n_samples = 100
        image_size = 16
        dataset = CenterSurroundDataset(n_samples=n_samples, image_size=image_size)

        # Check dataset size
        self.assertEqual(len(dataset), n_samples)

        # Check item format
        x, y = dataset[0]
        self.assertEqual(x.shape, (image_size * image_size,))  # Flattened image
        self.assertIn(y.item(), [0, 1, 2])  # Class index should be 0, 1, or 2

        # Check class-specific properties
        for i in range(min(10, n_samples)):
            x, y = dataset[i]
            x_reshaped = x.reshape(image_size, image_size)

            if y.item() == 0:
                # Class 0 should have similar values throughout (uniform)
                std = x_reshaped.std().item()
                self.assertLess(
                    std, 0.1, f"Class 0 should have uniform values, sample {i}"
                )

            elif y.item() == 1:
                # Class 1 should have higher values in the center
                center = image_size // 2
                center_val = x_reshaped[center, center].item()
                edge_val = x_reshaped[0, 0].item()
                self.assertGreater(
                    center_val,
                    edge_val,
                    f"Center should be higher than edge for class 1, sample {i}",
                )

            elif y.item() == 2:
                # Class 2 should have lower values in the center
                center = image_size // 2
                center_val = x_reshaped[center, center].item()
                edge_val = x_reshaped[0, 0].item()
                self.assertGreater(
                    edge_val,
                    center_val,
                    f"Edge should be higher than center for class 2, sample {i}",
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
        self.assertEqual(x.shape, (image_size * image_size,))  # Flattened image
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

        # Images should have different shapes
        self.assertEqual(ds1[0][0].shape, (16 * 16,))
        self.assertEqual(ds2[0][0].shape, (32 * 32,))

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
        # Since the current implementation doesn't accept a seed parameter,
        # we'll set the numpy random seed manually
        seed = 42
        np.random.seed(seed)
        ds1 = MultiXORDataset(n_samples=10, n_bits=8)

        np.random.seed(seed)
        ds2 = MultiXORDataset(n_samples=10, n_bits=8)

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
        np.random.seed(seed + 1)
        ds3 = MultiXORDataset(n_samples=10, n_bits=8)

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
