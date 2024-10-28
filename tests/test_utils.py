import unittest

import torch

from dendritic_modeling.utils import Shaper

class TestShaper(unittest.TestCase):
    def test_initialization(self):
        shape = (2, 3)
        shaper = Shaper(shape)
        self.assertEqual(shaper.shape, shape)

    def test_reshape_valid_shape(self):
        shaper = Shaper((2, 3))
        x = torch.randn(6)  
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 3]))

    def test_reshape_invalid_shape(self):
        shaper = Shaper((2, 4))  
        x = torch.randn(6)  
        with self.assertRaises(RuntimeError):
            shaper.reshape(x)  

    def test_reshape_higher_dimensions(self):
        shaper = Shaper((2, 2, 3))
        x = torch.randn(12)  
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 2, 3]))


if __name__ == '__main__':
    unittest.main()