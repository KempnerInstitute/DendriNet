import unittest
from dendritic_modeling.evaluation import SimpleSum

class TestSimpleSum(unittest.TestCase):
    
    def test_sum_positive_numbers(self):
        """Test summing a list of positive numbers"""
        obj = SimpleSum([1, 2, 3, 4, 5])
        self.assertEqual(obj.sum(), 15)

    def test_sum_negative_numbers(self):
        """Test summing a list of negative numbers"""
        obj = SimpleSum([-1, -2, -3, -4, -5])
        self.assertEqual(obj.sum(), -15)

    def test_sum_mixed_numbers(self):
        """Test summing a list of mixed positive and negative numbers"""
        obj = SimpleSum([1, -2, 3, -4, 5])
        self.assertEqual(obj.sum(), 3)

    def test_sum_empty_list(self):
        """Test summing an empty list"""
        obj = SimpleSum([])
        self.assertEqual(obj.sum(), 0)

    def test_sum_with_floats(self):
        """Test summing a list of floats"""
        obj = SimpleSum([1.5, 2.5, 3.0])
        self.assertAlmostEqual(obj.sum(), 7.0)

if __name__ == '__main__':
    unittest.main()
