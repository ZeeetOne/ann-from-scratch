"""
Unit Tests for Activation Functions

Tests all activation functions for correctness.

Author: ANN from Scratch Team
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from backend.core import Sigmoid, ReLU, Linear, Softmax, Threshold, ActivationFactory


class TestActivationFunctions(unittest.TestCase):
    """Test activation functions"""

    def test_sigmoid_forward(self):
        """Test sigmoid forward pass"""
        sigmoid = Sigmoid()

        # Test with single value
        x = np.array([0.0])
        result = sigmoid.forward(x)
        self.assertAlmostEqual(result[0], 0.5, places=5)

        # Test with array
        x = np.array([[-500, 0, 500]])
        result = sigmoid.forward(x)
        self.assertAlmostEqual(result[0, 0], 0.0, places=5)
        self.assertAlmostEqual(result[0, 1], 0.5, places=5)
        self.assertAlmostEqual(result[0, 2], 1.0, places=5)

    def test_sigmoid_derivative(self):
        """Test sigmoid derivative"""
        sigmoid = Sigmoid()

        x = np.array([0.0])
        result = sigmoid.derivative(x)
        self.assertAlmostEqual(result[0], 0.25, places=5)

    def test_relu_forward(self):
        """Test ReLU forward pass"""
        relu = ReLU()

        x = np.array([[-2, -1, 0, 1, 2]])
        result = relu.forward(x)
        expected = np.array([[0, 0, 0, 1, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_relu_derivative(self):
        """Test ReLU derivative"""
        relu = ReLU()

        x = np.array([[-2, -1, 0, 1, 2]])
        result = relu.derivative(x)
        expected = np.array([[0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_linear_forward(self):
        """Test linear activation"""
        linear = Linear()

        x = np.array([[-2, -1, 0, 1, 2]])
        result = linear.forward(x)
        np.testing.assert_array_equal(result, x)

    def test_softmax_forward(self):
        """Test softmax forward pass"""
        softmax = Softmax()

        x = np.array([[1, 2, 3]])
        result = softmax.forward(x)

        # Check sum to 1
        self.assertAlmostEqual(np.sum(result), 1.0, places=5)

        # Check all positive
        self.assertTrue(np.all(result > 0))

        # Check monotonic (larger input -> larger output)
        self.assertTrue(result[0, 0] < result[0, 1] < result[0, 2])

    def test_activation_factory(self):
        """Test activation factory"""
        sigmoid = ActivationFactory.create('sigmoid')
        self.assertIsInstance(sigmoid, Sigmoid)

        relu = ActivationFactory.create('ReLU')
        self.assertIsInstance(relu, ReLU)

        # Test invalid activation
        with self.assertRaises(ValueError):
            ActivationFactory.create('invalid')

    def test_threshold(self):
        """Test threshold activation"""
        threshold = Threshold(threshold=0.5)

        x = np.array([[0, 0.3, 0.5, 0.7, 1.0]])
        result = threshold.forward(x)
        expected = np.array([[0, 0, 0, 1, 1]])
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
