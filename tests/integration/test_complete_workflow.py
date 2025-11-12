"""
Integration Test for Complete Workflow

Tests end-to-end functionality: build, forward, backward, train.

Author: ANN from Scratch Team
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from backend.core import NeuralNetwork


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete workflow from building to training"""

    def test_and_gate_training(self):
        """
        Test training AND gate with known results.
        This verifies the entire pipeline works correctly.
        """
        # Build network (2-2-1)
        network = NeuralNetwork()
        network.add_layer(2, 'linear')  # Input
        network.add_layer(2, 'sigmoid')  # Hidden
        network.add_layer(1, 'sigmoid')  # Output

        # Set initial weights
        network.set_connections(
            1,
            [[0, 1], [0, 1]],
            [[0.5, -0.3], [-0.4, 0.6]],
            [0.1, -0.2]
        )

        network.set_connections(
            2,
            [[0, 1]],
            [[0.8, -0.5]],
            [0.2]
        )

        # Prepare AND gate data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])

        # Test forward pass (before training)
        y_pred_before = network.forward(X)
        self.assertEqual(y_pred_before.shape, (4, 1))

        # Test training
        history = network.train(
            X, y,
            epochs=500,
            learning_rate=0.5,
            optimizer='gd',
            loss_function='mse',
            verbose=False
        )

        # Verify training reduces loss
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        self.assertLess(final_loss, initial_loss)

        # Test predictions after training
        y_pred_after, y_pred_probs = network.predict(X, threshold=0.5)

        # Verify predictions are correct (or close)
        # For AND gate: [0, 0] -> 0, [0, 1] -> 0, [1, 0] -> 0, [1, 1] -> 1
        accuracy = np.mean((y_pred_after == y).astype(float))
        self.assertGreater(accuracy, 0.5)  # At least 50% accuracy

        print(f"\nAND Gate Training Test:")
        print(f"Initial Loss: {initial_loss:.6f}")
        print(f"Final Loss: {final_loss:.6f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Predictions:")
        for i in range(len(X)):
            print(f"  {X[i]} -> {y_pred_probs[i][0]:.4f} (target: {y[i][0]})")

    def test_forward_backward_consistency(self):
        """
        Test that forward and backward passes are consistent.
        Gradients should reduce loss when applied correctly.
        """
        # Build simple network
        network = NeuralNetwork()
        network.add_layer(2, 'linear')
        network.add_layer(2, 'sigmoid')
        network.add_layer(1, 'sigmoid')

        network.set_connections(
            1,
            [[0, 1], [0, 1]],
            [[0.5, 0.5], [0.5, 0.5]],
            [0.0, 0.0]
        )

        network.set_connections(
            2,
            [[0, 1]],
            [[0.5, 0.5]],
            [0.0]
        )

        # Single sample
        X = np.array([[1.0, 0.0]])
        y = np.array([[1.0]])

        # Forward pass
        from backend.core import MeanSquaredError
        loss_fn = MeanSquaredError()

        y_pred = network.forward(X)
        loss_before = loss_fn.calculate(y, y_pred)

        # Backward pass
        weight_grads, bias_grads = network.backward(y, y_pred, loss_fn)

        # Manually update weights (gradient descent step)
        learning_rate = 0.1
        for layer_idx in range(1, len(network.layers)):
            if layer_idx < len(weight_grads):
                for node_idx in range(len(weight_grads[layer_idx])):
                    if weight_grads[layer_idx][node_idx]:
                        for w_idx in range(len(weight_grads[layer_idx][node_idx])):
                            network.weights[layer_idx][node_idx][w_idx] -= (
                                learning_rate * weight_grads[layer_idx][node_idx][w_idx]
                            )
                        network.biases[layer_idx][node_idx] -= (
                            learning_rate * bias_grads[layer_idx][node_idx]
                        )

        # Forward pass again
        y_pred_after = network.forward(X)
        loss_after = loss_fn.calculate(y, y_pred_after)

        # Loss should decrease (or stay same if already at minimum)
        self.assertLessEqual(loss_after, loss_before)

        print(f"\nForward-Backward Consistency Test:")
        print(f"Loss before update: {loss_before:.6f}")
        print(f"Loss after update: {loss_after:.6f}")
        print(f"Loss reduction: {loss_before - loss_after:.6f}")

    def test_multiclass_classification(self):
        """
        Test multi-class classification with softmax.
        """
        # Build network (3-4-2 with softmax)
        network = NeuralNetwork()
        network.add_layer(3, 'linear')
        network.add_layer(4, 'sigmoid')
        network.add_layer(2, 'softmax')

        # Set connections
        network.set_connections(
            1,
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
            [[0.5, 0.3, -0.2], [-0.4, 0.6, 0.1], [0.2, -0.5, 0.4], [0.7, 0.2, -0.3]],
            [0.1, -0.2, 0.3, 0.0]
        )

        network.set_connections(
            2,
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0.8, -0.3, 0.6, 0.4], [-0.5, 0.7, -0.2, 0.3]],
            [0.1, -0.1]
        )

        # Test data (2 classes)
        X = np.array([
            [0.5, 0.5, 0.8],  # Class 0
            [0.2, 0.3, 0.9],  # Class 0
            [0.8, 0.8, 0.1],  # Class 1
            [0.9, 0.9, 0.0]   # Class 1
        ])
        y = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ])

        # Forward pass
        y_pred = network.forward(X)

        # Check softmax properties
        # Sum should be 1 for each sample
        sums = np.sum(y_pred, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(4), decimal=5)

        # All probabilities should be between 0 and 1
        self.assertTrue(np.all(y_pred >= 0))
        self.assertTrue(np.all(y_pred <= 1))

        # Train
        history = network.train(
            X, y,
            epochs=200,
            learning_rate=0.5,
            optimizer='sgd',
            loss_function='categorical',
            verbose=False
        )

        # Verify loss decreases
        self.assertLess(history['loss'][-1], history['loss'][0])

        print(f"\nMulti-Class Classification Test:")
        print(f"Initial Loss: {history['loss'][0]:.6f}")
        print(f"Final Loss: {history['loss'][-1]:.6f}")

        # Make predictions
        y_pred_after, _ = network.predict(X)
        accuracy = np.mean((y_pred_after == y).astype(float))
        print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    unittest.main()
