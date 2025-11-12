"""
Manual Verification Tests

Comprehensive tests comparing neural network computations with manual calculations.
Tests forward pass, backward pass, and weight updates step by step.

Author: ANN from Scratch Team
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from backend.core import NeuralNetwork, Sigmoid, MeanSquaredError


class TestManualVerification(unittest.TestCase):
    """
    Test suite to verify that computations match manual calculations.

    This ensures correctness of:
    - Forward propagation
    - Activation functions
    - Loss calculation
    - Backpropagation
    - Weight updates
    """

    def test_forward_pass_manual(self):
        """
        Test forward pass with manual calculation verification

        Network: 2-2-1
        Input: [0.5, 0.3]
        """
        print("\n" + "="*60)
        print("TEST: Forward Pass Manual Verification")
        print("="*60)

        # Build simple network
        network = NeuralNetwork()
        network.add_layer(2, 'linear')
        network.add_layer(2, 'sigmoid')
        network.add_layer(1, 'sigmoid')

        # Set known weights
        network.set_connections(
            1,
            [[0, 1], [0, 1]],
            [[0.5, 0.3], [0.2, 0.4]],
            [0.1, -0.1]
        )
        network.set_connections(
            2,
            [[0, 1]],
            [[0.6, 0.7]],
            [0.2]
        )

        # Input
        X = np.array([[0.5, 0.3]])

        print("\nNetwork Architecture:")
        print("  Layer 0 (Input): 2 nodes")
        print("  Layer 1 (Hidden): 2 nodes, sigmoid")
        print("  Layer 2 (Output): 1 node, sigmoid")

        print("\nWeights Layer 1:")
        print(f"  Node 0: w=[0.5, 0.3], b=0.1")
        print(f"  Node 1: w=[0.2, 0.4], b=-0.1")

        print("\nWeights Layer 2:")
        print(f"  Node 0: w=[0.6, 0.7], b=0.2")

        # Manual calculation - Layer 1
        print("\n" + "-"*60)
        print("MANUAL CALCULATION - Layer 1 (Hidden)")
        print("-"*60)

        z1_0 = 0.5 * 0.5 + 0.3 * 0.3 + 0.1  # 0.25 + 0.09 + 0.1 = 0.44
        z1_1 = 0.2 * 0.5 + 0.4 * 0.3 - 0.1  # 0.1 + 0.12 - 0.1 = 0.12

        print(f"z1[0] = 0.5*0.5 + 0.3*0.3 + 0.1 = {z1_0:.6f}")
        print(f"z1[1] = 0.2*0.5 + 0.4*0.3 - 0.1 = {z1_1:.6f}")

        # Sigmoid activation
        sigmoid = Sigmoid()
        a1_0 = float(sigmoid.forward(np.array([z1_0]))[0])
        a1_1 = float(sigmoid.forward(np.array([z1_1]))[0])

        print(f"a1[0] = sigmoid({z1_0:.6f}) = {a1_0:.6f}")
        print(f"a1[1] = sigmoid({z1_1:.6f}) = {a1_1:.6f}")

        # Manual calculation - Layer 2
        print("\n" + "-"*60)
        print("MANUAL CALCULATION - Layer 2 (Output)")
        print("-"*60)

        z2_0 = 0.6 * a1_0 + 0.7 * a1_1 + 0.2
        print(f"z2[0] = 0.6*{a1_0:.6f} + 0.7*{a1_1:.6f} + 0.2 = {z2_0:.6f}")

        a2_0 = float(sigmoid.forward(np.array([z2_0]))[0])
        print(f"a2[0] = sigmoid({z2_0:.6f}) = {a2_0:.6f}")

        # Network calculation
        print("\n" + "-"*60)
        print("NETWORK CALCULATION")
        print("-"*60)

        y_pred = network.forward(X)
        print(f"Network output: {y_pred[0][0]:.6f}")

        # Verify
        print("\n" + "-"*60)
        print("VERIFICATION")
        print("-"*60)

        print(f"Manual output:  {a2_0:.6f}")
        print(f"Network output: {y_pred[0][0]:.6f}")
        print(f"Difference:     {abs(a2_0 - y_pred[0][0]):.10f}")

        self.assertAlmostEqual(a2_0, y_pred[0][0], places=6,
                               msg="Forward pass doesn't match manual calculation")

        print("\n[OK] Forward pass verified: Manual == Network")

    def test_backward_pass_manual(self):
        """
        Test backward pass with manual calculation verification

        Verifies gradient computation for a simple network.
        """
        print("\n" + "="*60)
        print("TEST: Backward Pass Manual Verification")
        print("="*60)

        # Build simple network (2-2-1)
        network = NeuralNetwork()
        network.add_layer(2, 'linear')
        network.add_layer(2, 'sigmoid')
        network.add_layer(1, 'sigmoid')

        # Set known weights
        network.set_connections(
            1,
            [[0, 1], [0, 1]],
            [[0.5, 0.3], [0.2, 0.4]],
            [0.1, -0.1]
        )
        network.set_connections(
            2,
            [[0, 1]],
            [[0.6, 0.7]],
            [0.2]
        )

        # Single sample
        X = np.array([[1.0, 0.5]])
        y_true = np.array([[1.0]])

        print("\nInput: [1.0, 0.5]")
        print("Target: [1.0]")

        # Forward pass
        y_pred = network.forward(X)
        print(f"\nPrediction: {y_pred[0][0]:.6f}")

        # Get layer outputs
        a0 = network.layer_outputs[0][0]  # Input
        a1 = network.layer_outputs[1][0]  # Hidden
        a2 = network.layer_outputs[2][0]  # Output

        z1 = network.layer_z_values[1][0]  # Hidden pre-activation
        z2 = network.layer_z_values[2][0]  # Output pre-activation

        print(f"\nLayer activations:")
        print(f"  a0 (input): {a0}")
        print(f"  a1 (hidden): {a1}")
        print(f"  a2 (output): {a2}")

        # Manual backward pass calculation
        print("\n" + "-"*60)
        print("MANUAL BACKWARD CALCULATION (MSE Loss)")
        print("-"*60)

        # Loss derivative: dL/da = 2(a - y)/n
        dL_da2 = 2 * (a2 - y_true[0]) / y_true.size
        print(f"\ndL/da2 = 2 * ({a2[0]:.6f} - {y_true[0][0]:.6f}) / 1")
        print(f"       = {dL_da2[0]:.6f}")

        # Activation derivative: da/dz = sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        da2_dz2 = a2 * (1 - a2)
        print(f"\nda2/dz2 = {a2[0]:.6f} * (1 - {a2[0]:.6f})")
        print(f"        = {da2_dz2[0]:.6f}")

        # Output layer delta
        delta2 = dL_da2 * da2_dz2
        print(f"\ndelta2 = dL/da2 * da2/dz2")
        print(f"       = {dL_da2[0]:.6f} * {da2_dz2[0]:.6f}")
        print(f"       = {delta2[0]:.6f}")

        # Weight gradients for layer 2
        print(f"\nWeight gradients Layer 2:")
        dL_dw2_0 = delta2[0] * a1[0]
        dL_dw2_1 = delta2[0] * a1[1]
        dL_db2 = delta2[0]

        print(f"  dL/dw2[0] = delta2 * a1[0] = {delta2[0]:.6f} * {a1[0]:.6f} = {dL_dw2_0:.6f}")
        print(f"  dL/dw2[1] = delta2 * a1[1] = {delta2[0]:.6f} * {a1[1]:.6f} = {dL_dw2_1:.6f}")
        print(f"  dL/db2 = delta2 = {dL_db2:.6f}")

        # Backprop to hidden layer
        w2_0 = 0.6
        w2_1 = 0.7
        dL_da1_0 = delta2[0] * w2_0
        dL_da1_1 = delta2[0] * w2_1

        print(f"\nBackprop to hidden layer:")
        print(f"  dL/da1[0] = delta2 * w2[0] = {delta2[0]:.6f} * {w2_0} = {dL_da1_0:.6f}")
        print(f"  dL/da1[1] = delta2 * w2[1] = {delta2[0]:.6f} * {w2_1} = {dL_da1_1:.6f}")

        # Network backward pass
        print("\n" + "-"*60)
        print("NETWORK BACKWARD CALCULATION")
        print("-"*60)

        loss_fn = MeanSquaredError()
        weight_grads, bias_grads = network.backward(y_true, y_pred, loss_fn)

        print(f"\nNetwork gradients Layer 2:")
        print(f"  dL/dw2[0] = {weight_grads[2][0][0]:.6f}")
        print(f"  dL/dw2[1] = {weight_grads[2][0][1]:.6f}")
        print(f"  dL/db2 = {bias_grads[2][0]:.6f}")

        # Verify
        print("\n" + "-"*60)
        print("VERIFICATION")
        print("-"*60)

        self.assertAlmostEqual(dL_dw2_0, weight_grads[2][0][0], places=6)
        self.assertAlmostEqual(dL_dw2_1, weight_grads[2][0][1], places=6)
        self.assertAlmostEqual(dL_db2, bias_grads[2][0], places=6)

        print("\n[OK] Backward pass verified: Manual == Network")
        print(f"  Weight gradient [0] match: {abs(dL_dw2_0 - weight_grads[2][0][0]) < 1e-6}")
        print(f"  Weight gradient [1] match: {abs(dL_dw2_1 - weight_grads[2][0][1]) < 1e-6}")
        print(f"  Bias gradient match: {abs(dL_db2 - bias_grads[2][0]) < 1e-6}")

    def test_weight_update_manual(self):
        """
        Test weight update with manual calculation verification

        Verifies that weight update follows: w_new = w_old - lr * gradient
        """
        print("\n" + "="*60)
        print("TEST: Weight Update Manual Verification")
        print("="*60)

        # Build simple network
        network = NeuralNetwork()
        network.add_layer(2, 'linear')
        network.add_layer(1, 'sigmoid')

        # Set initial weights
        w_init = [[0.5, 0.3]]
        b_init = [0.1]

        network.set_connections(1, [[0, 1]], w_init, b_init)

        print("\nInitial weights:")
        print(f"  w = {w_init[0]}")
        print(f"  b = {b_init}")

        # Data
        X = np.array([[1.0, 0.5]])
        y = np.array([[1.0]])
        learning_rate = 0.1

        print(f"\nLearning rate: {learning_rate}")
        print(f"Input: {X[0]}")
        print(f"Target: {y[0][0]}")

        # Forward pass
        y_pred = network.forward(X)
        print(f"Initial prediction: {y_pred[0][0]:.6f}")

        # Calculate gradients
        loss_fn = MeanSquaredError()
        weight_grads, bias_grads = network.backward(y, y_pred, loss_fn)

        print(f"\nGradients:")
        print(f"  dL/dw[0] = {weight_grads[1][0][0]:.6f}")
        print(f"  dL/dw[1] = {weight_grads[1][0][1]:.6f}")
        print(f"  dL/db = {bias_grads[1][0]:.6f}")

        # Manual weight update
        print("\n" + "-"*60)
        print("MANUAL WEIGHT UPDATE")
        print("-"*60)

        w_new_0 = w_init[0][0] - learning_rate * weight_grads[1][0][0]
        w_new_1 = w_init[0][1] - learning_rate * weight_grads[1][0][1]
        b_new = b_init[0] - learning_rate * bias_grads[1][0]

        print(f"\nw_new[0] = {w_init[0][0]} - {learning_rate} * {weight_grads[1][0][0]:.6f}")
        print(f"         = {w_new_0:.6f}")

        print(f"\nw_new[1] = {w_init[0][1]} - {learning_rate} * {weight_grads[1][0][1]:.6f}")
        print(f"         = {w_new_1:.6f}")

        print(f"\nb_new = {b_init[0]} - {learning_rate} * {bias_grads[1][0]:.6f}")
        print(f"      = {b_new:.6f}")

        # Train for 1 epoch
        network.train(X, y, epochs=1, learning_rate=learning_rate, verbose=False)

        # Get updated weights
        w_network_0 = network.weights[1][0][0]
        w_network_1 = network.weights[1][0][1]
        b_network = network.biases[1][0]

        print("\n" + "-"*60)
        print("NETWORK WEIGHT UPDATE")
        print("-"*60)

        print(f"\nw_network[0] = {w_network_0:.6f}")
        print(f"w_network[1] = {w_network_1:.6f}")
        print(f"b_network = {b_network:.6f}")

        # Verify
        print("\n" + "-"*60)
        print("VERIFICATION")
        print("-"*60)

        self.assertAlmostEqual(w_new_0, w_network_0, places=6)
        self.assertAlmostEqual(w_new_1, w_network_1, places=6)
        self.assertAlmostEqual(b_new, b_network, places=6)

        print("\n[OK] Weight update verified: Manual == Network")
        print(f"  Weight [0] match: {abs(w_new_0 - w_network_0) < 1e-6}")
        print(f"  Weight [1] match: {abs(w_new_1 - w_network_1) < 1e-6}")
        print(f"  Bias match: {abs(b_new - b_network) < 1e-6}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
