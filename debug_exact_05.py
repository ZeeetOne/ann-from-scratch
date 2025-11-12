"""
Find exact scenario where predictions stuck at EXACTLY 0.5
"""
import numpy as np
from ann_core import NeuralNetwork

print("=" * 70)
print("DEBUG: Find Exact 0.5 Stuck Scenario")
print("=" * 70)

def test_scenario(X, y, lr, epochs, desc):
    """Test with specific parameters"""
    print(f"\n{desc}")
    print(f"  Data: {X.shape}, LR: {lr}, Epochs: {epochs}")

    # Create network
    nn = NeuralNetwork()
    nn.add_layer(2, 'linear')
    nn.add_layer(3, 'relu')
    nn.add_layer(1, 'sigmoid')

    # Set connections
    connections_layer1 = [[0, 1], [0, 1], [0, 1]]
    weights_layer1 = [[0.5, 0.3], [-0.4, 0.6], [0.2, -0.5]]
    biases_layer1 = [0.1, -0.2, 0.3]
    nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

    connections_layer2 = [[0, 1, 2]]
    weights_layer2 = [[0.7, -0.4, 0.5]]
    biases_layer2 = [0.2]
    nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

    # Train
    history = nn.train(X, y, epochs=epochs, learning_rate=lr, optimizer='gd',
                      loss_function='binary', verbose=False)

    # Check predictions
    y_pred = nn.forward(X)
    print(f"  Predictions: {y_pred.flatten()}")

    # Check if EXACTLY 0.5
    exactly_half = np.allclose(y_pred, 0.5, atol=0.001)
    if exactly_half:
        print(f"  [STUCK] All predictions ~0.5!")

        # Debug info
        print(f"\n  Debug info:")
        print(f"    Hidden layer output: {nn.layer_outputs[1]}")
        print(f"    Dead ReLU neurons: {np.sum(np.all(nn.layer_outputs[1] == 0, axis=0))}/3")
        print(f"    Loss progression: {history['loss'][0]:.6f} -> {history['loss'][-1]:.6f}")

        return True
    else:
        print(f"  [OK] Diverse predictions")
        return False

# Scenario 1: Identical samples (no variance)
print("\n" + "=" * 70)
print("SCENARIO 1: Identical Samples (No Variance)")
print("=" * 70)

X_identical = np.array([
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5]
])
y_identical = np.array([[1], [1], [1], [1]])

stuck1 = test_scenario(X_identical, y_identical, 1.0, 100, "Identical samples")

# Scenario 2: Very low learning rate
print("\n" + "=" * 70)
print("SCENARIO 2: Very Low Learning Rate")
print("=" * 70)

X_normal = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])
y_normal = np.array([[1], [1], [0], [0]])

stuck2 = test_scenario(X_normal, y_normal, 0.001, 100, "Very low LR (0.001)")

# Scenario 3: Bad initialization + certain data
print("\n" + "=" * 70)
print("SCENARIO 3: Bad Weight Initialization")
print("=" * 70)

nn_bad = NeuralNetwork()
nn_bad.add_layer(2, 'linear')
nn_bad.add_layer(3, 'relu')
nn_bad.add_layer(1, 'sigmoid')

# All weights = 0 (bad init!)
connections_layer1 = [[0, 1], [0, 1], [0, 1]]
weights_layer1 = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
biases_layer1 = [0.0, 0.0, 0.0]
nn_bad.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

connections_layer2 = [[0, 1, 2]]
weights_layer2 = [[0.0, 0.0, 0.0]]
biases_layer2 = [0.0]
nn_bad.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print("  All weights = 0, all biases = 0")
y_pred = nn_bad.forward(X_normal)
print(f"  Predictions: {y_pred.flatten()}")
print(f"  [STUCK] Predictions = {y_pred[0][0]:.6f} (exactly 0.5!)")

# Scenario 4: All ReLU neurons dead
print("\n" + "=" * 70)
print("SCENARIO 4: All ReLU Neurons Dead")
print("=" * 70)

nn_dead = NeuralNetwork()
nn_dead.add_layer(2, 'linear')
nn_dead.add_layer(3, 'relu')
nn_dead.add_layer(1, 'sigmoid')

# Weights that cause all negative weighted sums (kill all ReLU)
connections_layer1 = [[0, 1], [0, 1], [0, 1]]
weights_layer1 = [[-5.0, -5.0], [-5.0, -5.0], [-5.0, -5.0]]
biases_layer1 = [-1.0, -1.0, -1.0]
nn_dead.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

connections_layer2 = [[0, 1, 2]]
weights_layer2 = [[1.0, 1.0, 1.0]]
biases_layer2 = [0.0]
nn_dead.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print("  Negative weights + negative biases -> all ReLU outputs = 0")
y_pred = nn_dead.forward(X_normal)
print(f"  Predictions: {y_pred.flatten()}")
print(f"  Hidden layer: {nn_dead.layer_outputs[1]}")
print(f"  [STUCK] All ReLU dead! Predictions = {y_pred[0][0]:.6f}")

# Summary
print("\n" + "=" * 70)
print("SCENARIOS THAT CAUSE 0.5 STUCK:")
print("=" * 70)
print("\n1. ❌ Identical samples (no variance to learn)")
print("2. ❌ Learning rate too low (no meaningful updates)")
print("3. ❌ All weights initialized to 0 (symmetry problem)")
print("4. ❌ All ReLU neurons dead (zero gradient)")
print("\n" + "-" * 70)
print("USER'S LIKELY ISSUE:")
print("-" * 70)
print("\nMost common causes:")
print("  1. Dataset values too large/small causing all ReLU = 0")
print("  2. Learning rate too small (try 0.5 - 1.0)")
print("  3. Not enough training epochs")
print("  4. Dataset not normalized")
print("\nSOLUTION:")
print("  ✓ Normalize data: X = (X - X.min()) / (X.max() - X.min())")
print("  ✓ Use learning rate: 0.5 - 1.0")
print("  ✓ Train for 200+ epochs")
print("  ✓ Check hidden layer has non-zero activations")
print("=" * 70)
