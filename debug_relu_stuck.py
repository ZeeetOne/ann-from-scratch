"""
Debug why predictions stuck at 0.5 dengan custom 2-3-1 ReLU network
"""
import numpy as np
from ann_core import NeuralNetwork

print("=" * 70)
print("DEBUG: Predictions Stuck at 0.5 - Custom 2-3-1 ReLU")
print("=" * 70)

def test_training(X, y, description):
    """Test training with given data"""
    print(f"\n{description}")
    print(f"  Input shape: {X.shape}, range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Target shape: {y.shape}")

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

    # Predictions BEFORE training
    y_pred_before = nn.forward(X)
    print(f"\n  BEFORE training:")
    print(f"    Predictions: {y_pred_before.flatten()}")
    print(f"    Unique values: {np.unique(y_pred_before)}")

    # Check hidden layer
    hidden_output = nn.layer_outputs[1]
    print(f"\n    Hidden layer (ReLU):")
    print(f"      Output: {hidden_output}")
    print(f"      Range: [{hidden_output.min():.3f}, {hidden_output.max():.3f}]")
    print(f"      Dead neurons: {np.sum(np.all(hidden_output == 0, axis=0))}/{hidden_output.shape[1]}")

    # Train
    history = nn.train(X, y, epochs=200, learning_rate=1.0, optimizer='gd',
                      loss_function='binary', verbose=False)

    # Predictions AFTER training
    y_pred_after = nn.forward(X)

    print(f"\n  AFTER training:")
    print(f"    Initial Loss: {history['loss'][0]:.6f}")
    print(f"    Final Loss:   {history['loss'][-1]:.6f}")
    print(f"    Improvement:  {(history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100:.2f}%")

    print(f"\n    Predictions: {y_pred_after.flatten()}")
    print(f"    Unique values: {np.unique(y_pred_after)}")

    # Check if stuck at 0.5
    stuck = np.all(np.abs(y_pred_after - 0.5) < 0.05)
    if stuck:
        print(f"    [WARNING] STUCK at ~0.5!")
    else:
        print(f"    [OK] DIVERSE predictions!")

    # Accuracy
    y_pred_classes = (y_pred_after > 0.5).astype(int)
    accuracy = np.mean(y_pred_classes == y)
    print(f"    Accuracy: {accuracy*100:.2f}%")

    # Check hidden layer after training
    hidden_output = nn.layer_outputs[1]
    print(f"\n    Hidden layer (ReLU) after training:")
    print(f"      Range: [{hidden_output.min():.3f}, {hidden_output.max():.3f}]")
    print(f"      Dead neurons: {np.sum(np.all(hidden_output == 0, axis=0))}/{hidden_output.shape[1]}")

    return not stuck

# Test 1: RAW data (NOT normalized) - likely user's case
print("\n" + "=" * 70)
print("TEST 1: RAW Data (NOT Normalized) - User's Case")
print("=" * 70)

X_raw = np.array([
    [2, 8],   # AND gate-like
    [5, 8],
    [3, 10],
    [6, 9]
])
y_raw = np.array([[1], [1], [1], [0]])

success1 = test_training(X_raw, y_raw, "Test 1: RAW data (values 2-10)")

# Test 2: NORMALIZED data [0, 1]
print("\n" + "=" * 70)
print("TEST 2: NORMALIZED Data [0, 1]")
print("=" * 70)

# Normalize X_raw
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X_norm = (X_raw - X_min) / (X_max - X_min)

success2 = test_training(X_norm, y_raw, "Test 2: NORMALIZED data")

# Test 3: Very small values (near zero)
print("\n" + "=" * 70)
print("TEST 3: Very Small Values")
print("=" * 70)

X_small = np.array([
    [0.01, 0.02],
    [0.03, 0.04],
    [0.02, 0.05],
    [0.04, 0.03]
])

success3 = test_training(X_small, y_raw, "Test 3: Small values (0.01-0.05)")

# Test 4: Negative values
print("\n" + "=" * 70)
print("TEST 4: Negative Values (ReLU killer)")
print("=" * 70)

X_negative = np.array([
    [-2, -1],
    [-3, -2],
    [-1, -3],
    [-4, -1]
])

success4 = test_training(X_negative, y_raw, "Test 4: Negative values")

# Summary
print("\n" + "=" * 70)
print("SUMMARY & SOLUTION")
print("=" * 70)

results = {
    "RAW data (2-10)": success1,
    "NORMALIZED [0,1]": success2,
    "Small values (0.01-0.05)": success3,
    "Negative values": success4
}

print("\nResults:")
for test, success in results.items():
    status = "[OK] Diverse" if success else "[FAIL] Stuck at 0.5"
    print(f"  {test:25s} {status}")

print("\n" + "-" * 70)
print("CONCLUSION:")
print("-" * 70)

if not success1 and success2:
    print("\n✓ Problem: Dataset NOT NORMALIZED!")
    print("  Raw data causes issues with ReLU and sigmoid.")
    print("\n✓ Solution: NORMALIZE input to [0, 1] range:")
    print("    X_normalized = (X - X.min()) / (X.max() - X.min())")

elif not success4:
    print("\n✓ Problem: Negative inputs kill ReLU neurons!")
    print("  ReLU outputs 0 for all negative inputs.")
    print("\n✓ Solution: Use normalized [0, 1] data or switch to Leaky ReLU")

elif not success1 and not success3:
    print("\n✓ Multiple issues detected!")
    print("  Always normalize data to [0, 1] for best results.")

print("\n" + "=" * 70)
