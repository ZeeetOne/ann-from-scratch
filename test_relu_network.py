"""
Test 2-3-1 network dengan ReLU activation (seperti user)
"""
import numpy as np
from ann_core import NeuralNetwork

print("=" * 70)
print("TEST: 2-3-1 Network dengan ReLU di Hidden Layer")
print("=" * 70)

# Create network dengan ReLU
print("\n1. Creating 2-3-1 network with ReLU...")
nn = NeuralNetwork()
nn.add_layer(2, 'linear')   # Input
nn.add_layer(3, 'relu')     # Hidden dengan ReLU!
nn.add_layer(1, 'sigmoid')  # Output

print("   Network:")
print(f"   Layer 0 (Input):  {nn.layers[0]} nodes, activation: {nn.activations[0]}")
print(f"   Layer 1 (Hidden): {nn.layers[1]} nodes, activation: {nn.activations[1]}")
print(f"   Layer 2 (Output): {nn.layers[2]} nodes, activation: {nn.activations[2]}")

# Set connections
print("\n2. Setting connections...")
connections_layer1 = [[0, 1], [0, 1], [0, 1]]
weights_layer1 = [[0.5, 0.3], [-0.4, 0.6], [0.2, -0.5]]
biases_layer1 = [0.1, -0.2, 0.3]
nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

connections_layer2 = [[0, 1, 2]]
weights_layer2 = [[0.7, -0.4, 0.5]]
biases_layer2 = [0.2]
nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print("   [OK] Connections set")

# Test with NORMALIZED data
print("\n3. Test with NORMALIZED data [0, 1]...")
X_norm = np.array([[0.5, 0.8], [0.3, 0.6], [0.7, 0.9]])
y = np.array([[1], [0], [1]])

try:
    print("   Forward pass...")
    y_pred = nn.forward(X_norm)
    print(f"   [OK] Output: {y_pred.flatten()}")

    print("\n   Predict...")
    y_classes, y_probs = nn.predict(X_norm)
    print(f"   [OK] Classes: {y_classes.flatten()}")

    print("\n   Training...")
    history = nn.train(X_norm, y, epochs=10, learning_rate=0.1, optimizer='gd', loss_function='binary', verbose=False)
    print(f"   [OK] Loss: {history['loss'][0]:.6f} -> {history['loss'][-1]:.6f}")

except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test with LARGE values (NOT normalized)
print("\n4. Test with LARGE values (NOT normalized)...")
X_large = np.array([[5, 8], [3, 6], [7, 9]])  # Much larger values

try:
    print("   Forward pass...")
    y_pred = nn.forward(X_large)
    print(f"   [OK] Output: {y_pred.flatten()}")
    print(f"   Output range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")

    # Check hidden layer activations
    print(f"\n   Hidden layer output (ReLU):")
    print(f"   {nn.layer_outputs[1]}")
    print(f"   Range: [{nn.layer_outputs[1].min():.3f}, {nn.layer_outputs[1].max():.3f}]")

    print("\n   Predict...")
    y_classes, y_probs = nn.predict(X_large)
    print(f"   [OK] Classes: {y_classes.flatten()}")

    print("\n   Training...")
    history = nn.train(X_large, y, epochs=10, learning_rate=0.1, optimizer='gd', loss_function='binary', verbose=False)
    print(f"   [OK] Loss: {history['loss'][0]:.6f} -> {history['loss'][-1]:.6f}")

except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test with VERY LARGE values
print("\n5. Test with VERY LARGE values...")
X_huge = np.array([[50, 80], [30, 60], [70, 90]])  # Very large

try:
    print("   Forward pass...")
    y_pred = nn.forward(X_huge)
    print(f"   [OK] Output: {y_pred.flatten()}")

    print("\n   Predict...")
    y_classes, y_probs = nn.predict(X_huge)
    print(f"   [OK] Classes: {y_classes.flatten()}")

except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nReLU doesn't cause the 'index out of bounds' error directly.")
print("The error is likely from:")
print("  1. Wrong connection indices (e.g., index 2 when only 0-1 exist)")
print("  2. Dataset parsing issues (wrong number of features/targets)")
print("  3. Frontend form validation issues")
print("\nUser should check:")
print("  - Connection indices match number of nodes in previous layer")
print("  - Dataset has correct number of features (2 for 2-3-1 network)")
print("=" * 70)
