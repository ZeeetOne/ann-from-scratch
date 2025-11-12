"""
Debug error di predict() atau train()
"""
import numpy as np
from ann_core import NeuralNetwork

print("=" * 70)
print("DEBUG: Testing predict() and train() with 2-3-1")
print("=" * 70)

# Create 2-3-1 network
nn = NeuralNetwork()
nn.add_layer(2, 'linear')
nn.add_layer(3, 'sigmoid')
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

# Test data
X = np.array([
    [0.5, 0.8],
    [0.3, 0.6],
    [0.7, 0.9]
])
y = np.array([[1], [0], [1]])

print("\nTest data:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Test 1: predict()
print("\n1. Testing predict()...")
try:
    y_pred_classes, y_pred_probs = nn.predict(X)
    print(f"   [OK] Predict completed")
    print(f"   Classes shape: {y_pred_classes.shape}")
    print(f"   Probs shape: {y_pred_probs.shape}")
    print(f"   Classes: {y_pred_classes.flatten()}")
    print(f"   Probs: {y_pred_probs.flatten()}")
except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: backward()
print("\n2. Testing backward()...")
try:
    y_pred = nn.forward(X)
    weight_grads, bias_grads = nn.backward(y, y_pred, 'binary')
    print(f"   [OK] Backward completed")
    print(f"   Weight grads layers: {len(weight_grads)}")
    print(f"   Bias grads layers: {len(bias_grads)}")
except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: train()
print("\n3. Testing train()...")
try:
    history = nn.train(X, y, epochs=10, learning_rate=0.5, optimizer='gd', loss_function='binary', verbose=False)
    print(f"   [OK] Training completed")
    print(f"   Initial loss: {history['loss'][0]:.6f}")
    print(f"   Final loss: {history['loss'][-1]:.6f}")
except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Different input sizes
print("\n4. Testing with different input configurations...")

test_cases = [
    ("Single sample", np.array([[0.5, 0.8]]), np.array([[1]])),
    ("Multiple samples", np.array([[0.5, 0.8], [0.3, 0.6]]), np.array([[1], [0]])),
    ("1D input (will be reshaped)", np.array([0.5, 0.8]), np.array([[1]])),
]

for name, X_test, y_test in test_cases:
    print(f"\n   Test: {name}")
    print(f"   X shape: {X_test.shape}")
    try:
        y_pred = nn.forward(X_test)
        print(f"   [OK] Forward: output shape {y_pred.shape}")
    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {e}")

print("\n" + "=" * 70)
