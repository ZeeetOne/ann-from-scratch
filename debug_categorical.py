"""
Debug script untuk menemukan masalah categorical cross-entropy
"""
import numpy as np
from ann_core import NeuralNetwork

# Create simple network
nn = NeuralNetwork()
nn.add_layer(3, 'linear')   # Input layer
nn.add_layer(4, 'sigmoid')  # Hidden layer
nn.add_layer(2, 'softmax')  # Output layer with softmax

# Set connections for layer 1 (input to hidden)
connections_layer1 = [
    [0, 1, 2],  # Node 0 connects to all 3 input nodes
    [0, 1, 2],  # Node 1 connects to all 3 input nodes
    [0, 1, 2],  # Node 2 connects to all 3 input nodes
    [0, 1, 2]   # Node 3 connects to all 3 input nodes
]
weights_layer1 = [
    [0.5, -0.3, 0.8],
    [-0.4, 0.6, -0.2],
    [0.7, -0.5, 0.3],
    [-0.6, 0.4, -0.7]
]
biases_layer1 = [0.1, -0.2, 0.3, -0.1]
nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

# Set connections for layer 2 (hidden to output)
connections_layer2 = [
    [0, 1, 2, 3],  # Output node 0 connects to all 4 hidden nodes
    [0, 1, 2, 3]   # Output node 1 connects to all 4 hidden nodes
]
weights_layer2 = [
    [0.2, -0.4, 0.6, -0.3],
    [-0.5, 0.3, -0.1, 0.7]
]
biases_layer2 = [0.1, -0.1]
nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print("=" * 70)
print("DEBUGGING CATEGORICAL CROSS-ENTROPY")
print("=" * 70)

# Simple test data (3 samples, NOT symmetric)
X = np.array([
    [25, 1010, 85],  # Sample 1
    [30, 1015, 45],  # Sample 2
    [22, 1005, 90]   # Sample 3
])
y_true = np.array([
    [1, 0],  # Sample 1: class 0
    [0, 1],  # Sample 2: class 1
    [1, 0]   # Sample 3: class 0  <- More class 0 samples
])

print("\nInput data:")
print("X shape:", X.shape)
print("y_true shape:", y_true.shape)
print("y_true:\n", y_true)

# Forward pass
print("\n" + "-" * 70)
print("FORWARD PASS")
print("-" * 70)
y_pred = nn.forward(X)
print("y_pred shape:", y_pred.shape)
print("y_pred:\n", y_pred)
print("\nSoftmax output sums (should be 1.0):", np.sum(y_pred, axis=1))

# Calculate initial loss
loss_before = nn.calculate_loss(y_true, y_pred, 'categorical')
print("\nInitial Loss (categorical):", loss_before)

# Check backward pass
print("\n" + "-" * 70)
print("BACKWARD PASS")
print("-" * 70)

# Manual calculation to debug
from ann_core import LossFunctions, ActivationFunctions
loss_derivative_fn = LossFunctions.get_loss_derivative('categorical')
dL_da = loss_derivative_fn(y_true, y_pred)
print("\nLoss derivative (dL/da):")
print(dL_da)

activation_derivative_fn = ActivationFunctions.get_activation_derivative('softmax')
da_dz = activation_derivative_fn(nn.layer_z_values[2])
print("\nActivation derivative (da/dz) for softmax:")
print(da_dz)

delta_manual = dL_da * da_dz
print("\nDelta (dL/da * da/dz):")
print(delta_manual)

print("\nPrevious layer output (hidden layer):")
print(nn.layer_outputs[1])

weight_grads, bias_grads = nn.backward(y_true, y_pred, 'categorical')

print("\nWeight gradients for layer 2 (output):")
for i, grads in enumerate(weight_grads[2]):
    print(f"  Node {i}: {grads}")

print("\nBias gradients for layer 2 (output):")
for i, grad in enumerate(bias_grads[2]):
    print(f"  Node {i}: {grad}")

# Check if gradients are zero
all_zero = True
for layer_idx in range(1, len(nn.layers)):
    for node_grads in weight_grads[layer_idx]:
        if np.any(np.abs(node_grads) > 1e-10):
            all_zero = False
            break

if all_zero:
    print("\n!!! WARNING: ALL GRADIENTS ARE ZERO !!!")
else:
    print("\n[OK] Gradients are non-zero")

# Manual update
print("\n" + "-" * 70)
print("MANUAL WEIGHT UPDATE")
print("-" * 70)
learning_rate = 0.5

print("\nOld weights layer 2 node 0:", nn.weights[2][0])
print("Gradients:", weight_grads[2][0])

# Update weights manually
for layer_idx in range(1, len(nn.layers)):
    for node_idx in range(len(nn.weights[layer_idx])):
        for w_idx in range(len(nn.weights[layer_idx][node_idx])):
            old_w = nn.weights[layer_idx][node_idx][w_idx]
            grad = weight_grads[layer_idx][node_idx][w_idx]
            new_w = old_w - learning_rate * grad
            nn.weights[layer_idx][node_idx][w_idx] = new_w

            if layer_idx == 2 and node_idx == 0 and w_idx == 0:
                print(f"\nExample update (layer 2, node 0, weight 0):")
                print(f"  Old weight: {old_w:.6f}")
                print(f"  Gradient:   {grad:.6f}")
                print(f"  New weight: {new_w:.6f}")
                print(f"  Change:     {new_w - old_w:.6f}")

        # Update bias
        old_b = nn.biases[layer_idx][node_idx]
        grad_b = bias_grads[layer_idx][node_idx]
        new_b = old_b - learning_rate * grad_b
        nn.biases[layer_idx][node_idx] = new_b

print("\nNew weights layer 2 node 0:", nn.weights[2][0])

# Forward pass again
y_pred_after = nn.forward(X)
loss_after = nn.calculate_loss(y_true, y_pred_after, 'categorical')

print("\n" + "-" * 70)
print("RESULTS")
print("-" * 70)
print(f"Loss BEFORE update: {loss_before:.6f}")
print(f"Loss AFTER update:  {loss_after:.6f}")
print(f"Loss change:        {loss_before - loss_after:.6f}")

if abs(loss_before - loss_after) < 1e-10:
    print("\n!!! BUG DETECTED: Loss did not change !!!")
else:
    print("\n[OK] Loss changed successfully!")

print("\n" + "=" * 70)
