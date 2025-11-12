"""
Test script for ANN core library
"""

import numpy as np
from ann_core import NeuralNetwork

# Create test dataset
X = np.array([
    [2, 8],
    [5, 8],
    [3, 10],
    [1, 15],
    [3, 12],
    [4, 6],
    [4, 8],
    [2, 10],
    [5, 12],
    [2, 6]
])

y_true = np.array([[1], [1], [1], [0], [0], [1], [1], [0], [1], [1]])

print("=" * 60)
print("Testing ANN from Scratch")
print("=" * 60)

# Create a simple 2-3-1 network
nn = NeuralNetwork()

# Layer 0: Input layer (2 nodes)
nn.add_layer(2, 'linear')

# Layer 1: Hidden layer (3 nodes)
nn.add_layer(3, 'sigmoid')

# Layer 2: Output layer (1 node)
nn.add_layer(1, 'sigmoid')

print("\nNetwork created with architecture: 2-3-1")
print(nn.get_architecture_summary())

# Set connections for layer 1 (input to hidden)
connections_layer1 = [
    [0, 1],  # Node 0 connects to input nodes 0 and 1
    [0, 1],  # Node 1 connects to input nodes 0 and 1
    [0, 1]   # Node 2 connects to input nodes 0 and 1
]
weights_layer1 = [
    [0.5, 0.3],   # Weights for node 0
    [-0.4, 0.6],  # Weights for node 1
    [0.2, -0.5]   # Weights for node 2
]
biases_layer1 = [0.1, -0.2, 0.3]

nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

# Set connections for layer 2 (hidden to output)
connections_layer2 = [
    [0, 1, 2]  # Output node connects to all 3 hidden nodes
]
weights_layer2 = [
    [0.8, -0.3, 0.6]  # Weights for output node
]
biases_layer2 = [0.0]

nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print("\n" + "=" * 60)
print("Making predictions...")
print("=" * 60)

# Make predictions
y_pred_classes, y_pred_probs = nn.predict(X, threshold=0.5)

# Display results
print("\nResults:")
print("-" * 60)
print(f"{'xi':<5} {'xj':<5} {'Actual':<10} {'Prediction':<12} {'y-hat':<10}")
print("-" * 60)

for i in range(len(X)):
    actual = "Yes (1)" if y_true[i][0] == 1 else "No (0)"
    prediction = "Yes" if y_pred_classes[i][0] == 1 else "No"
    prob = y_pred_probs[i][0]

    print(f"{X[i][0]:<5} {X[i][1]:<5} {actual:<10} {prediction:<12} {prob:.3f}")

# Calculate metrics
accuracy = np.mean(y_pred_classes == y_true)
loss_mse = nn.calculate_loss(y_true, y_pred_probs, 'mse')
loss_binary = nn.calculate_loss(y_true, y_pred_probs, 'binary')

print("-" * 60)
print(f"\nMetrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"MSE Loss: {loss_mse:.4f}")
print(f"Binary Cross-Entropy Loss: {loss_binary:.4f}")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
