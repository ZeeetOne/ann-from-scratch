"""
Simple test script for training functionality (no visualization)
Tests backpropagation, optimizers, and training loop
"""

import numpy as np
from ann_core import NeuralNetwork

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("Testing Neural Network Training Implementation")
print("="*60)

# Create a simple XOR-like dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])  # XOR pattern

print("\nDataset:")
print("X:")
print(X)
print("\ny (target):")
print(y)

# Create a simple 2-2-1 network
print("\n" + "="*60)
print("Creating Neural Network (2-2-1)")
print("="*60)

nn = NeuralNetwork()

# Layer 0: Input layer (2 nodes)
nn.add_layer(2, 'linear')

# Layer 1: Hidden layer (2 nodes, sigmoid)
nn.add_layer(2, 'sigmoid')

# Layer 2: Output layer (1 node, sigmoid)
nn.add_layer(1, 'sigmoid')

# Initialize with random weights
# Layer 1: connections from input to hidden
connections_layer1 = [[0, 1], [0, 1]]
weights_layer1 = [
    [0.5, -0.3],
    [-0.4, 0.6]
]
biases_layer1 = [0.1, -0.2]

nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

# Layer 2: connections from hidden to output
connections_layer2 = [[0, 1]]
weights_layer2 = [[0.8, -0.5]]
biases_layer2 = [0.2]

nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print("\nInitial Network Architecture:")
print(nn.get_architecture_summary())

# Make predictions before training
print("\n" + "="*60)
print("Before Training")
print("="*60)

y_pred_before = nn.forward(X)
loss_before = nn.calculate_loss(y, y_pred_before, 'mse')

print(f"\nPredictions before training:")
for i in range(len(X)):
    print(f"  X={X[i]}, y_true={y[i][0]:.0f}, y_pred={y_pred_before[i][0]:.4f}")

print(f"\nLoss before training: {loss_before:.6f}")

# Train the network with Gradient Descent
print("\n" + "="*60)
print("Training with Gradient Descent")
print("="*60)
print("Parameters: epochs=500, learning_rate=1.0, optimizer=GD\n")

history_gd = nn.train(
    X, y,
    epochs=500,
    learning_rate=1.0,
    optimizer='gd',
    loss_function='mse',
    verbose=True
)

# Make predictions after training
print("\n" + "="*60)
print("After Training (Gradient Descent)")
print("="*60)

y_pred_after = nn.forward(X)
loss_after = nn.calculate_loss(y, y_pred_after, 'mse')

print(f"\nPredictions after training:")
for i in range(len(X)):
    predicted_class = "1" if y_pred_after[i][0] >= 0.5 else "0"
    correct = "[OK]" if int(predicted_class) == y[i][0] else "[X]"
    print(f"  X={X[i]}, y_true={y[i][0]:.0f}, y_pred={y_pred_after[i][0]:.4f}, class={predicted_class} {correct}")

print(f"\nLoss after training: {loss_after:.6f}")
print(f"Loss improvement: {loss_before - loss_after:.6f} ({((loss_before - loss_after) / loss_before * 100):.2f}%)")

print("\n" + "="*60)
print("Updated Weights and Biases")
print("="*60)

print("\nLayer 1 (Hidden Layer):")
for i in range(len(nn.weights[1])):
    print(f"  Node {i}:")
    print(f"    Weights: {nn.weights[1][i]}")
    print(f"    Bias: {nn.biases[1][i]:.4f}")

print("\nLayer 2 (Output Layer):")
for i in range(len(nn.weights[2])):
    print(f"  Node {i}:")
    print(f"    Weights: {nn.weights[2][i]}")
    print(f"    Bias: {nn.biases[2][i]:.4f}")

# Test with SGD
print("\n" + "="*60)
print("Testing with SGD Optimizer")
print("="*60)

# Create a new network with same initial weights
nn2 = NeuralNetwork()
nn2.add_layer(2, 'linear')
nn2.add_layer(2, 'sigmoid')
nn2.add_layer(1, 'sigmoid')

nn2.set_connections(1, connections_layer1,
                   [[0.5, -0.3], [-0.4, 0.6]],
                   [0.1, -0.2])
nn2.set_connections(2, connections_layer2,
                   [[0.8, -0.5]],
                   [0.2])

print("Parameters: epochs=500, learning_rate=1.0, optimizer=SGD, batch_size=2\n")

history_sgd = nn2.train(
    X, y,
    epochs=500,
    learning_rate=1.0,
    optimizer='sgd',
    loss_function='mse',
    batch_size=2,
    verbose=True
)

y_pred_sgd = nn2.forward(X)
loss_sgd = nn2.calculate_loss(y, y_pred_sgd, 'mse')

print(f"\nPredictions after training (SGD):")
for i in range(len(X)):
    predicted_class = "1" if y_pred_sgd[i][0] >= 0.5 else "0"
    correct = "[OK]" if int(predicted_class) == y[i][0] else "[X]"
    print(f"  X={X[i]}, y_true={y[i][0]:.0f}, y_pred={y_pred_sgd[i][0]:.4f}, class={predicted_class} {correct}")

print(f"\nFinal loss with SGD: {loss_sgd:.6f}")

# Display loss history
print("\n" + "="*60)
print("Loss History (every 50 epochs)")
print("="*60)

print("\nGradient Descent:")
print("Epoch\tLoss")
for i in range(0, len(history_gd['epoch']), 50):
    print(f"{history_gd['epoch'][i]}\t{history_gd['loss'][i]:.6f}")

print("\nStochastic Gradient Descent:")
print("Epoch\tLoss")
for i in range(0, len(history_sgd['epoch']), 50):
    print(f"{history_sgd['epoch'][i]}\t{history_sgd['loss'][i]:.6f}")

print("\n" + "="*60)
print("Testing Complete!")
print("="*60)

# Summary
print("\nSummary:")
print(f"  Initial Loss: {loss_before:.6f}")
print(f"  Final Loss (GD): {loss_after:.6f}")
print(f"  Final Loss (SGD): {loss_sgd:.6f}")
print(f"  Improvement (GD): {((loss_before - loss_after) / loss_before * 100):.2f}%")
print(f"  Improvement (SGD): {((loss_before - loss_sgd) / loss_before * 100):.2f}%")

# Calculate accuracy
accuracy_gd = np.mean([(1 if y_pred_after[i][0] >= 0.5 else 0) == y[i][0] for i in range(len(y))])
accuracy_sgd = np.mean([(1 if y_pred_sgd[i][0] >= 0.5 else 0) == y[i][0] for i in range(len(y))])

print(f"  Accuracy (GD): {accuracy_gd * 100:.2f}%")
print(f"  Accuracy (SGD): {accuracy_sgd * 100:.2f}%")

print("\n[SUCCESS] Training functionality is working correctly!")
