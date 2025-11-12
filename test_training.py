"""
Test script for training functionality
Tests backpropagation, optimizers, and training loop
"""

import numpy as np
import matplotlib.pyplot as plt
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
    [np.random.randn(), np.random.randn()],
    [np.random.randn(), np.random.randn()]
]
biases_layer1 = [np.random.randn(), np.random.randn()]

nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

# Layer 2: connections from hidden to output
connections_layer2 = [[0, 1]]
weights_layer2 = [[np.random.randn(), np.random.randn()]]
biases_layer2 = [np.random.randn()]

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
    print(f"X={X[i]}, y_true={y[i][0]:.0f}, y_pred={y_pred_before[i][0]:.4f}")

print(f"\nLoss before training: {loss_before:.6f}")

# Train the network
print("\n" + "="*60)
print("Training with Gradient Descent")
print("="*60)

history_gd = nn.train(
    X, y,
    epochs=1000,
    learning_rate=0.5,
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
    print(f"X={X[i]}, y_true={y[i][0]:.0f}, y_pred={y_pred_after[i][0]:.4f}")

print(f"\nLoss after training: {loss_after:.6f}")
print(f"Loss improvement: {loss_before - loss_after:.6f}")

print("\nUpdated Network Architecture:")
print(nn.get_architecture_summary())

# Test with SGD
print("\n" + "="*60)
print("Testing with SGD Optimizer")
print("="*60)

# Create a new network
nn2 = NeuralNetwork()
nn2.add_layer(2, 'linear')
nn2.add_layer(2, 'sigmoid')
nn2.add_layer(1, 'sigmoid')

# Use same initial weights for fair comparison
nn2.set_connections(1, connections_layer1, weights_layer1, biases_layer1)
nn2.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

history_sgd = nn2.train(
    X, y,
    epochs=1000,
    learning_rate=0.5,
    optimizer='sgd',
    loss_function='mse',
    batch_size=2,
    verbose=True
)

y_pred_sgd = nn2.forward(X)
loss_sgd = nn2.calculate_loss(y, y_pred_sgd, 'mse')

print(f"\nFinal loss with SGD: {loss_sgd:.6f}")

# Plot loss curves
print("\n" + "="*60)
print("Plotting Loss Curves")
print("="*60)

plt.figure(figsize=(12, 5))

# Plot GD loss
plt.subplot(1, 2, 1)
plt.plot(history_gd['epoch'], history_gd['loss'], 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve - Gradient Descent')
plt.grid(True)

# Plot SGD loss
plt.subplot(1, 2, 2)
plt.plot(history_sgd['epoch'], history_sgd['loss'], 'r-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve - SGD')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_loss_curves.png', dpi=150, bbox_inches='tight')
print("\nLoss curves saved to 'training_loss_curves.png'")

# Compare both optimizers
plt.figure(figsize=(10, 6))
plt.plot(history_gd['epoch'], history_gd['loss'], 'b-', linewidth=2, label='Gradient Descent')
plt.plot(history_sgd['epoch'], history_sgd['loss'], 'r-', linewidth=2, label='SGD')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison: GD vs SGD')
plt.legend()
plt.grid(True)
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
print("Optimizer comparison saved to 'optimizer_comparison.png'")

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

plt.show()
