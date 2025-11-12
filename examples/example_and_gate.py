"""
Example: Training AND Gate

Simple example showing how to train a neural network to learn the AND gate logic.

Author: ANN from Scratch Team
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from backend.core import NeuralNetwork


def main():
    print("=" * 60)
    print(" AND Gate Training Example")
    print("=" * 60)

    # Build network (2-2-1)
    print("\n1. Building Network (2-2-1)...")
    network = NeuralNetwork()
    network.add_layer(2, 'linear')    # Input layer
    network.add_layer(2, 'sigmoid')   # Hidden layer
    network.add_layer(1, 'sigmoid')   # Output layer

    # Set connections
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

    print("Network built successfully!")
    print(network.get_architecture_summary())

    # Prepare AND gate data
    print("\n2. Preparing AND Gate Data...")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Test before training
    print("\n3. Predictions Before Training:")
    y_pred_before, y_prob_before = network.predict(X)
    for i in range(len(X)):
        print(f"   {X[i]} -> {y_prob_before[i][0]:.4f} (target: {y[i][0]})")

    # Train
    print("\n4. Training (500 epochs, lr=0.5, optimizer=GD)...")
    history = network.train(
        X, y,
        epochs=500,
        learning_rate=0.5,
        optimizer='gd',
        loss_function='mse',
        verbose=True
    )

    # Test after training
    print("\n5. Predictions After Training:")
    y_pred_after, y_prob_after = network.predict(X)
    for i in range(len(X)):
        correct = "✓" if y_pred_after[i][0] == y[i][0] else "✗"
        print(f"   {X[i]} -> {y_prob_after[i][0]:.4f} (target: {y[i][0]}) {correct}")

    # Calculate accuracy
    accuracy = np.mean((y_pred_after == y).astype(float))

    print("\n6. Results:")
    print(f"   Initial Loss: {history['loss'][0]:.6f}")
    print(f"   Final Loss: {history['loss'][-1]:.6f}")
    print(f"   Improvement: {(1 - history['loss'][-1] / history['loss'][0]) * 100:.2f}%")
    print(f"   Accuracy: {accuracy * 100:.2f}%")

    print("\n" + "=" * 60)
    print(" Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
