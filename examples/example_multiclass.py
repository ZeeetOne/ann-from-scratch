"""
Example: Multi-Class Classification with Softmax

Example showing how to train a multi-class classifier using softmax activation.

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
    print(" Multi-Class Classification Example")
    print("=" * 60)

    # Build network (3-4-2 with softmax)
    print("\n1. Building Network (3-4-2 with Softmax)...")
    network = NeuralNetwork()
    network.add_layer(3, 'linear')     # Input: 3 features
    network.add_layer(4, 'sigmoid')    # Hidden: 4 neurons
    network.add_layer(2, 'softmax')    # Output: 2 classes

    # Set connections
    network.set_connections(
        1,
        [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[0.5, 0.3, -0.2], [-0.4, 0.6, 0.1], [0.2, -0.5, 0.4], [0.7, 0.2, -0.3]],
        [0.1, -0.2, 0.3, 0.0]
    )
    network.set_connections(
        2,
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        [[0.8, -0.3, 0.6, 0.4], [-0.5, 0.7, -0.2, 0.3]],
        [0.1, -0.1]
    )

    print("Network built successfully!")
    print(f"Classification Type: {network.get_classification_type()}")
    print(f"Recommended Loss: {network.get_recommended_loss()}")

    # Prepare data (2 classes)
    print("\n2. Preparing Dataset (Weather Classification)...")
    X = np.array([
        [0.5, 0.5, 0.8],  # Rain
        [0.2, 0.3, 0.9],  # Rain
        [0.3, 0.4, 0.85], # Rain
        [0.8, 0.8, 0.1],  # Sunny
        [0.9, 0.9, 0.0],  # Sunny
        [0.7, 0.75, 0.15] # Sunny
    ])
    y = np.array([
        [1, 0],  # Class 0 (Rain)
        [1, 0],
        [1, 0],
        [0, 1],  # Class 1 (Sunny)
        [0, 1],
        [0, 1]
    ])

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Classes: Rain (1,0), Sunny (0,1)")

    # Test before training
    print("\n3. Predictions Before Training:")
    y_pred_before, y_prob_before = network.predict(X)
    for i in range(len(X)):
        class_name = "Rain" if np.argmax(y[i]) == 0 else "Sunny"
        pred_name = "Rain" if np.argmax(y_prob_before[i]) == 0 else "Sunny"
        print(f"   Sample {i}: {pred_name} (prob: {y_prob_before[i]}) | True: {class_name}")

    # Train
    print("\n4. Training (300 epochs, lr=0.5, optimizer=SGD)...")
    history = network.train(
        X, y,
        epochs=300,
        learning_rate=0.5,
        optimizer='sgd',
        loss_function='categorical',
        verbose=True
    )

    # Test after training
    print("\n5. Predictions After Training:")
    y_pred_after, y_prob_after = network.predict(X)

    correct = 0
    for i in range(len(X)):
        class_name = "Rain" if np.argmax(y[i]) == 0 else "Sunny"
        pred_name = "Rain" if np.argmax(y_prob_after[i]) == 0 else "Sunny"
        is_correct = pred_name == class_name
        correct += int(is_correct)
        symbol = "✓" if is_correct else "✗"
        print(f"   Sample {i}: {pred_name} (prob: {y_prob_after[i]}) | True: {class_name} {symbol}")

    accuracy = correct / len(X)

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
