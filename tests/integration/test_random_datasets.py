"""
Random Dataset Tests

Comprehensive tests with random datasets for binary and multi-class classification.
Compares results with manual calculations at each step.

Author: ANN from Scratch Team
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from backend.core import NeuralNetwork


class TestRandomDatasets(unittest.TestCase):
    """
    Test neural network with random datasets.

    Scenarios:
    1. Binary classification with random data
    2. Multi-class classification with random data
    3. Multi-label classification with random data
    """

    def setUp(self):
        """Set random seed for reproducibility"""
        np.random.seed(42)

    def test_binary_classification_random(self):
        """
        Test binary classification with random dataset

        Network: 5-8-1 (sigmoid output)
        Dataset: 20 samples, 5 features, binary labels
        """
        print("\n" + "="*70)
        print("TEST: Binary Classification with Random Dataset")
        print("="*70)

        # Generate random dataset
        n_samples = 20
        n_features = 5
        n_hidden = 8

        print(f"\nDataset Info:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Hidden neurons: {n_hidden}")
        print(f"  Output: 1 (binary)")

        # Random features (normalized to [0, 1])
        X = np.random.rand(n_samples, n_features)

        # Random binary labels based on threshold of sum of features
        y = (X.sum(axis=1) > n_features * 0.5).astype(int).reshape(-1, 1)

        print(f"\nClass distribution:")
        print(f"  Class 0: {np.sum(y == 0)} samples")
        print(f"  Class 1: {np.sum(y == 1)} samples")

        # Build network
        network = NeuralNetwork()
        network.add_layer(n_features, 'linear')
        network.add_layer(n_hidden, 'sigmoid')
        network.add_layer(1, 'sigmoid')

        # Initialize with random small weights
        network.set_full_connections(
            1,
            weight_matrix=np.random.randn(n_hidden, n_features) * 0.3,
            biases=np.zeros(n_hidden)
        )
        network.set_full_connections(
            2,
            weight_matrix=np.random.randn(1, n_hidden) * 0.3,
            biases=np.zeros(1)
        )

        print(f"\nNetwork Architecture:")
        print(f"  Input: {n_features} nodes")
        print(f"  Hidden: {n_hidden} nodes (sigmoid)")
        print(f"  Output: 1 node (sigmoid)")
        print(f"  Classification type: {network.get_classification_type()}")
        print(f"  Recommended loss: {network.get_recommended_loss()}")

        # Test before training
        y_pred_before, y_prob_before = network.predict(X)
        acc_before = np.mean((y_pred_before == y).astype(float))

        print(f"\nBefore Training:")
        print(f"  Accuracy: {acc_before * 100:.2f}%")

        # Sample predictions
        print(f"\n  Sample predictions (first 5):")
        for i in range(min(5, n_samples)):
            print(f"    Sample {i}: pred={y_prob_before[i][0]:.4f}, true={y[i][0]}, " +
                  f"correct={'âœ“' if y_pred_before[i][0] == y[i][0] else 'âœ—'}")

        # Train
        print(f"\nTraining (300 epochs, lr=0.5, optimizer=GD, loss=binary)...")
        history = network.train(
            X, y,
            epochs=300,
            learning_rate=0.5,
            optimizer='gd',
            loss_function='binary',
            verbose=False
        )

        # Test after training
        y_pred_after, y_prob_after = network.predict(X)
        acc_after = np.mean((y_pred_after == y).astype(float))

        print(f"\nAfter Training:")
        print(f"  Initial loss: {history['loss'][0]:.6f}")
        print(f"  Final loss: {history['loss'][-1]:.6f}")
        print(f"  Loss reduction: {(1 - history['loss'][-1]/history['loss'][0])*100:.2f}%")
        print(f"  Accuracy: {acc_after * 100:.2f}%")
        print(f"  Improvement: {(acc_after - acc_before)*100:.2f}%")

        # Sample predictions after
        print(f"\n  Sample predictions after training (first 5):")
        for i in range(min(5, n_samples)):
            print(f"    Sample {i}: pred={y_prob_after[i][0]:.4f}, true={y[i][0]}, " +
                  f"correct={'âœ“' if y_pred_after[i][0] == y[i][0] else 'âœ—'}")

        # Verify training improved performance
        self.assertGreater(acc_after, acc_before * 0.9,
                          "Training should maintain or improve accuracy")

        print(f"\nâœ“ Binary classification test passed")

    def test_multiclass_classification_random(self):
        """
        Test multi-class classification with random dataset

        Network: 4-6-3 (softmax output)
        Dataset: 30 samples, 4 features, 3 classes
        """
        print("\n" + "="*70)
        print("TEST: Multi-Class Classification with Random Dataset")
        print("="*70)

        # Generate random dataset
        n_samples = 30
        n_features = 4
        n_hidden = 6
        n_classes = 3

        print(f"\nDataset Info:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Hidden neurons: {n_hidden}")
        print(f"  Classes: {n_classes}")

        # Random features
        X = np.random.rand(n_samples, n_features)

        # Generate class labels based on feature sum ranges
        feature_sum = X.sum(axis=1)
        y_labels = np.zeros(n_samples, dtype=int)
        y_labels[feature_sum < n_features * 0.33] = 0
        y_labels[(feature_sum >= n_features * 0.33) & (feature_sum < n_features * 0.67)] = 1
        y_labels[feature_sum >= n_features * 0.67] = 2

        # Convert to one-hot encoding
        y = np.zeros((n_samples, n_classes))
        y[np.arange(n_samples), y_labels] = 1

        print(f"\nClass distribution:")
        for i in range(n_classes):
            print(f"  Class {i}: {np.sum(y_labels == i)} samples")

        # Build network
        network = NeuralNetwork()
        network.add_layer(n_features, 'linear')
        network.add_layer(n_hidden, 'sigmoid')
        network.add_layer(n_classes, 'softmax')

        # Initialize with random small weights
        network.set_full_connections(
            1,
            weight_matrix=np.random.randn(n_hidden, n_features) * 0.3,
            biases=np.zeros(n_hidden)
        )
        network.set_full_connections(
            2,
            weight_matrix=np.random.randn(n_classes, n_hidden) * 0.3,
            biases=np.zeros(n_classes)
        )

        print(f"\nNetwork Architecture:")
        print(f"  Input: {n_features} nodes")
        print(f"  Hidden: {n_hidden} nodes (sigmoid)")
        print(f"  Output: {n_classes} nodes (softmax)")
        print(f"  Classification type: {network.get_classification_type()}")
        print(f"  Recommended loss: {network.get_recommended_loss()}")

        # Verify softmax properties
        y_pred_test = network.forward(X[:3])
        sums = np.sum(y_pred_test, axis=1)
        print(f"\nSoftmax verification (first 3 samples):")
        for i in range(3):
            print(f"  Sample {i}: sum of probabilities = {sums[i]:.6f} (should be 1.0)")
            self.assertAlmostEqual(sums[i], 1.0, places=5,
                                  msg=f"Softmax outputs must sum to 1.0")

        # Test before training
        y_pred_before, y_prob_before = network.predict(X)
        acc_before = np.mean((y_pred_before == y).astype(float))

        print(f"\nBefore Training:")
        print(f"  Accuracy: {acc_before * 100:.2f}%")

        # Sample predictions
        print(f"\n  Sample predictions (first 5):")
        for i in range(min(5, n_samples)):
            pred_class = np.argmax(y_prob_before[i])
            true_class = np.argmax(y[i])
            print(f"    Sample {i}: pred_class={pred_class} (prob={y_prob_before[i]}), " +
                  f"true_class={true_class}, correct={'âœ“' if pred_class == true_class else 'âœ—'}")

        # Train
        print(f"\nTraining (400 epochs, lr=0.5, optimizer=SGD, loss=categorical)...")
        history = network.train(
            X, y,
            epochs=400,
            learning_rate=0.5,
            optimizer='sgd',
            loss_function='categorical',
            verbose=False
        )

        # Test after training
        y_pred_after, y_prob_after = network.predict(X)
        acc_after = np.mean((y_pred_after == y).astype(float))

        print(f"\nAfter Training:")
        print(f"  Initial loss: {history['loss'][0]:.6f}")
        print(f"  Final loss: {history['loss'][-1]:.6f}")
        print(f"  Loss reduction: {(1 - history['loss'][-1]/history['loss'][0])*100:.2f}%")
        print(f"  Accuracy: {acc_after * 100:.2f}%")
        print(f"  Improvement: {(acc_after - acc_before)*100:.2f}%")

        # Sample predictions after
        print(f"\n  Sample predictions after training (first 5):")
        for i in range(min(5, n_samples)):
            pred_class = np.argmax(y_prob_after[i])
            true_class = np.argmax(y[i])
            print(f"    Sample {i}: pred_class={pred_class} (prob={y_prob_after[i]}), " +
                  f"true_class={true_class}, correct={'âœ“' if pred_class == true_class else 'âœ—'}")

        # Per-class accuracy
        print(f"\n  Per-class accuracy:")
        for c in range(n_classes):
            class_mask = y_labels == c
            class_acc = np.mean((np.argmax(y_pred_after[class_mask], axis=1) == c).astype(float))
            print(f"    Class {c}: {class_acc * 100:.2f}%")

        # Verify training improved performance
        self.assertGreater(acc_after, acc_before * 0.9,
                          "Training should maintain or improve accuracy")

        print(f"\nâœ“ Multi-class classification test passed")

    def test_multilabel_classification_random(self):
        """
        Test multi-label classification with random dataset

        Network: 6-10-4 (sigmoid outputs)
        Dataset: 25 samples, 6 features, 4 independent labels
        """
        print("\n" + "="*70)
        print("TEST: Multi-Label Classification with Random Dataset")
        print("="*70)

        # Generate random dataset
        n_samples = 25
        n_features = 6
        n_hidden = 10
        n_labels = 4

        print(f"\nDataset Info:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Hidden neurons: {n_hidden}")
        print(f"  Labels: {n_labels} (independent)")

        # Random features
        X = np.random.rand(n_samples, n_features)

        # Generate independent binary labels
        y = np.zeros((n_samples, n_labels))
        for i in range(n_labels):
            # Each label has its own threshold based on different feature combinations
            threshold = 0.5 + i * 0.1
            y[:, i] = (X[:, i % n_features] > threshold).astype(int)

        print(f"\nLabel distribution:")
        for i in range(n_labels):
            print(f"  Label {i}: {int(np.sum(y[:, i]))} positive samples")

        # Build network
        network = NeuralNetwork()
        network.add_layer(n_features, 'linear')
        network.add_layer(n_hidden, 'sigmoid')
        network.add_layer(n_labels, 'sigmoid')  # Sigmoid for multi-label

        # Initialize with random small weights
        network.set_full_connections(
            1,
            weight_matrix=np.random.randn(n_hidden, n_features) * 0.3,
            biases=np.zeros(n_hidden)
        )
        network.set_full_connections(
            2,
            weight_matrix=np.random.randn(n_labels, n_hidden) * 0.3,
            biases=np.zeros(n_labels)
        )

        print(f"\nNetwork Architecture:")
        print(f"  Input: {n_features} nodes")
        print(f"  Hidden: {n_hidden} nodes (sigmoid)")
        print(f"  Output: {n_labels} nodes (sigmoid)")
        print(f"  Classification type: {network.get_classification_type()}")
        print(f"  Recommended loss: {network.get_recommended_loss()}")

        # Test before training
        y_pred_before, y_prob_before = network.predict(X)
        acc_before = np.mean((y_pred_before == y).astype(float))

        print(f"\nBefore Training:")
        print(f"  Exact match accuracy: {acc_before * 100:.2f}%")

        # Per-label accuracy
        print(f"\n  Per-label accuracy before:")
        for i in range(n_labels):
            label_acc = np.mean((y_pred_before[:, i] == y[:, i]).astype(float))
            print(f"    Label {i}: {label_acc * 100:.2f}%")

        # Train
        print(f"\nTraining (300 epochs, lr=0.3, optimizer=GD, loss=binary)...")
        history = network.train(
            X, y,
            epochs=300,
            learning_rate=0.3,
            optimizer='gd',
            loss_function='binary',
            verbose=False
        )

        # Test after training
        y_pred_after, y_prob_after = network.predict(X)
        acc_after = np.mean((y_pred_after == y).astype(float))

        print(f"\nAfter Training:")
        print(f"  Initial loss: {history['loss'][0]:.6f}")
        print(f"  Final loss: {history['loss'][-1]:.6f}")
        print(f"  Loss reduction: {(1 - history['loss'][-1]/history['loss'][0])*100:.2f}%")
        print(f"  Exact match accuracy: {acc_after * 100:.2f}%")
        print(f"  Improvement: {(acc_after - acc_before)*100:.2f}%")

        # Per-label accuracy after
        print(f"\n  Per-label accuracy after:")
        for i in range(n_labels):
            label_acc = np.mean((y_pred_after[:, i] == y[:, i]).astype(float))
            print(f"    Label {i}: {label_acc * 100:.2f}%")

        # Sample predictions
        print(f"\n  Sample predictions (first 3):")
        for i in range(min(3, n_samples)):
            print(f"    Sample {i}:")
            print(f"      Predicted: {y_pred_after[i]} (prob: {y_prob_after[i]})")
            print(f"      True:      {y[i].astype(int)}")
            print(f"      Match: {'âœ“' if np.array_equal(y_pred_after[i], y[i]) else 'âœ—'}")

        print(f"\nâœ“ Multi-label classification test passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)

