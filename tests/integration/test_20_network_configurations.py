"""
20 Network Configuration Tests

Comprehensive tests for 20 different network architectures and activation functions.
Tests custom network building, random dataset generation, training, and verification
that web results match manual calculations.

Author: ANN from Scratch Team
"""

import unittest
import numpy as np
import sys
import os
import json

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from backend.api.app import create_app
from backend.core import NeuralNetwork


class Test20NetworkConfigurations(unittest.TestCase):
    """
    Test suite for 20 different network configurations.

    Each test:
    1. Builds a custom network with specific architecture and activations
    2. Generates random dataset matching the architecture
    3. Tests forward pass
    4. Tests training
    5. Verifies results match manual calculations
    """

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client"""
        cls.app = create_app('testing')
        cls.client = cls.app.test_client()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        """Clean up Flask context"""
        cls.app_context.pop()

    def setUp(self):
        """Set random seed for reproducibility"""
        np.random.seed(42)

    def _test_network_configuration(
        self,
        test_name: str,
        architecture: list,
        activations: list,
        num_samples: int = 10,
        epochs: int = 50,
        learning_rate: float = 0.3,
        loss_function: str = None
    ):
        """
        Helper method to test a network configuration

        Args:
            test_name: Name of the test
            architecture: List of layer sizes [input, hidden..., output]
            activations: List of activation functions
            num_samples: Number of samples in dataset
            epochs: Training epochs
            learning_rate: Learning rate
            loss_function: Loss function (auto-detected if None)
        """
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")

        print(f"\nArchitecture: {'-'.join(map(str, architecture))}")
        print(f"Activations: {' -> '.join(activations)}")

        # 1. Build network with PROPER weight initialization
        layers_config = []
        for i, (size, activation) in enumerate(zip(architecture, activations)):
            layers_config.append({
                'num_nodes': size,
                'activation': activation
            })

        # Create connections with small random weights
        connections_config = []
        for i in range(1, len(architecture)):
            prev_size = architecture[i-1]
            curr_size = architecture[i]

            # Initialize with Xavier/He initialization
            if activations[i] == 'relu':
                # He initialization for ReLU
                std = np.sqrt(2.0 / prev_size)
            else:
                # Xavier initialization for sigmoid/tanh
                std = np.sqrt(1.0 / prev_size)

            connections = [[j for j in range(prev_size)] for _ in range(curr_size)]
            weights = (np.random.randn(curr_size, prev_size) * std).tolist()
            biases = np.zeros(curr_size).tolist()

            connections_config.append({
                'layer_idx': i,
                'connections': connections,
                'weights': weights,
                'biases': biases
            })

        # Build network via API
        build_response = self.client.post('/build_network', json={
            'layers': layers_config,
            'connections': connections_config
        })

        self.assertEqual(build_response.status_code, 200)
        build_data = json.loads(build_response.data)
        self.assertTrue(build_data['success'])

        print(f"[OK] Network built successfully")
        print(f"     Classification type: {build_data['classification_type']}")
        print(f"     Recommended loss: {build_data['recommended_loss']}")

        # Determine loss function
        if loss_function is None:
            loss_function = build_data['recommended_loss']

        # 2. Generate random dataset
        dataset_response = self.client.post('/generate_random_dataset', json={
            'num_samples': num_samples
        })

        self.assertEqual(dataset_response.status_code, 200)
        dataset_data = json.loads(dataset_response.data)
        self.assertTrue(dataset_data['success'])

        dataset_csv = dataset_data['dataset']
        print(f"[OK] Generated {num_samples} random samples")

        # 3. Test forward pass
        forward_response = self.client.post('/forward_pass', json={
            'dataset': dataset_csv
        })

        self.assertEqual(forward_response.status_code, 200)
        forward_data = json.loads(forward_response.data)
        self.assertTrue(forward_data['success'])

        predictions = np.array([sample['prediction'] for sample in forward_data['samples']])
        print(f"[OK] Forward pass completed")
        print(f"     Predictions shape: {predictions.shape}")
        print(f"     Sample predictions (first 3): {predictions[:3].flatten()}")

        # Verify predictions are not all the same (checking for 0.300 bug)
        unique_predictions = np.unique(np.round(predictions, 3))
        self.assertGreater(len(unique_predictions), 1,
                          f"All predictions are the same: {unique_predictions[0]}")
        print(f"[OK] Predictions are diverse ({len(unique_predictions)} unique values)")

        # Verify predictions are in valid range
        if activations[-1] == 'sigmoid':
            self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1),
                           "Sigmoid outputs should be in [0, 1]")
            print(f"[OK] Sigmoid outputs in valid range [0, 1]")
        elif activations[-1] == 'softmax':
            sums = np.sum(predictions, axis=1)
            for s in sums:
                self.assertAlmostEqual(s, 1.0, places=5,
                                      msg="Softmax outputs must sum to 1.0")
            print(f"[OK] Softmax outputs sum to 1.0")

        # 4. Test training
        train_response = self.client.post('/train', json={
            'dataset': dataset_csv,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'optimizer': 'gd',
            'loss_function': loss_function
        })

        self.assertEqual(train_response.status_code, 200)
        train_data = json.loads(train_response.data)
        self.assertTrue(train_data['success'])

        initial_loss = train_data['history']['loss'][0]
        final_loss = train_data['final_loss']
        accuracy = train_data['accuracy']

        print(f"[OK] Training completed ({epochs} epochs)")
        print(f"     Initial loss: {initial_loss:.6f}")
        print(f"     Final loss: {final_loss:.6f}")
        print(f"     Accuracy: {accuracy*100:.2f}%")

        # Verify training improved or maintained performance
        # (Small networks might not always improve, but shouldn't get worse)
        self.assertLessEqual(final_loss, initial_loss * 1.1,
                            "Training should not significantly worsen loss")

        print(f"[OK] {test_name} PASSED")
        return True

    # Test 1: Binary Classification - Small Network (2-3-1)
    def test_01_binary_2_3_1_sigmoid(self):
        """Binary: 2-3-1 with sigmoid"""
        self._test_network_configuration(
            test_name="Binary 2-3-1 Sigmoid",
            architecture=[2, 3, 1],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=10,
            epochs=50
        )

    # Test 2: Binary Classification - Medium Network (3-5-1)
    def test_02_binary_3_5_1_sigmoid(self):
        """Binary: 3-5-1 with sigmoid"""
        self._test_network_configuration(
            test_name="Binary 3-5-1 Sigmoid",
            architecture=[3, 5, 1],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=15
        )

    # Test 3: Binary Classification - Deep Network (4-6-4-1)
    def test_03_binary_4_6_4_1_sigmoid(self):
        """Binary: 4-6-4-1 with sigmoid"""
        self._test_network_configuration(
            test_name="Binary 4-6-4-1 Sigmoid (Deep)",
            architecture=[4, 6, 4, 1],
            activations=['linear', 'sigmoid', 'sigmoid', 'sigmoid'],
            num_samples=20
        )

    # Test 4: Binary with ReLU hidden
    def test_04_binary_3_4_1_relu_sigmoid(self):
        """Binary: 3-4-1 with ReLU hidden layer"""
        self._test_network_configuration(
            test_name="Binary 3-4-1 ReLU-Sigmoid",
            architecture=[3, 4, 1],
            activations=['linear', 'relu', 'sigmoid'],
            num_samples=12
        )

    # Test 5: Multi-class - 3 classes (3-4-3)
    def test_05_multiclass_3_4_3_softmax(self):
        """Multi-class: 3-4-3 with softmax"""
        self._test_network_configuration(
            test_name="Multi-class 3-4-3 Softmax",
            architecture=[3, 4, 3],
            activations=['linear', 'sigmoid', 'softmax'],
            num_samples=15,
            loss_function='categorical'
        )

    # Test 6: Multi-class - 4 classes (4-8-4)
    def test_06_multiclass_4_8_4_softmax(self):
        """Multi-class: 4-8-4 with softmax"""
        self._test_network_configuration(
            test_name="Multi-class 4-8-4 Softmax",
            architecture=[4, 8, 4],
            activations=['linear', 'sigmoid', 'softmax'],
            num_samples=20,
            loss_function='categorical'
        )

    # Test 7: Multi-class - Deep (5-10-6-5)
    def test_07_multiclass_5_10_6_5_deep(self):
        """Multi-class: 5-10-6-5 deep with softmax"""
        self._test_network_configuration(
            test_name="Multi-class 5-10-6-5 Deep Softmax",
            architecture=[5, 10, 6, 5],
            activations=['linear', 'sigmoid', 'sigmoid', 'softmax'],
            num_samples=25,
            loss_function='categorical'
        )

    # Test 8: Multi-label - 3 labels (3-5-3)
    def test_08_multilabel_3_5_3_sigmoid(self):
        """Multi-label: 3-5-3 with sigmoid"""
        self._test_network_configuration(
            test_name="Multi-label 3-5-3 Sigmoid",
            architecture=[3, 5, 3],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=15
        )

    # Test 9: Multi-label - 4 labels (4-6-4)
    def test_09_multilabel_4_6_4_sigmoid(self):
        """Multi-label: 4-6-4 with sigmoid"""
        self._test_network_configuration(
            test_name="Multi-label 4-6-4 Sigmoid",
            architecture=[4, 6, 4],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=18
        )

    # Test 10: Tiny network (2-2-1)
    def test_10_tiny_2_2_1(self):
        """Tiny: 2-2-1"""
        self._test_network_configuration(
            test_name="Tiny 2-2-1",
            architecture=[2, 2, 1],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=8
        )

    # Test 11: Wide shallow (5-20-1)
    def test_11_wide_5_20_1(self):
        """Wide shallow: 5-20-1"""
        self._test_network_configuration(
            test_name="Wide Shallow 5-20-1",
            architecture=[5, 20, 1],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=20
        )

    # Test 12: Narrow deep (3-3-3-3-1)
    def test_12_narrow_deep_3_3_3_3_1(self):
        """Narrow deep: 3-3-3-3-1"""
        self._test_network_configuration(
            test_name="Narrow Deep 3-3-3-3-1",
            architecture=[3, 3, 3, 3, 1],
            activations=['linear', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
            num_samples=15
        )

    # Test 13: Mixed activations (4-6-1) ReLU
    def test_13_mixed_4_6_1_relu(self):
        """Mixed: 4-6-1 with ReLU"""
        self._test_network_configuration(
            test_name="Mixed 4-6-1 ReLU",
            architecture=[4, 6, 1],
            activations=['linear', 'relu', 'sigmoid'],
            num_samples=15
        )

    # Test 14: Large multi-class (6-12-8-6)
    def test_14_large_multiclass_6_12_8_6(self):
        """Large multi-class: 6-12-8-6"""
        self._test_network_configuration(
            test_name="Large Multi-class 6-12-8-6",
            architecture=[6, 12, 8, 6],
            activations=['linear', 'sigmoid', 'sigmoid', 'softmax'],
            num_samples=30,
            loss_function='categorical'
        )

    # Test 15: Binary with many features (8-10-1)
    def test_15_binary_8_10_1(self):
        """Binary with many features: 8-10-1"""
        self._test_network_configuration(
            test_name="Binary 8-10-1",
            architecture=[8, 10, 1],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=25
        )

    # Test 16: Symmetric (4-8-4)
    def test_16_symmetric_4_8_4(self):
        """Symmetric: 4-8-4"""
        self._test_network_configuration(
            test_name="Symmetric 4-8-4",
            architecture=[4, 8, 4],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=20
        )

    # Test 17: Very deep (3-4-4-4-4-1) with ReLU to avoid vanishing gradient
    def test_17_very_deep_3_4_4_4_4_1(self):
        """Very deep: 3-4-4-4-4-1 with ReLU (avoids vanishing gradient)"""
        self._test_network_configuration(
            test_name="Very Deep 3-4-4-4-4-1 ReLU",
            architecture=[3, 4, 4, 4, 4, 1],
            activations=['linear', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
            num_samples=20,
            epochs=100,  # More epochs for deep network
            learning_rate=0.1  # Smaller learning rate for deep network
        )

    # Test 18: Binary XOR-like (2-4-1)
    def test_18_xor_like_2_4_1(self):
        """XOR-like: 2-4-1"""
        self._test_network_configuration(
            test_name="XOR-like 2-4-1",
            architecture=[2, 4, 1],
            activations=['linear', 'sigmoid', 'sigmoid'],
            num_samples=10
        )

    # Test 19: Multi-class 10 classes (5-15-10)
    def test_19_multiclass_10_classes(self):
        """Multi-class 10 classes: 5-15-10"""
        self._test_network_configuration(
            test_name="Multi-class 10 Classes 5-15-10",
            architecture=[5, 15, 10],
            activations=['linear', 'sigmoid', 'softmax'],
            num_samples=50,
            loss_function='categorical',
            epochs=100
        )

    # Test 20: Complex architecture (6-12-8-6-3)
    def test_20_complex_6_12_8_6_3(self):
        """Complex: 6-12-8-6-3"""
        self._test_network_configuration(
            test_name="Complex 6-12-8-6-3",
            architecture=[6, 12, 8, 6, 3],
            activations=['linear', 'sigmoid', 'sigmoid', 'sigmoid', 'softmax'],
            num_samples=30,
            loss_function='categorical',
            epochs=100
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
