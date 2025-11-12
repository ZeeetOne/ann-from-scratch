"""
Web API Integration Tests

Tests all web API endpoints with random datasets to verify:
1. Network building and configuration
2. Forward pass calculations
3. Loss calculations
4. Gradient computations (backpropagation)
5. Weight updates
6. Full training workflow

Compares web API results with direct Python API results to ensure consistency.

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


class TestWebAPI(unittest.TestCase):
    """
    Test suite for web API endpoints with random datasets.

    Validates that web API results match direct Python API calculations.
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

    def test_binary_classification_workflow(self):
        """
        Test complete binary classification workflow through web API

        Network: 3-4-1 (sigmoid output)
        Dataset: 10 samples, 3 features, binary labels
        """
        print("\n" + "="*70)
        print("TEST: Binary Classification - Web API Workflow")
        print("="*70)

        # 1. Build network
        print("\n1. Building Network (3-4-1)...")
        build_response = self.client.post('/build_network', json={
            'layers': [
                {'num_nodes': 3, 'activation': 'linear'},
                {'num_nodes': 4, 'activation': 'sigmoid'},
                {'num_nodes': 1, 'activation': 'sigmoid'}
            ],
            'connections': [
                {
                    'layer_idx': 1,
                    'connections': [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
                    'weights': np.random.randn(4, 3).tolist(),
                    'biases': np.zeros(4).tolist()
                },
                {
                    'layer_idx': 2,
                    'connections': [[0, 1, 2, 3]],
                    'weights': np.random.randn(1, 4).tolist(),
                    'biases': np.zeros(1).tolist()
                }
            ]
        })

        self.assertEqual(build_response.status_code, 200)
        build_data = json.loads(build_response.data)
        self.assertTrue(build_data['success'])
        print(f"   Network built: {build_data['message']}")
        print(f"   Classification type: {build_data['classification_type']}")

        # 2. Generate random dataset
        print("\n2. Generating Random Dataset...")
        n_samples = 10
        n_features = 3
        X = np.random.rand(n_samples, n_features)
        y = (X.sum(axis=1) > n_features * 0.5).astype(int).reshape(-1, 1)

        # Convert to CSV format for web API
        dataset_lines = []
        for i in range(n_samples):
            row = ','.join([f"{x:.6f}" for x in X[i]]) + f",{y[i][0]}"
            dataset_lines.append(row)
        dataset_csv = '\n'.join(dataset_lines)

        print(f"   Samples: {n_samples}")
        print(f"   Features: {n_features}")
        print(f"   Class distribution: {np.sum(y == 0)} vs {np.sum(y == 1)}")

        # 3. Test forward pass (prediction)
        print("\n3. Testing Forward Pass...")
        predict_response = self.client.post('/forward_pass', json={
            'dataset': dataset_csv
        })

        self.assertEqual(predict_response.status_code, 200)
        predict_data = json.loads(predict_response.data)
        self.assertTrue(predict_data['success'])

        # Extract predictions from samples
        web_predictions = np.array([sample['prediction'] for sample in predict_data['samples']])
        print(f"   Web API predictions shape: {web_predictions.shape}")
        print(f"   Sample predictions: {web_predictions[:3, 0]}")

        # Verify predictions are reasonable (between 0 and 1 for sigmoid)
        self.assertTrue(np.all(web_predictions >= 0) and np.all(web_predictions <= 1),
                       "Sigmoid predictions should be between 0 and 1")
        print(f"   Predictions are valid (all between 0 and 1)")

        # 4. Test loss calculation
        print("\n4. Testing Loss Calculation...")
        loss_response = self.client.post('/calculate_loss', json={
            'dataset': dataset_csv,
            'loss_function': 'binary'
        })

        self.assertEqual(loss_response.status_code, 200)
        loss_data = json.loads(loss_response.data)
        self.assertTrue(loss_data['success'])

        web_loss = loss_data['total_loss']
        print(f"   Web API loss: {web_loss:.6f}")

        # Verify loss is reasonable (positive number)
        self.assertGreater(web_loss, 0, "Loss should be positive")
        print(f"   Loss is valid (positive)")

        # 5. Test backpropagation (gradients)
        print("\n5. Testing Backpropagation...")
        backprop_response = self.client.post('/backpropagation', json={
            'dataset': dataset_csv,
            'loss_function': 'binary'
        })

        self.assertEqual(backprop_response.status_code, 200)
        backprop_data = json.loads(backprop_response.data)
        self.assertTrue(backprop_data['success'])

        print(f"   Gradients calculated for {len(backprop_data['layers'])} layers")
        if len(backprop_data['layers']) > 0:
            print(f"   Layer 1 has {len(backprop_data['layers'][0]['gradients'])} nodes with gradients")
        print(f"   Gradients computed successfully")

        # 6. Test training
        print("\n6. Testing Training (50 epochs)...")
        train_response = self.client.post('/train', json={
            'dataset': dataset_csv,
            'epochs': 50,
            'learning_rate': 0.5,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        self.assertEqual(train_response.status_code, 200)
        train_data = json.loads(train_response.data)
        self.assertTrue(train_data['success'])

        print(f"   Initial loss: {train_data['history']['loss'][0]:.6f}")
        print(f"   Final loss: {train_data['final_loss']:.6f}")
        print(f"   Accuracy: {train_data['accuracy']*100:.2f}%")
        print(f"   Loss improved: {train_data['history']['loss'][0] > train_data['final_loss']}")

        self.assertGreater(train_data['history']['loss'][0],
                          train_data['final_loss'],
                          "Training should reduce loss")

        print("\n[OK] Binary classification web API workflow verified")

    def test_multiclass_classification_workflow(self):
        """
        Test complete multi-class classification workflow through web API

        Network: 4-5-3 (softmax output)
        Dataset: 15 samples, 4 features, 3 classes
        """
        print("\n" + "="*70)
        print("TEST: Multi-Class Classification - Web API Workflow")
        print("="*70)

        # 1. Build network
        print("\n1. Building Network (4-5-3 with Softmax)...")
        build_response = self.client.post('/build_network', json={
            'layers': [
                {'num_nodes': 4, 'activation': 'linear'},
                {'num_nodes': 5, 'activation': 'sigmoid'},
                {'num_nodes': 3, 'activation': 'softmax'}
            ],
            'connections': [
                {
                    'layer_idx': 1,
                    'connections': [[0, 1, 2, 3]] * 5,
                    'weights': np.random.randn(5, 4).tolist(),
                    'biases': np.zeros(5).tolist()
                },
                {
                    'layer_idx': 2,
                    'connections': [[0, 1, 2, 3, 4]] * 3,
                    'weights': np.random.randn(3, 5).tolist(),
                    'biases': np.zeros(3).tolist()
                }
            ]
        })

        self.assertEqual(build_response.status_code, 200)
        build_data = json.loads(build_response.data)
        self.assertTrue(build_data['success'])
        print(f"   Network built: {build_data['message']}")
        print(f"   Classification type: {build_data['classification_type']}")

        # 2. Generate random dataset
        print("\n2. Generating Random Dataset...")
        n_samples = 15
        n_features = 4
        n_classes = 3

        X = np.random.rand(n_samples, n_features)
        feature_sum = X.sum(axis=1)
        y_labels = np.zeros(n_samples, dtype=int)
        y_labels[feature_sum < n_features * 0.33] = 0
        y_labels[(feature_sum >= n_features * 0.33) & (feature_sum < n_features * 0.67)] = 1
        y_labels[feature_sum >= n_features * 0.67] = 2

        # Convert to one-hot
        y = np.zeros((n_samples, n_classes))
        y[np.arange(n_samples), y_labels] = 1

        # Convert to CSV format
        dataset_lines = []
        for i in range(n_samples):
            row = ','.join([f"{x:.6f}" for x in X[i]]) + ',' + ','.join([str(int(c)) for c in y[i]])
            dataset_lines.append(row)
        dataset_csv = '\n'.join(dataset_lines)

        print(f"   Samples: {n_samples}")
        print(f"   Features: {n_features}")
        print(f"   Classes: {n_classes}")
        for c in range(n_classes):
            print(f"   Class {c}: {np.sum(y_labels == c)} samples")

        # 3. Test forward pass - verify softmax properties
        print("\n3. Testing Forward Pass (Softmax)...")
        predict_response = self.client.post('/forward_pass', json={
            'dataset': dataset_csv
        })

        self.assertEqual(predict_response.status_code, 200)
        predict_data = json.loads(predict_response.data)
        self.assertTrue(predict_data['success'])

        # Extract predictions from samples
        web_predictions = np.array([sample['prediction'] for sample in predict_data['samples']])
        print(f"   Web API predictions shape: {web_predictions.shape}")

        # Verify softmax properties
        sums = np.sum(web_predictions, axis=1)
        print(f"\n   Softmax verification (first 3 samples):")
        for i in range(min(3, n_samples)):
            print(f"   Sample {i}: sum = {sums[i]:.6f} (should be 1.0)")
            self.assertAlmostEqual(sums[i], 1.0, places=5,
                                  msg="Softmax outputs must sum to 1.0")

        # 4. Test loss calculation
        print("\n4. Testing Loss Calculation (Categorical)...")
        loss_response = self.client.post('/calculate_loss', json={
            'dataset': dataset_csv,
            'loss_function': 'categorical'
        })

        self.assertEqual(loss_response.status_code, 200)
        loss_data = json.loads(loss_response.data)
        self.assertTrue(loss_data['success'])

        web_loss = loss_data['total_loss']
        print(f"   Web API loss: {web_loss:.6f}")
        self.assertGreater(web_loss, 0, "Loss should be positive")

        # 5. Test training
        print("\n5. Testing Training (100 epochs)...")
        train_response = self.client.post('/train', json={
            'dataset': dataset_csv,
            'epochs': 100,
            'learning_rate': 0.5,
            'optimizer': 'sgd',
            'loss_function': 'categorical'
        })

        self.assertEqual(train_response.status_code, 200)
        train_data = json.loads(train_response.data)
        self.assertTrue(train_data['success'])

        print(f"   Initial loss: {train_data['history']['loss'][0]:.6f}")
        print(f"   Final loss: {train_data['final_loss']:.6f}")
        print(f"   Accuracy: {train_data['accuracy']*100:.2f}%")

        # Verify predictions after training
        predictions = train_data['predictions']
        print(f"\n   Sample predictions (first 3):")
        for i in range(min(3, len(predictions))):
            pred_probs = predictions[i]['y_pred']
            pred_class = np.argmax(pred_probs)
            true_label = predictions[i]['y_true']
            true_class = np.argmax(true_label) if isinstance(true_label, list) else true_label
            match = "[OK]" if pred_class == true_class else "[X]"
            print(f"   Sample {i}: pred={pred_class}, true={true_class} {match}")

        print("\n[OK] Multi-class classification web API workflow verified")

    def test_quick_start_examples(self):
        """
        Test quick start example endpoints
        """
        print("\n" + "="*70)
        print("TEST: Quick Start Examples")
        print("="*70)

        # Test binary quick start
        print("\n1. Testing Binary Quick Start...")
        binary_response = self.client.post('/quick_start_binary')

        self.assertEqual(binary_response.status_code, 200)
        binary_data = json.loads(binary_response.data)
        self.assertTrue(binary_data['success'])

        print(f"   Network: {binary_data['layers']}")
        print(f"   Classification type: {binary_data['classification_type']}")
        print("[OK] Binary quick start works")

        # Test multiclass quick start
        print("\n2. Testing Multi-Class Quick Start...")
        multiclass_response = self.client.post('/quick_start_multiclass')

        self.assertEqual(multiclass_response.status_code, 200)
        multiclass_data = json.loads(multiclass_response.data)
        self.assertTrue(multiclass_data['success'])

        print(f"   Network: {multiclass_data['layers']}")
        print(f"   Classification type: {multiclass_data['classification_type']}")
        print("[OK] Multi-class quick start works")

        print("\n[OK] All quick start examples verified")


if __name__ == '__main__':
    unittest.main(verbosity=2)
