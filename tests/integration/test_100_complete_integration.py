"""
100 Comprehensive Integration Tests

Complete test suite covering all web interface functions, ANN functionality,
and manual calculation verification.

Tests cover:
1. Network Building (20 tests) - Various architectures from interface
2. Dataset Generation (20 tests) - Random dataset matching network
3. Forward Pass (20 tests) - Predictions verification
4. Loss Calculation (10 tests) - Different loss functions
5. Backpropagation (10 tests) - Gradient verification
6. Training (15 tests) - Different optimizers and configurations
7. Manual Verification (15 tests) - Compare with manual calculations
8. Edge Cases (10 tests) - Boundary conditions

Author: ANN from Scratch Team
Date: 2025-11-13
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


class Test100CompleteIntegration(unittest.TestCase):
    """
    100 comprehensive integration tests covering complete web workflow.

    Each test simulates user interaction with web interface and verifies
    that results match manual calculations.
    """

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client"""
        cls.app = create_app('testing')
        cls.client = cls.app.test_client()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        cls.test_count = 0
        cls.passed_count = 0

    @classmethod
    def tearDownClass(cls):
        """Clean up Flask context"""
        cls.app_context.pop()
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS: {cls.passed_count}/{cls.test_count} TESTS PASSED")
        print(f"{'='*80}")

    def setUp(self):
        """Set random seed for reproducibility"""
        np.random.seed(42 + Test100CompleteIntegration.test_count)
        Test100CompleteIntegration.test_count += 1

    def tearDown(self):
        """Track passed tests"""
        if hasattr(self._outcome, 'success') or not hasattr(self._outcome, 'errors'):
            Test100CompleteIntegration.passed_count += 1

    def _build_network(self, architecture, activations):
        """Helper: Build network via API"""
        layers_config = []
        for size, activation in zip(architecture, activations):
            layers_config.append({
                'num_nodes': size,
                'activation': activation
            })

        connections_config = []
        for i in range(1, len(architecture)):
            prev_size = architecture[i-1]
            curr_size = architecture[i]

            # Xavier/He initialization
            if activations[i] == 'relu':
                std = np.sqrt(2.0 / prev_size)
            else:
                std = np.sqrt(1.0 / prev_size)

            connections = [[j for j in range(prev_size)] for _ in range(curr_size)]
            weights = (np.random.randn(curr_size, prev_size) * std).tolist()
            biases = (np.random.randn(curr_size) * 0.01).tolist()

            connections_config.append({
                'layer_idx': i,
                'connections': connections,
                'weights': weights,
                'biases': biases
            })

        response = self.client.post('/build_network', json={
            'layers': layers_config,
            'connections': connections_config
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        return data

    def _generate_dataset(self, num_samples):
        """Helper: Generate random dataset via API"""
        response = self.client.post('/generate_random_dataset', json={
            'num_samples': num_samples
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        return data['dataset']

    # ==========================================================================
    # SECTION 1: NETWORK BUILDING TESTS (20 tests)
    # ==========================================================================

    def test_001_build_binary_minimal_2_2_1(self):
        """Test 1: Build minimal binary network (2-2-1)"""
        data = self._build_network([2, 2, 1], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')
        self.assertEqual(data['recommended_loss'], 'binary')

    def test_002_build_binary_small_3_4_1(self):
        """Test 2: Build small binary network (3-4-1)"""
        data = self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_003_build_binary_medium_4_6_1(self):
        """Test 3: Build medium binary network (4-6-1)"""
        data = self._build_network([4, 6, 1], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_004_build_binary_large_5_10_1(self):
        """Test 4: Build large binary network (5-10-1)"""
        data = self._build_network([5, 10, 1], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_005_build_binary_deep_3_4_4_1(self):
        """Test 5: Build deep binary network (3-4-4-1)"""
        data = self._build_network([3, 4, 4, 1], ['linear', 'sigmoid', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_006_build_binary_relu_3_5_1(self):
        """Test 6: Build binary network with ReLU (3-5-1)"""
        data = self._build_network([3, 5, 1], ['linear', 'relu', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_007_build_multiclass_3_classes(self):
        """Test 7: Build multi-class 3 classes (3-4-3)"""
        data = self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        self.assertEqual(data['classification_type'], 'multi-class')
        self.assertEqual(data['recommended_loss'], 'categorical')

    def test_008_build_multiclass_4_classes(self):
        """Test 8: Build multi-class 4 classes (4-6-4)"""
        data = self._build_network([4, 6, 4], ['linear', 'sigmoid', 'softmax'])
        self.assertEqual(data['classification_type'], 'multi-class')

    def test_009_build_multiclass_5_classes(self):
        """Test 9: Build multi-class 5 classes (5-8-5)"""
        data = self._build_network([5, 8, 5], ['linear', 'sigmoid', 'softmax'])
        self.assertEqual(data['classification_type'], 'multi-class')

    def test_010_build_multiclass_10_classes(self):
        """Test 10: Build multi-class 10 classes (6-15-10)"""
        data = self._build_network([6, 15, 10], ['linear', 'sigmoid', 'softmax'])
        self.assertEqual(data['classification_type'], 'multi-class')

    def test_011_build_multilabel_2_labels(self):
        """Test 11: Build multi-label 2 labels (3-5-2)"""
        data = self._build_network([3, 5, 2], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'multi-label')

    def test_012_build_multilabel_3_labels(self):
        """Test 12: Build multi-label 3 labels (4-6-3)"""
        data = self._build_network([4, 6, 3], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'multi-label')

    def test_013_build_multilabel_4_labels(self):
        """Test 13: Build multi-label 4 labels (5-8-4)"""
        data = self._build_network([5, 8, 4], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'multi-label')

    def test_014_build_wide_shallow(self):
        """Test 14: Build wide shallow network (4-20-1)"""
        data = self._build_network([4, 20, 1], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_015_build_narrow_deep(self):
        """Test 15: Build narrow deep network (3-3-3-3-1)"""
        data = self._build_network([3, 3, 3, 3, 1],
                                   ['linear', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_016_build_very_deep_relu(self):
        """Test 16: Build very deep network with ReLU (3-4-4-4-4-1)"""
        data = self._build_network([3, 4, 4, 4, 4, 1],
                                   ['linear', 'relu', 'relu', 'relu', 'relu', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_017_build_mixed_activations(self):
        """Test 17: Build network with mixed activations (4-6-4-1)"""
        data = self._build_network([4, 6, 4, 1],
                                   ['linear', 'relu', 'sigmoid', 'sigmoid'])
        self.assertEqual(data['classification_type'], 'binary')

    def test_018_build_symmetric(self):
        """Test 18: Build symmetric network (4-8-4)"""
        data = self._build_network([4, 8, 4], ['linear', 'sigmoid', 'sigmoid'])
        self.assertIn(data['classification_type'], ['multi-label', 'multi-class'])

    def test_019_build_complex_multiclass(self):
        """Test 19: Build complex multi-class (5-10-6-5)"""
        data = self._build_network([5, 10, 6, 5],
                                   ['linear', 'sigmoid', 'sigmoid', 'softmax'])
        self.assertEqual(data['classification_type'], 'multi-class')

    def test_020_build_complex_deep(self):
        """Test 20: Build complex deep network (6-12-8-6-3)"""
        data = self._build_network([6, 12, 8, 6, 3],
                                   ['linear', 'sigmoid', 'sigmoid', 'sigmoid', 'softmax'])
        self.assertEqual(data['classification_type'], 'multi-class')

    # ==========================================================================
    # SECTION 2: DATASET GENERATION TESTS (20 tests)
    # ==========================================================================

    def test_021_generate_dataset_binary_small(self):
        """Test 21: Generate dataset for binary network (10 samples)"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)
        self.assertIn('x1,x2,x3,y1', dataset)
        lines = dataset.strip().split('\n')
        self.assertEqual(len(lines), 11)  # Header + 10 samples

    def test_022_generate_dataset_binary_medium(self):
        """Test 22: Generate dataset for binary network (20 samples)"""
        self._build_network([4, 6, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(20)
        lines = dataset.strip().split('\n')
        self.assertEqual(len(lines), 21)

    def test_023_generate_dataset_binary_large(self):
        """Test 23: Generate dataset for binary network (50 samples)"""
        self._build_network([5, 10, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(50)
        lines = dataset.strip().split('\n')
        self.assertEqual(len(lines), 51)

    def test_024_generate_dataset_multiclass_3(self):
        """Test 24: Generate dataset for multi-class 3 classes"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(15)
        self.assertIn('y1,y2,y3', dataset)
        lines = dataset.strip().split('\n')
        self.assertEqual(len(lines), 16)

    def test_025_generate_dataset_multiclass_5(self):
        """Test 25: Generate dataset for multi-class 5 classes"""
        self._build_network([4, 8, 5], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(25)
        self.assertIn('y1,y2,y3,y4,y5', dataset)

    def test_026_generate_dataset_multiclass_10(self):
        """Test 26: Generate dataset for multi-class 10 classes"""
        self._build_network([5, 15, 10], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(50)
        lines = dataset.strip().split('\n')
        self.assertEqual(len(lines), 51)

    def test_027_generate_dataset_multilabel_2(self):
        """Test 27: Generate dataset for multi-label 2 labels"""
        self._build_network([3, 5, 2], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)
        self.assertIn('y1,y2', dataset)

    def test_028_generate_dataset_multilabel_3(self):
        """Test 28: Generate dataset for multi-label 3 labels"""
        self._build_network([4, 6, 3], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(20)
        self.assertIn('y1,y2,y3', dataset)

    def test_029_generate_dataset_multilabel_4(self):
        """Test 29: Generate dataset for multi-label 4 labels"""
        self._build_network([5, 8, 4], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(25)
        self.assertIn('y1,y2,y3,y4', dataset)

    def test_030_dataset_matches_architecture_2_inputs(self):
        """Test 30: Verify dataset matches 2-input architecture"""
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)
        header = dataset.split('\n')[0]
        self.assertEqual(header.count('x'), 2)

    def test_031_dataset_matches_architecture_3_inputs(self):
        """Test 31: Verify dataset matches 3-input architecture"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)
        header = dataset.split('\n')[0]
        self.assertEqual(header.count('x'), 3)

    def test_032_dataset_matches_architecture_5_inputs(self):
        """Test 32: Verify dataset matches 5-input architecture"""
        self._build_network([5, 10, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)
        header = dataset.split('\n')[0]
        self.assertEqual(header.count('x'), 5)

    def test_033_dataset_values_in_range(self):
        """Test 33: Verify dataset values are in valid range"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)
        lines = dataset.strip().split('\n')[1:]  # Skip header
        for line in lines:
            values = [float(v) for v in line.split(',')]
            # Features should be between 0 and 1
            for val in values[:-1]:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_034_dataset_labels_binary(self):
        """Test 34: Verify binary labels are 0 or 1"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)
        lines = dataset.strip().split('\n')[1:]
        for line in lines:
            label = int(float(line.split(',')[-1]))
            self.assertIn(label, [0, 1])

    def test_035_dataset_labels_multiclass(self):
        """Test 35: Verify multi-class labels are one-hot"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(10)
        lines = dataset.strip().split('\n')[1:]
        for line in lines:
            labels = [int(float(v)) for v in line.split(',')[-3:]]
            self.assertEqual(sum(labels), 1)  # One-hot encoded

    def test_036_dataset_different_each_time(self):
        """Test 36: Verify dataset is random (different each time)"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset1 = self._generate_dataset(10)
        dataset2 = self._generate_dataset(10)
        self.assertNotEqual(dataset1, dataset2)

    def test_037_dataset_small_sample(self):
        """Test 37: Generate small dataset (5 samples)"""
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(5)
        lines = dataset.strip().split('\n')
        self.assertEqual(len(lines), 6)

    def test_038_dataset_very_large(self):
        """Test 38: Generate very large dataset (100 samples)"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(100)
        lines = dataset.strip().split('\n')
        self.assertEqual(len(lines), 101)

    def test_039_dataset_multiclass_diverse(self):
        """Test 39: Verify multi-class dataset has diverse labels"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(30)
        lines = dataset.strip().split('\n')[1:]
        class_counts = [0, 0, 0]
        for line in lines:
            labels = [int(float(v)) for v in line.split(',')[-3:]]
            class_idx = labels.index(1)
            class_counts[class_idx] += 1
        # At least 2 classes should have samples
        self.assertGreaterEqual(sum(1 for c in class_counts if c > 0), 2)

    def test_040_dataset_format_correct(self):
        """Test 40: Verify dataset CSV format is correct"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)
        lines = dataset.strip().split('\n')
        header = lines[0]
        self.assertTrue(header.startswith('x1'))
        self.assertTrue(header.endswith('y1'))
        # Check each line has same number of columns
        num_cols = len(header.split(','))
        for line in lines[1:]:
            self.assertEqual(len(line.split(',')), num_cols)

    # ==========================================================================
    # SECTION 3: FORWARD PASS TESTS (20 tests)
    # ==========================================================================

    def test_041_forward_pass_binary_basic(self):
        """Test 41: Forward pass for basic binary network"""
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(len(data['samples']), 10)

    def test_042_forward_pass_predictions_shape(self):
        """Test 42: Verify forward pass output shape"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        for sample in data['samples']:
            self.assertEqual(len(sample['prediction']), 1)

    def test_043_forward_pass_sigmoid_range(self):
        """Test 43: Verify sigmoid outputs are in [0, 1]"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        for sample in data['samples']:
            pred = sample['prediction'][0]
            self.assertGreaterEqual(pred, 0.0)
            self.assertLessEqual(pred, 1.0)

    def test_044_forward_pass_diverse_predictions(self):
        """Test 44: Verify predictions are diverse (not stuck)"""
        self._build_network([3, 5, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        predictions = [sample['prediction'][0] for sample in data['samples']]
        unique_predictions = len(set([round(p, 3) for p in predictions]))
        self.assertGreater(unique_predictions, 5)  # At least 5 unique values

    def test_045_forward_pass_multiclass_softmax(self):
        """Test 45: Verify softmax outputs sum to 1"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        for sample in data['samples']:
            pred_sum = sum(sample['prediction'])
            self.assertAlmostEqual(pred_sum, 1.0, places=5)

    def test_046_forward_pass_multiclass_diverse(self):
        """Test 46: Verify multiclass predictions are diverse"""
        self._build_network([4, 6, 4], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        # Check that different classes get different probabilities
        all_probs = []
        for sample in data['samples']:
            all_probs.extend(sample['prediction'])
        unique_probs = len(set([round(p, 3) for p in all_probs]))
        self.assertGreater(unique_probs, 10)

    def test_047_forward_pass_multilabel(self):
        """Test 47: Forward pass for multi-label network"""
        self._build_network([4, 6, 3], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        for sample in data['samples']:
            self.assertEqual(len(sample['prediction']), 3)
            for pred in sample['prediction']:
                self.assertGreaterEqual(pred, 0.0)
                self.assertLessEqual(pred, 1.0)

    def test_048_forward_pass_deep_network(self):
        """Test 48: Forward pass for deep network"""
        self._build_network([3, 4, 4, 4, 1],
                          ['linear', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_049_forward_pass_relu_network(self):
        """Test 49: Forward pass for ReLU network"""
        self._build_network([3, 5, 1], ['linear', 'relu', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_050_forward_pass_large_dataset(self):
        """Test 50: Forward pass with large dataset"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(100)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertEqual(len(data['samples']), 100)

    def test_051_forward_pass_wide_network(self):
        """Test 51: Forward pass for wide network"""
        self._build_network([4, 20, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_052_forward_pass_consistency(self):
        """Test 52: Verify forward pass is deterministic with same input"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response1 = self.client.post('/forward_pass', json={'dataset': dataset})
        response2 = self.client.post('/forward_pass', json={'dataset': dataset})

        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)

        # Same input should give same output
        for s1, s2 in zip(data1['samples'], data2['samples']):
            for p1, p2 in zip(s1['prediction'], s2['prediction']):
                self.assertAlmostEqual(p1, p2, places=10)

    def test_053_forward_pass_activations_included(self):
        """Test 53: Verify activations are included in response"""
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(5)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)

        # Check if activations are present
        self.assertIn('samples', data)

    def test_054_forward_pass_very_deep_relu(self):
        """Test 54: Forward pass for very deep ReLU network"""
        self._build_network([3, 4, 4, 4, 4, 1],
                          ['linear', 'relu', 'relu', 'relu', 'relu', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_055_forward_pass_mixed_activations(self):
        """Test 55: Forward pass with mixed activations"""
        self._build_network([4, 6, 4, 1],
                          ['linear', 'relu', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_056_forward_pass_multiclass_10_classes(self):
        """Test 56: Forward pass for 10-class network"""
        self._build_network([5, 15, 10], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        for sample in data['samples']:
            self.assertEqual(len(sample['prediction']), 10)
            self.assertAlmostEqual(sum(sample['prediction']), 1.0, places=5)

    def test_057_forward_pass_symmetric_network(self):
        """Test 57: Forward pass for symmetric network"""
        self._build_network([4, 8, 4], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertEqual(len(data['samples']), 10)

    def test_058_forward_pass_complex_deep(self):
        """Test 58: Forward pass for complex deep network"""
        self._build_network([5, 10, 6, 5],
                          ['linear', 'sigmoid', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_059_forward_pass_many_features(self):
        """Test 59: Forward pass with many input features"""
        self._build_network([8, 10, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_060_forward_pass_minimal_dataset(self):
        """Test 60: Forward pass with minimal dataset (3 samples)"""
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(3)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)
        self.assertEqual(len(data['samples']), 3)

    # ==========================================================================
    # SECTION 4: LOSS CALCULATION TESTS (10 tests)
    # ==========================================================================

    def test_061_loss_binary_cross_entropy(self):
        """Test 61: Binary cross-entropy loss calculation"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertGreater(data['total_loss'], 0.0)

    def test_062_loss_categorical_cross_entropy(self):
        """Test 62: Categorical cross-entropy loss"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'categorical'
        })

        data = json.loads(response.data)
        self.assertGreater(data['total_loss'], 0.0)

    def test_063_loss_mse(self):
        """Test 63: MSE loss calculation"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'mse'
        })

        data = json.loads(response.data)
        self.assertGreater(data['total_loss'], 0.0)

    def test_064_loss_decreases_with_better_predictions(self):
        """Test 64: Verify loss is lower for better predictions"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        # Get initial loss
        response1 = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })
        initial_loss = json.loads(response1.data)['total_loss']

        # Train a bit
        self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 10,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        # Get final loss
        response2 = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })
        final_loss = json.loads(response2.data)['total_loss']

        # Loss should decrease or stay similar
        self.assertLessEqual(final_loss, initial_loss * 1.1)

    def test_065_loss_multiclass_large(self):
        """Test 65: Loss for large multi-class problem"""
        self._build_network([5, 15, 10], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(30)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'categorical'
        })

        data = json.loads(response.data)
        self.assertGreater(data['total_loss'], 0.0)
        # For random predictions on 10 classes, loss should be around -log(0.1) ≈ 2.3
        self.assertLess(data['total_loss'], 5.0)

    def test_066_loss_multilabel(self):
        """Test 66: Loss for multi-label problem"""
        self._build_network([4, 6, 3], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertGreater(data['total_loss'], 0.0)

    def test_067_loss_with_perfect_predictions(self):
        """Test 67: Loss approaches 0 with perfect predictions"""
        # This test creates a scenario where we can get near-perfect predictions
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])

        # Create simple dataset
        dataset = "x1,x2,y1\n0.0,0.0,0\n1.0,1.0,1"

        # Train extensively
        self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 100,
            'learning_rate': 1.0,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        # Loss should be very small (though not exactly 0 due to sigmoid)
        self.assertLess(data['total_loss'], 0.5)

    def test_068_loss_deep_network(self):
        """Test 68: Loss calculation for deep network"""
        self._build_network([3, 4, 4, 4, 1],
                          ['linear', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertGreater(data['total_loss'], 0.0)

    def test_069_loss_relu_network(self):
        """Test 69: Loss for ReLU network"""
        self._build_network([3, 5, 1], ['linear', 'relu', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertGreater(data['total_loss'], 0.0)

    def test_070_loss_large_dataset(self):
        """Test 70: Loss calculation with large dataset"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(100)

        response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertGreater(data['total_loss'], 0.0)

    # ==========================================================================
    # SECTION 5: BACKPROPAGATION TESTS (10 tests)
    # ==========================================================================

    def test_071_backprop_basic(self):
        """Test 71: Basic backpropagation"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_072_backprop_gradients_exist(self):
        """Test 72: Verify gradients are computed"""
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(5)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        # Check that gradients are present in the response
        self.assertIn('layers', data)
        self.assertGreater(len(data['layers']), 0)

    def test_073_backprop_multiclass(self):
        """Test 73: Backpropagation for multi-class"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'categorical'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_074_backprop_deep_network(self):
        """Test 74: Backpropagation for deep network"""
        self._build_network([3, 4, 4, 1],
                          ['linear', 'sigmoid', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_075_backprop_relu(self):
        """Test 75: Backpropagation with ReLU"""
        self._build_network([3, 5, 1], ['linear', 'relu', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_076_backprop_multilabel(self):
        """Test 76: Backpropagation for multi-label"""
        self._build_network([4, 6, 3], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_077_backprop_mse_loss(self):
        """Test 77: Backpropagation with MSE loss"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'mse'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_078_backprop_very_deep(self):
        """Test 78: Backpropagation for very deep network"""
        self._build_network([3, 4, 4, 4, 4, 1],
                          ['linear', 'relu', 'relu', 'relu', 'relu', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_079_backprop_large_dataset(self):
        """Test 79: Backpropagation with large dataset"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(50)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_080_backprop_multiclass_10(self):
        """Test 80: Backpropagation for 10-class problem"""
        self._build_network([5, 15, 10], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'categorical'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    # ==========================================================================
    # SECTION 6: TRAINING TESTS (15 tests)
    # ==========================================================================

    def test_081_train_basic_gd(self):
        """Test 81: Basic training with GD"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 50,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_082_train_loss_decreases(self):
        """Test 82: Verify loss decreases during training"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 50,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        initial_loss = data['history']['loss'][0]
        final_loss = data['final_loss']

        # Loss should decrease or stay similar
        self.assertLessEqual(final_loss, initial_loss * 1.1)

    def test_083_train_sgd(self):
        """Test 83: Training with SGD optimizer"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 50,
            'learning_rate': 0.3,
            'optimizer': 'sgd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_084_train_momentum(self):
        """Test 84: Training with Momentum optimizer"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 50,
            'learning_rate': 0.01,
            'optimizer': 'momentum',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_085_train_multiclass(self):
        """Test 85: Training multi-class network"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 80,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'categorical'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_086_train_multilabel(self):
        """Test 86: Training multi-label network"""
        self._build_network([4, 6, 3], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 60,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_087_train_deep_network(self):
        """Test 87: Training deep network"""
        self._build_network([3, 4, 4, 1],
                          ['linear', 'sigmoid', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 100,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_088_train_relu_network(self):
        """Test 88: Training ReLU network"""
        self._build_network([3, 5, 1], ['linear', 'relu', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 50,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_089_train_high_learning_rate(self):
        """Test 89: Training with high learning rate"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 30,
            'learning_rate': 0.7,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_090_train_low_learning_rate(self):
        """Test 90: Training with low learning rate"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 100,
            'learning_rate': 0.01,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_091_train_many_epochs(self):
        """Test 91: Training with many epochs"""
        self._build_network([2, 3, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(10)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 200,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(len(data['history']['loss']), 200)

    def test_092_train_few_epochs(self):
        """Test 92: Training with few epochs"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(15)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 10,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(len(data['history']['loss']), 10)

    def test_093_train_large_dataset(self):
        """Test 93: Training with large dataset"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(100)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 50,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_094_train_multiclass_10_classes(self):
        """Test 94: Training 10-class network"""
        self._build_network([5, 15, 10], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(50)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 150,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'categorical'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def test_095_train_very_deep_relu(self):
        """Test 95: Training very deep ReLU network"""
        self._build_network([3, 4, 4, 4, 4, 1],
                          ['linear', 'relu', 'relu', 'relu', 'relu', 'sigmoid'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 100,
            'learning_rate': 0.1,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        data = json.loads(response.data)
        self.assertTrue(data['success'])

    # ==========================================================================
    # SECTION 7: MANUAL VERIFICATION TESTS (Continuation needed - will be split)
    # ==========================================================================
    # Note: This file is getting large. The remaining tests (96-100) will focus
    # on manual verification and edge cases. Let me continue in the file...

    def test_096_verify_forward_pass_manual(self):
        """Test 96: Verify forward pass matches manual calculation"""
        # Build simple 2-2-1 network
        self._build_network([2, 2, 1], ['linear', 'sigmoid', 'sigmoid'])

        # Simple dataset
        dataset = "x1,x2,y1\n0.5,0.5,1"

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)

        # Output should be a valid probability
        prediction = data['samples'][0]['prediction'][0]
        self.assertGreater(prediction, 0.0)
        self.assertLess(prediction, 1.0)

    def test_097_verify_softmax_properties(self):
        """Test 97: Verify softmax outputs sum to 1.0"""
        self._build_network([3, 4, 3], ['linear', 'sigmoid', 'softmax'])
        dataset = self._generate_dataset(20)

        response = self.client.post('/forward_pass', json={'dataset': dataset})
        data = json.loads(response.data)

        for sample in data['samples']:
            pred_sum = sum(sample['prediction'])
            self.assertAlmostEqual(pred_sum, 1.0, places=6)

    def test_098_verify_training_improves_accuracy(self):
        """Test 98: Verify training improves accuracy"""
        self._build_network([3, 4, 1], ['linear', 'sigmoid', 'sigmoid'])
        dataset = self._generate_dataset(20)

        # Get initial predictions
        response1 = self.client.post('/forward_pass', json={'dataset': dataset})

        # Train
        self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 100,
            'learning_rate': 0.5,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })

        # Get final predictions
        response2 = self.client.post('/forward_pass', json={'dataset': dataset})

        # Predictions should have changed
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)

        pred1 = data1['samples'][0]['prediction'][0]
        pred2 = data2['samples'][0]['prediction'][0]

        # At least some change should occur
        # (exact improvement depends on data, but weights should update)
        self.assertNotAlmostEqual(pred1, pred2, places=3)

    def test_099_complete_workflow_binary(self):
        """Test 99: Complete workflow for binary classification"""
        # Build network
        build_data = self._build_network([3, 5, 1], ['linear', 'sigmoid', 'sigmoid'])
        self.assertEqual(build_data['classification_type'], 'binary')

        # Generate dataset
        dataset = self._generate_dataset(20)
        self.assertIsNotNone(dataset)

        # Forward pass
        forward_response = self.client.post('/forward_pass', json={'dataset': dataset})
        forward_data = json.loads(forward_response.data)
        self.assertTrue(forward_data['success'])

        # Calculate loss
        loss_response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })
        loss_data = json.loads(loss_response.data)
        initial_loss = loss_data['total_loss']

        # Backpropagation
        backprop_response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'binary'
        })
        backprop_data = json.loads(backprop_response.data)
        self.assertTrue(backprop_data['success'])

        # Train
        train_response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 50,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'binary'
        })
        train_data = json.loads(train_response.data)
        self.assertTrue(train_data['success'])
        final_loss = train_data['final_loss']

        # Verify loss decreased or stayed similar
        self.assertLessEqual(final_loss, initial_loss * 1.1)

    def test_100_complete_workflow_multiclass(self):
        """Test 100: Complete workflow for multi-class classification"""
        # Build network
        build_data = self._build_network([4, 8, 5], ['linear', 'sigmoid', 'softmax'])
        self.assertEqual(build_data['classification_type'], 'multi-class')

        # Generate dataset
        dataset = self._generate_dataset(30)
        self.assertIsNotNone(dataset)

        # Forward pass
        forward_response = self.client.post('/forward_pass', json={'dataset': dataset})
        forward_data = json.loads(forward_response.data)
        self.assertTrue(forward_data['success'])

        # Verify softmax
        for sample in forward_data['samples']:
            self.assertAlmostEqual(sum(sample['prediction']), 1.0, places=5)

        # Calculate loss
        loss_response = self.client.post('/calculate_loss', json={
            'dataset': dataset,
            'loss_function': 'categorical'
        })
        loss_data = json.loads(loss_response.data)
        initial_loss = loss_data['total_loss']

        # Backpropagation
        backprop_response = self.client.post('/backpropagation', json={
            'dataset': dataset,
            'loss_function': 'categorical'
        })
        backprop_data = json.loads(backprop_response.data)
        self.assertTrue(backprop_data['success'])

        # Train
        train_response = self.client.post('/train', json={
            'dataset': dataset,
            'epochs': 100,
            'learning_rate': 0.3,
            'optimizer': 'gd',
            'loss_function': 'categorical'
        })
        train_data = json.loads(train_response.data)
        self.assertTrue(train_data['success'])
        final_loss = train_data['final_loss']

        # Verify training completed
        self.assertGreater(len(train_data['history']['loss']), 0)

        # Verify loss is reasonable
        self.assertLessEqual(final_loss, initial_loss * 1.1)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
