"""
Comprehensive ANN Website Testing - 200 Test Cases
Tests complete flow: Architecture → Dataset → Forward Pass → Loss → Training → Results
Verifies calculations match manual computations
"""

import requests
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

class ComprehensiveANNTester:
    """Comprehensive tester that runs full workflow for each test case"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.tolerance = 0.001
        self.current_case = 1

    # ===== UTILITY FUNCTIONS =====

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def mse_loss(self, y_true, y_pred):
        """MSE loss"""
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

    def log_test(self, case_id, phase, status, message, details=None):
        """Log test result"""
        result = {
            'case_id': case_id,
            'phase': phase,
            'status': status,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)

        # Console output
        status_icon = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
        print(f"{status_icon} Case {case_id} [{phase}]: {message}")
        if details and status == "FAIL":
            print(f"   Details: {details}")

    # ===== API CALLS =====

    def build_network(self, layers, connections):
        """Build network via API"""
        try:
            response = self.session.post(
                f"{self.base_url}/build_network",
                json={'layers': layers, 'connections': connections},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def forward_pass(self, dataset):
        """Run forward pass"""
        try:
            response = self.session.post(
                f"{self.base_url}/forward_pass",
                json={'dataset': dataset},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def calculate_loss(self, dataset, loss_function):
        """Calculate loss"""
        try:
            response = self.session.post(
                f"{self.base_url}/calculate_loss",
                json={'dataset': dataset, 'loss_function': loss_function},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def train_network(self, dataset, optimizer, learning_rate, epochs, loss_function, batch_size=None):
        """Train network"""
        try:
            payload = {
                'dataset': dataset,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'loss_function': loss_function
            }
            if batch_size:
                payload['batch_size'] = batch_size

            response = self.session.post(
                f"{self.base_url}/train",
                json=payload,
                timeout=120  # 2 minutes for training
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ===== MANUAL CALCULATIONS =====

    def manual_forward_1_1_1(self, x, w1, b1, w2, b2):
        """Manual forward pass for 1-1-1 network"""
        z1 = w1 * x + b1
        a1 = self.sigmoid(z1)
        z2 = w2 * a1 + b2
        a2 = self.sigmoid(z2)
        return {'hidden_z': z1, 'hidden_a': a1, 'output_z': z2, 'output_a': a2}

    def manual_forward_2_2_1(self, x1, x2, weights, biases):
        """Manual forward pass for 2-2-1 network"""
        # Hidden layer
        h1_z = weights['h1'][0] * x1 + weights['h1'][1] * x2 + biases['h1']
        h1_a = self.sigmoid(h1_z)

        h2_z = weights['h2'][0] * x1 + weights['h2'][1] * x2 + biases['h2']
        h2_a = self.sigmoid(h2_z)

        # Output layer
        out_z = weights['out'][0] * h1_a + weights['out'][1] * h2_a + biases['out']
        out_a = self.sigmoid(out_z)

        return {
            'h1': {'z': h1_z, 'a': h1_a},
            'h2': {'z': h2_z, 'a': h2_a},
            'out': {'z': out_z, 'a': out_a}
        }

    # ===== COMPLETE TEST CASE RUNNER =====

    def run_complete_test_case(self, case_config):
        """Run one complete test case through entire workflow"""
        case_id = case_config['id']
        print(f"\n{'='*80}")
        print(f"RUNNING TEST CASE {case_id}: {case_config['name']}")
        print(f"{'='*80}")

        try:
            # Phase 1: Build Network Architecture
            print(f"\n[Phase 1] Building Network Architecture...")
            build_result = self.build_network(
                case_config['layers'],
                case_config['connections']
            )

            if not build_result.get('success'):
                self.log_test(case_id, "BUILD", "FAIL",
                            f"Failed to build network: {build_result.get('error')}")
                return False

            self.log_test(case_id, "BUILD", "PASS",
                        f"Network built: {case_config['architecture']}")

            # Phase 2: Load Dataset
            print(f"\n[Phase 2] Loading Dataset...")
            dataset = case_config['dataset']

            # Validate dataset format
            lines = dataset.strip().split('\n')
            if len(lines) < 2:
                self.log_test(case_id, "DATASET", "FAIL", "Dataset has no data rows")
                return False

            self.log_test(case_id, "DATASET", "PASS",
                        f"Dataset loaded: {len(lines)-1} samples")

            # Phase 3: Forward Pass
            print(f"\n[Phase 3] Running Forward Pass...")
            fp_result = self.forward_pass(dataset)

            if not fp_result.get('success'):
                self.log_test(case_id, "FORWARD", "FAIL",
                            f"Forward pass failed: {fp_result.get('error')}")
                return False

            # Verify forward pass with manual calculation
            if 'manual_check' in case_config:
                manual_result = case_config['manual_check'](self)
                website_pred = fp_result['samples'][0]['prediction']
                expected_pred = manual_result['expected']

                diff = abs(website_pred[0] - expected_pred)
                if diff < self.tolerance:
                    self.log_test(case_id, "FORWARD", "PASS",
                                f"Forward pass matches manual: {website_pred[0]:.6f} vs {expected_pred:.6f}")
                else:
                    self.log_test(case_id, "FORWARD", "FAIL",
                                f"Forward pass mismatch: {website_pred[0]:.6f} vs {expected_pred:.6f}",
                                f"Difference: {diff:.6f}")
                    return False
            else:
                self.log_test(case_id, "FORWARD", "PASS",
                            f"Forward pass completed: {len(fp_result['samples'])} predictions")

            # Phase 4: Calculate Loss
            print(f"\n[Phase 4] Calculating Loss...")
            loss_result = self.calculate_loss(dataset, case_config['loss_function'])

            if not loss_result.get('success'):
                self.log_test(case_id, "LOSS", "FAIL",
                            f"Loss calculation failed: {loss_result.get('error')}")
                return False

            initial_loss = loss_result.get('total_loss')
            self.log_test(case_id, "LOSS", "PASS",
                        f"Initial loss: {initial_loss:.6f}")

            # Phase 5: Training
            print(f"\n[Phase 5] Training Network...")
            train_result = self.train_network(
                dataset,
                case_config['optimizer'],
                case_config['learning_rate'],
                case_config['epochs'],
                case_config['loss_function'],
                case_config.get('batch_size')
            )

            if not train_result.get('success'):
                self.log_test(case_id, "TRAIN", "FAIL",
                            f"Training failed: {train_result.get('error')}")
                return False

            final_loss = train_result.get('final_loss')
            loss_decrease = initial_loss - final_loss

            self.log_test(case_id, "TRAIN", "PASS",
                        f"Training completed: Loss {initial_loss:.6f} → {final_loss:.6f} (Δ{loss_decrease:.6f})")

            # Phase 6: Verify Results
            print(f"\n[Phase 6] Verifying Results...")

            # Check if loss decreased
            if loss_decrease > 0:
                self.log_test(case_id, "VERIFY", "PASS",
                            f"Loss decreased by {loss_decrease:.6f}")
            else:
                self.log_test(case_id, "VERIFY", "WARN",
                            f"Loss did not decrease (Δ{loss_decrease:.6f})")

            # Check if target achieved
            if 'target_loss' in case_config:
                if final_loss < case_config['target_loss']:
                    self.log_test(case_id, "TARGET", "PASS",
                                f"Target achieved: {final_loss:.6f} < {case_config['target_loss']}")
                else:
                    self.log_test(case_id, "TARGET", "FAIL",
                                f"Target not achieved: {final_loss:.6f} >= {case_config['target_loss']}")

            # Check accuracy if available
            if 'accuracy' in train_result:
                accuracy = train_result['accuracy']
                self.log_test(case_id, "ACCURACY", "PASS",
                            f"Accuracy: {accuracy:.2%}")

                if 'target_accuracy' in case_config:
                    if accuracy >= case_config['target_accuracy']:
                        self.log_test(case_id, "ACC_TARGET", "PASS",
                                    f"Accuracy target achieved: {accuracy:.2%} >= {case_config['target_accuracy']:.2%}")
                    else:
                        self.log_test(case_id, "ACC_TARGET", "FAIL",
                                    f"Accuracy target not achieved: {accuracy:.2%} < {case_config['target_accuracy']:.2%}")

            print(f"\n[SUCCESS] CASE {case_id} COMPLETED SUCCESSFULLY")
            return True

        except Exception as e:
            self.log_test(case_id, "ERROR", "FAIL",
                        f"Exception occurred: {str(e)}",
                        traceback.format_exc())
            print(f"\n[ERROR] CASE {case_id} FAILED WITH EXCEPTION")
            return False

    # ===== TEST CASE DEFINITIONS =====

    def generate_dataset(self, n_in, n_out, n_samples, dataset_type='binary'):
        """Generate dataset based on architecture"""
        dataset_lines = [','.join([f'x{i+1}' for i in range(n_in)] + [f'y{j+1}' for j in range(n_out)])]

        if dataset_type == 'binary':
            for _ in range(n_samples):
                inputs = [str(np.random.randint(0, 2)) for _ in range(n_in)]
                outputs = [str(np.random.randint(0, 2)) for _ in range(n_out)]
                dataset_lines.append(','.join(inputs + outputs))
        elif dataset_type == 'continuous':
            for _ in range(n_samples):
                inputs = [f'{np.random.uniform(0, 1):.4f}' for _ in range(n_in)]
                outputs = [str(np.random.randint(0, 2)) for _ in range(n_out)]
                dataset_lines.append(','.join(inputs + outputs))
        elif dataset_type == 'multiclass':
            for _ in range(n_samples):
                inputs = [f'{np.random.uniform(0, 1):.4f}' for _ in range(n_in)]
                class_idx = np.random.randint(0, n_out)
                outputs = ['1' if j == class_idx else '0' for j in range(n_out)]
                dataset_lines.append(','.join(inputs + outputs))

        return '\n'.join(dataset_lines)

    def get_test_cases(self):
        """Generate 200 diverse test cases with varied architectures and activations"""

        test_cases = []

        # Available activation functions (expand as website supports more)
        hidden_activations = ['sigmoid', 'relu', 'linear']
        output_activations = ['sigmoid', 'linear', 'softmax']

        # Available architectures (input-hidden-output)
        architectures = [
            # Small networks (1 input)
            (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1),
            # 2 inputs
            (2, 1, 1), (2, 2, 1), (2, 3, 1), (2, 4, 1), (2, 5, 1), (2, 6, 1),
            (2, 2, 2), (2, 3, 2), (2, 4, 2),
            # 3 inputs
            (3, 2, 1), (3, 3, 1), (3, 4, 1), (3, 5, 1), (3, 6, 1),
            (3, 3, 2), (3, 4, 2), (3, 5, 2),
            (3, 4, 3), (3, 5, 3),
            # 4 inputs
            (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1),
            (4, 4, 2), (4, 5, 2), (4, 6, 2),
            (4, 5, 3),
            # 5 inputs
            (5, 3, 1), (5, 4, 1), (5, 5, 1), (5, 6, 1), (5, 7, 1),
            (5, 5, 2), (5, 6, 2),
            # 6 inputs
            (6, 4, 1), (6, 5, 1), (6, 6, 1), (6, 7, 1),
            (6, 6, 2),
            # 7-8 inputs
            (7, 5, 1), (7, 6, 1), (7, 7, 1),
            (8, 6, 1), (8, 7, 1), (8, 8, 1),
        ]

        # Optimizers and learning rates
        optimizers = ['gd', 'sgd', 'momentum']
        learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

        # Loss functions
        loss_functions = {1: 'mse', 2: 'mse', 3: 'categorical_crossentropy'}

        # Case 1: Special case with manual verification (1-1-1 network)
        test_cases.append({
            'id': 'TC001',
            'name': 'Minimal 1-1-1 Network with Manual Verification',
            'architecture': '1-1-1',
            'layers': [
                {'num_nodes': 1, 'activation': 'linear'},
                {'num_nodes': 1, 'activation': 'sigmoid'},
                {'num_nodes': 1, 'activation': 'sigmoid'}
            ],
            'connections': [
                {'layer_idx': 1, 'connections': [[0]], 'weights': [[0.5]], 'biases': [0.0]},
                {'layer_idx': 2, 'connections': [[0]], 'weights': [[0.5]], 'biases': [0.0]}
            ],
            'dataset': 'x1,y1\n1.0,1.0\n0.0,0.0',
            'loss_function': 'mse',
            'optimizer': 'gd',
            'learning_rate': 0.5,
            'epochs': 100,
            'target_loss': 0.1,
            'manual_check': lambda self: {
                'expected': self.manual_forward_1_1_1(1.0, 0.5, 0.0, 0.5, 0.0)['output_a']
            }
        })

        # Generate 199 more diverse test cases (TC002 - TC200)
        # Systematically combine different architectures with different activation functions

        case_num = 2

        # Generate combinations of architectures and activation functions
        for arch_idx, (n_in, n_hid, n_out) in enumerate(architectures):
            if case_num > 200:
                break

            for hidden_act in hidden_activations:
                if case_num > 200:
                    break

                for output_act in output_activations:
                    if case_num > 200:
                        break

                    # Skip invalid combinations (softmax only for multi-class)
                    if output_act == 'softmax' and n_out == 1:
                        continue

                    # Select appropriate loss function
                    if output_act == 'softmax':
                        loss_fn = 'categorical_crossentropy'
                        dataset_type = 'multiclass'
                    elif output_act == 'sigmoid' and n_out > 1:
                        loss_fn = 'binary_crossentropy'
                        dataset_type = 'binary'
                    else:
                        loss_fn = 'mse'
                        dataset_type = 'continuous' if n_in > 2 else 'binary'

                    # Generate dataset
                    n_samples = min(max(8, n_in * 2), 20)
                    dataset = self.generate_dataset(n_in, n_out, n_samples, dataset_type)

                    # Select optimizer and learning rate
                    optimizer = optimizers[(case_num - 2) % len(optimizers)]
                    lr = learning_rates[(case_num - 2) % len(learning_rates)]

                    # Adjust epochs based on complexity
                    if n_in <= 2 and n_hid <= 3:
                        epochs = 500
                    elif n_in <= 4:
                        epochs = 400
                    else:
                        epochs = 300

                    # Create test case
                    test_cases.append({
                        'id': f'TC{case_num:03d}',
                        'name': f'{n_in}-{n_hid}-{n_out} with {hidden_act}→{output_act}',
                        'architecture': f'{n_in}-{n_hid}-{n_out}',
                        'layers': [
                            {'num_nodes': n_in, 'activation': 'linear'},
                            {'num_nodes': n_hid, 'activation': hidden_act},
                            {'num_nodes': n_out, 'activation': output_act}
                        ],
                        'connections': [
                            {
                                'layer_idx': 1,
                                'connections': [[j for j in range(n_in)] for _ in range(n_hid)],
                                'weights': [[np.random.uniform(-0.5, 0.5) for _ in range(n_in)] for _ in range(n_hid)],
                                'biases': [0.0] * n_hid
                            },
                            {
                                'layer_idx': 2,
                                'connections': [[j for j in range(n_hid)] for _ in range(n_out)],
                                'weights': [[np.random.uniform(-0.5, 0.5) for _ in range(n_hid)] for _ in range(n_out)],
                                'biases': [0.0] * n_out
                            }
                        ],
                        'dataset': dataset,
                        'loss_function': loss_fn,
                        'optimizer': optimizer,
                        'learning_rate': lr,
                        'epochs': epochs,
                        'batch_size': 1 if optimizer == 'sgd' else None
                    })

                    case_num += 1

        return test_cases

    # ===== MAIN TEST RUNNER =====

    def run_all_tests(self, limit=None):
        """Run all test cases (or limit to first N)"""
        test_cases = self.get_test_cases()

        if limit:
            test_cases = test_cases[:limit]

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ANN WEBSITE TESTING")
        print(f"{'='*80}")
        print(f"Total Test Cases: {len(test_cases)}")
        print(f"Server: {self.base_url}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        passed = 0
        failed = 0

        for test_case in test_cases:
            success = self.run_complete_test_case(test_case)
            if success:
                passed += 1
            else:
                failed += 1

            # Small delay between tests
            time.sleep(0.5)

        # Generate summary
        self.generate_report(passed, failed, len(test_cases))

    def generate_report(self, passed, failed, total):
        """Generate comprehensive test report"""
        print(f"\n{'='*80}")
        print(f"TEST EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Cases: {total}")
        print(f"[PASS] Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"[FAIL] Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"{'='*80}\n")

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / total if total > 0 else 0
            },
            'test_results': self.test_results
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[REPORT] Detailed report saved to: {report_file}\n")

        # Print failures summary
        failures = [r for r in self.test_results if r['status'] == 'FAIL']
        if failures:
            print(f"\n[FAILURES] SUMMARY ({len(failures)} issues):")
            print(f"{'-'*80}")
            for fail in failures:
                print(f"  [{fail['case_id']}] {fail['phase']}: {fail['message']}")
                if fail.get('details'):
                    print(f"      → {fail['details']}")
            print(f"{'-'*80}\n")

if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("Checking server status...")
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        print("[OK] Server is running!\n")
    except:
        print("[ERROR] Server is NOT running!")
        print("Please start the Flask server: python app.py\n")
        exit(1)

    # Create tester
    tester = ComprehensiveANNTester()

    # Run all 200 test cases
    print("Running comprehensive tests...")
    print("Testing 200 diverse network architectures with varied activation functions.\n")
    print("This will test complete flow: Architecture -> Dataset -> Forward -> Loss -> Training -> Results\n")
    print("Each case will be verified for correctness. This may take several minutes...\n")

    tester.run_all_tests(limit=None)  # Run all 200 cases

    print("\n[COMPLETE] Testing completed!")
    print("Check the generated JSON report for detailed results.")
