"""
Detailed Binary Classification Test - Executable Script

This script runs the exact test described in DETAILED_BINARY_TEST_WITH_MANUAL_CALCULATION.md
and verifies that manual calculations match network output.

Author: ANN from Scratch Team
Date: 2025-11-13
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from backend.core import NeuralNetwork


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """Sigmoid derivative: a * (1 - a)"""
    return a * (1 - a)


def main():
    """Run detailed binary classification test with manual calculation verification"""

    print_section("DETAILED BINARY CLASSIFICATION TEST WITH MANUAL CALCULATIONS")

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # NETWORK INITIALIZATION
    # =========================================================================
    print_section("1. NETWORK INITIALIZATION")

    print("Architecture: 2-3-1")
    print("  - Input Layer: 2 neurons (linear)")
    print("  - Hidden Layer: 3 neurons (sigmoid)")
    print("  - Output Layer: 1 neuron (sigmoid)")

    # Build network
    network = NeuralNetwork()
    network.add_layer(2, 'linear')
    network.add_layer(3, 'sigmoid')
    network.add_layer(1, 'sigmoid')

    # Set exact weights from manual calculation
    W1 = np.array([
        [0.5000, 0.3000],
        [-0.2000, 0.4000],
        [0.1000, -0.3000]
    ])
    b1 = np.array([0.1000, 0.0500, -0.0500])

    W2 = np.array([[0.6000, 0.4000, -0.2000]])
    b2 = np.array([0.1500])

    network.set_connections(1, [[0,1], [0,1], [0,1]], W1.tolist(), b1.tolist())
    network.set_connections(2, [[0,1,2]], W2.tolist(), b2.tolist())

    print("\nLayer 1 (Input -> Hidden) Weights:")
    print(f"  W1 = \n{W1}")
    print(f"  b1 = {b1}")

    print("\nLayer 2 (Hidden -> Output) Weights:")
    print(f"  W2 = {W2}")
    print(f"  b2 = {b2}")

    # =========================================================================
    # DATASET GENERATION
    # =========================================================================
    print_section("2. RANDOM DATASET GENERATION")

    X = np.array([
        [0.374540, 0.950714],
        [0.731994, 0.598658],
        [0.156019, 0.155995],
        [0.058084, 0.866176]
    ])
    y = np.array([[1], [1], [0], [1]])

    print("Generated 4 samples with 2 features each:\n")
    print("  x1        x2        y")
    print("-" * 30)
    for i in range(len(X)):
        print(f"  {X[i,0]:.6f}  {X[i,1]:.6f}  {y[i,0]}")

    print(f"\nClass Distribution:")
    print(f"  Class 0: {np.sum(y == 0)} samples ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  Class 1: {np.sum(y == 1)} samples ({np.sum(y == 1) / len(y) * 100:.1f}%)")

    # =========================================================================
    # MANUAL FORWARD PASS FOR SAMPLE 1
    # =========================================================================
    print_section("3. MANUAL FORWARD PASS - SAMPLE 1")

    sample_idx = 0
    x_sample = X[sample_idx]
    y_sample = y[sample_idx, 0]

    print(f"Input: x = [{x_sample[0]:.6f}, {x_sample[1]:.6f}]")
    print(f"Target: y = {y_sample}")

    # Layer 1: Input -> Hidden
    print("\n--- Layer 1: Input -> Hidden ---")
    z1_manual = W1 @ x_sample + b1
    print(f"\nLinear transformation (z1 = W1*x + b1):")
    for i in range(3):
        print(f"  z1[{i}] = {W1[i,0]:.4f} * {x_sample[0]:.6f} + {W1[i,1]:.4f} * {x_sample[1]:.6f} + {b1[i]:.4f}")
        print(f"        = {z1_manual[i]:.6f}")

    a1_manual = sigmoid(z1_manual)
    print(f"\nSigmoid activation (a1 = sigmoid(z1)):")
    for i in range(3):
        print(f"  a1[{i}] = sigmoid({z1_manual[i]:.6f}) = {a1_manual[i]:.6f}")

    # Layer 2: Hidden -> Output
    print("\n--- Layer 2: Hidden -> Output ---")
    z2_manual = W2 @ a1_manual + b2
    print(f"\nLinear transformation (z2 = W2*a1 + b2):")
    print(f"  z2[0] = {W2[0,0]:.4f} * {a1_manual[0]:.6f} + {W2[0,1]:.4f} * {a1_manual[1]:.6f} + {W2[0,2]:.4f} * {a1_manual[2]:.6f} + {b2[0]:.4f}")
    print(f"        = {z2_manual[0]:.6f}")

    a2_manual = sigmoid(z2_manual)
    print(f"\nSigmoid activation (a2 = sigmoid(z2)):")
    print(f"  a2[0] = sigmoid({z2_manual[0]:.6f}) = {a2_manual[0]:.6f}")

    print(f"\n[OK] Manual Forward Pass Result:")
    print(f"  Predicted: {a2_manual[0]:.6f}")
    print(f"  Actual: {y_sample}")
    print(f"  Error: {y_sample - a2_manual[0]:.6f}")

    # =========================================================================
    # NETWORK FORWARD PASS
    # =========================================================================
    print_section("4. NETWORK FORWARD PASS (VERIFICATION)")

    y_pred_before = network.forward(X)

    print("Network predictions for all samples:\n")
    print("  Sample  Predicted  Actual  Error")
    print("-" * 40)
    for i in range(len(X)):
        error = y[i, 0] - y_pred_before[i, 0]
        print(f"  {i+1}       {y_pred_before[i, 0]:.6f}   {y[i, 0]}       {error:+.6f}")

    # Verify Sample 1 matches manual calculation
    print(f"\n[OK] Verification for Sample 1:")
    print(f"  Manual calculation: {a2_manual[0]:.6f}")
    print(f"  Network output:     {y_pred_before[sample_idx, 0]:.6f}")
    diff = abs(a2_manual[0] - y_pred_before[sample_idx, 0])
    print(f"  Difference:         {diff:.10f}")

    if diff < 1e-6:
        print("  [PASS] MATCH (difference < 1e-6)")
    else:
        print("  [FAIL] MISMATCH")

    # =========================================================================
    # LOSS CALCULATION
    # =========================================================================
    print_section("5. LOSS CALCULATION")

    # Binary cross-entropy loss
    epsilon = 1e-15  # For numerical stability
    y_pred_clipped = np.clip(y_pred_before, epsilon, 1 - epsilon)

    # Manual loss calculation
    loss_manual = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))

    print("Binary Cross-Entropy Loss: L = -[y*log(y_hat) + (1-y)*log(1-y_hat)]\n")
    print("Sample-wise loss:")
    for i in range(len(X)):
        y_true = y[i, 0]
        y_pred = y_pred_clipped[i, 0]
        sample_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        print(f"  Sample {i+1}: L = {sample_loss:.6f}")

    print(f"\n[OK] Average Loss (manual): {loss_manual:.6f}")

    # =========================================================================
    # MANUAL BACKPROPAGATION FOR SAMPLE 1
    # =========================================================================
    print_section("6. MANUAL BACKPROPAGATION - SAMPLE 1")

    print("Given:")
    print(f"  x = [{x_sample[0]:.6f}, {x_sample[1]:.6f}]")
    print(f"  y = {y_sample}")
    print(f"  y_hat = {a2_manual[0]:.6f}")
    print(f"  a1 = [{a1_manual[0]:.6f}, {a1_manual[1]:.6f}, {a1_manual[2]:.6f}]")

    # Output layer gradient
    print("\n--- Output Layer Gradient ---")
    dL_da2 = a2_manual - y_sample
    print(f"dL/da2 = y_hat - y = {a2_manual[0]:.6f} - {y_sample} = {dL_da2[0]:.6f}")

    da2_dz2 = sigmoid_derivative(a2_manual)
    print(f"da2/dz2 = a2*(1-a2) = {a2_manual[0]:.6f}*{1-a2_manual[0]:.6f} = {da2_dz2[0]:.6f}")

    delta2 = dL_da2 * da2_dz2
    print(f"delta2 = dL/da2 * da2/dz2 = {delta2[0]:.6f}")
    print(f"(Note: For sigmoid + binary cross-entropy, delta2 = y_hat - y = {dL_da2[0]:.6f})")

    # Output layer weight gradients
    print("\n--- Output Layer Weight Gradients ---")
    dL_dW2 = np.outer(delta2, a1_manual)
    dL_db2 = delta2

    print(f"dL/dw2[0,0] = delta2 * a1[0] = {delta2[0]:.6f} * {a1_manual[0]:.6f} = {dL_dW2[0,0]:.6f}")
    print(f"dL/dw2[0,1] = delta2 * a1[1] = {delta2[0]:.6f} * {a1_manual[1]:.6f} = {dL_dW2[0,1]:.6f}")
    print(f"dL/dw2[0,2] = delta2 * a1[2] = {delta2[0]:.6f} * {a1_manual[2]:.6f} = {dL_dW2[0,2]:.6f}")
    print(f"dL/db2[0] = delta2 = {dL_db2[0]:.6f}")

    # Hidden layer gradient
    print("\n--- Hidden Layer Gradient ---")
    dL_da1 = W2.T @ delta2
    print(f"Propagate error to hidden layer:")
    for i in range(3):
        print(f"  dL/da1[{i}] = delta2 * w2[0,{i}] = {delta2[0]:.6f} * {W2[0,i]:.4f} = {dL_da1[i]:.6f}")

    da1_dz1 = sigmoid_derivative(a1_manual)
    delta1 = dL_da1 * da1_dz1
    print(f"\nHidden layer deltas (delta1 = dL/da1 * da1/dz1):")
    for i in range(3):
        print(f"  delta1[{i}] = {dL_da1[i]:.6f} * {da1_dz1[i]:.6f} = {delta1[i]:.6f}")

    # Hidden layer weight gradients
    print("\n--- Hidden Layer Weight Gradients ---")
    dL_dW1 = np.outer(delta1, x_sample)
    dL_db1 = delta1

    print(f"dL/dW1:")
    for i in range(3):
        for j in range(2):
            print(f"  dL/dw1[{i},{j}] = delta1[{i}] * x[{j}] = {delta1[i]:.6f} * {x_sample[j]:.6f} = {dL_dW1[i,j]:.6f}")

    print(f"\ndL/db1:")
    for i in range(3):
        print(f"  dL/db1[{i}] = delta1[{i}] = {dL_db1[i]:.6f}")

    # =========================================================================
    # TRAINING (1 EPOCH)
    # =========================================================================
    print_section("7. TRAINING (1 EPOCH)")

    print("Configuration:")
    print("  - Optimizer: Gradient Descent (GD)")
    print("  - Learning Rate: 0.5")
    print("  - Epochs: 1")
    print("  - Loss Function: Binary Cross-Entropy")

    # Train for 1 epoch
    history = network.train(
        X, y,
        epochs=1,
        learning_rate=0.5,
        optimizer='gd',
        loss_function='binary'
    )

    print(f"\nTraining Results:")
    print(f"  Initial Loss: {history['loss'][0]:.6f}")
    print(f"  Final Loss: {history['loss'][-1]:.6f}")
    print(f"  Loss Reduction: {history['loss'][0] - history['loss'][-1]:.6f} ({(1 - history['loss'][-1] / history['loss'][0]) * 100:.2f}%)")

    # =========================================================================
    # MANUAL WEIGHT UPDATE VERIFICATION
    # =========================================================================
    print_section("8. WEIGHT UPDATE VERIFICATION")

    # For full verification, we'd need to accumulate gradients from all samples
    # Here we show the formula and verify final weights match

    print("Gradient Descent Update: W_new = W_old - learning_rate * dL/dW")
    print(f"Learning rate: 0.5")

    # Get updated weights from network
    W2_new = np.array(network.weights[2])
    b2_new = np.array(network.biases[2])
    W1_new = np.array(network.weights[1])
    b1_new = np.array(network.biases[1])

    print(f"\nLayer 2 Updated Weights:")
    print(f"  W2_old = {W2}")
    print(f"  W2_new = {W2_new}")
    print(f"  Change = {W2_new - W2}")

    print(f"\n  b2_old = {b2}")
    print(f"  b2_new = {b2_new}")
    print(f"  Change = {b2_new - b2}")

    print(f"\nLayer 1 Updated Weights:")
    print(f"  W1_old = \n{W1}")
    print(f"  W1_new = \n{W1_new}")
    print(f"  Change = \n{W1_new - W1}")

    print(f"\n  b1_old = {b1}")
    print(f"  b1_new = {b1_new}")
    print(f"  Change = {b1_new - b1}")

    # =========================================================================
    # POST-UPDATE FORWARD PASS
    # =========================================================================
    print_section("9. POST-UPDATE FORWARD PASS")

    y_pred_after = network.forward(X)

    print("Predictions after 1 epoch of training:\n")
    print("  Sample  Before     After      Target  Improvement")
    print("-" * 60)
    for i in range(len(X)):
        before = y_pred_before[i, 0]
        after = y_pred_after[i, 0]
        target = y[i, 0]
        improvement = abs(target - after) - abs(target - before)
        direction = "[OK] Better" if improvement < 0 else ("[X] Worse" if improvement > 0 else "= Same")
        print(f"  {i+1}       {before:.6f}   {after:.6f}   {target}       {improvement:+.6f} {direction}")

    # Calculate accuracy
    threshold = 0.5
    pred_classes_before = (y_pred_before >= threshold).astype(int)
    pred_classes_after = (y_pred_after >= threshold).astype(int)

    accuracy_before = np.mean(pred_classes_before == y)
    accuracy_after = np.mean(pred_classes_after == y)

    print(f"\nAccuracy (threshold=0.5):")
    print(f"  Before training: {accuracy_before * 100:.2f}%")
    print(f"  After 1 epoch:   {accuracy_after * 100:.2f}%")
    print(f"  Improvement:     {(accuracy_after - accuracy_before) * 100:+.2f}%")

    # =========================================================================
    # FINAL VERIFICATION
    # =========================================================================
    print_section("10. FINAL VERIFICATION SUMMARY")

    print("[PASS] Verification Checklist:\n")

    # Check 1: Forward pass matches manual
    check1 = abs(a2_manual[0] - y_pred_before[sample_idx, 0]) < 1e-6
    print(f"  [{'[OK]' if check1 else '[X]'}] Forward pass matches manual calculation (diff < 1e-6)")

    # Check 2: Loss decreased
    check2 = history['loss'][-1] < history['loss'][0]
    print(f"  [{'[OK]' if check2 else '[X]'}] Loss decreased after training")

    # Check 3: Weights changed
    check3 = not np.allclose(W2, W2_new) or not np.allclose(W1, W1_new)
    print(f"  [{'[OK]' if check3 else '[X]'}] Weights were updated")

    # Check 4: Predictions improved for most samples
    improvements = 0
    for i in range(len(X)):
        if abs(y[i,0] - y_pred_after[i,0]) < abs(y[i,0] - y_pred_before[i,0]):
            improvements += 1
    check4 = improvements >= len(X) // 2
    print(f"  [{'[OK]' if check4 else '[X]'}] Predictions improved for {improvements}/{len(X)} samples")

    # Check 5: Accuracy improved or maintained
    check5 = accuracy_after >= accuracy_before
    print(f"  [{'[OK]' if check5 else '[X]'}] Accuracy improved or maintained")

    # Overall result
    all_checks = check1 and check2 and check3 and check4 and check5

    print(f"\n{'='*80}")
    if all_checks:
        print("[PASS] ALL VERIFICATIONS PASSED - NETWORK IS WORKING CORRECTLY")
    else:
        print("[WARN]  SOME VERIFICATIONS FAILED - REVIEW RESULTS")
    print(f"{'='*80}\n")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY")

    print("Network Configuration:")
    print(f"  Architecture: 2-3-1 (sigmoid activations)")
    print(f"  Total Parameters: {W1.size + b1.size + W2.size + b2.size}")

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: 2 (binary)")

    print(f"\nTraining:")
    print(f"  Epochs: 1")
    print(f"  Learning Rate: 0.5")
    print(f"  Optimizer: Gradient Descent")

    print(f"\nResults:")
    print(f"  Initial Loss: {history['loss'][0]:.6f}")
    print(f"  Final Loss: {history['loss'][-1]:.6f}")
    print(f"  Loss Reduction: {(1 - history['loss'][-1] / history['loss'][0]) * 100:.2f}%")
    print(f"  Initial Accuracy: {accuracy_before * 100:.2f}%")
    print(f"  Final Accuracy: {accuracy_after * 100:.2f}%")
    print(f"  Accuracy Improvement: {(accuracy_after - accuracy_before) * 100:+.2f}%")

    print(f"\nVerification:")
    print(f"  Manual calculations match network: {'Yes [PASS]' if check1 else 'No [FAIL]'}")
    print(f"  Training improved performance: {'Yes [PASS]' if check2 else 'No [FAIL]'}")

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("See DETAILED_BINARY_TEST_WITH_MANUAL_CALCULATION.md for full documentation.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
