"""
Test SGD optimizer robustness dengan shuffled data
SGD lebih sensitive karena mini-batch updates
"""
import requests
import numpy as np
import random

BASE_URL = 'http://localhost:5000'

def shuffle_dataset(dataset_str):
    """Shuffle dataset CSV"""
    lines = dataset_str.strip().split('\n')
    header = lines[0]
    data_lines = lines[1:]
    random.shuffle(data_lines)
    return header + '\n' + '\n'.join(data_lines)

def test_sgd_binary():
    """Test SGD untuk binary classification"""
    print("=" * 70)
    print("TEST 1: Binary Classification - SGD with Shuffled Dataset")
    print("=" * 70)

    # Load network
    response = requests.post(f'{BASE_URL}/quick_start_binary')
    result = response.json()
    dataset = result['example_dataset']

    results = []

    # Run 5 trials with different shuffles
    for trial in range(5):
        print(f"\nTrial {trial+1}/5...")

        # Shuffle dataset
        shuffled = shuffle_dataset(dataset) if trial > 0 else dataset

        # Reload network
        requests.post(f'{BASE_URL}/quick_start_binary')

        # Train with SGD
        train_data = {
            'dataset': shuffled,
            'epochs': 300,
            'learning_rate': 0.5,
            'optimizer': 'sgd',
            'loss_function': 'binary',
            'batch_size': 2  # Mini-batch
        }
        response = requests.post(f'{BASE_URL}/train', json=train_data)
        result_train = response.json()

        if result_train['success']:
            loss = result_train['final_loss']
            acc = result_train['accuracy']
            print(f"  Final Loss: {loss:.6f}, Accuracy: {acc*100:.2f}%")
            results.append({'loss': loss, 'accuracy': acc})
        else:
            print(f"  [FAIL] {result_train.get('error')}")
            return None

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    losses = [r['loss'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    print(f"\nFinal Loss across 5 trials:")
    for i, loss in enumerate(losses):
        print(f"  Trial {i+1}: {loss:.6f}")
    print(f"  Average: {np.mean(losses):.6f}")
    print(f"  Std Dev: {np.std(losses):.6f}")
    print(f"  Min:     {np.min(losses):.6f}")
    print(f"  Max:     {np.max(losses):.6f}")

    print(f"\nAccuracy across 5 trials:")
    for i, acc in enumerate(accuracies):
        print(f"  Trial {i+1}: {acc*100:.2f}%")
    print(f"  Average: {np.mean(accuracies)*100:.2f}%")
    print(f"  Std Dev: {np.std(accuracies)*100:.2f}%")

    # Check robustness
    all_good = all(loss < 0.15 and acc >= 0.8 for loss, acc in zip(losses, accuracies))
    consistent = np.std(losses) < 0.1

    print(f"\nResults:")
    if all_good:
        print("  [OK] All trials converged well")
    else:
        print("  [WARNING] Some trials struggled")

    if consistent:
        print("  [OK] Consistent across shuffles (std < 0.1)")
    else:
        print("  [WARNING] High variance across shuffles")

    return all_good and consistent

def test_sgd_multiclass():
    """Test SGD untuk multiclass classification"""
    print("\n" + "=" * 70)
    print("TEST 2: Multiclass Classification - SGD with Shuffled Dataset")
    print("=" * 70)

    # Load network
    response = requests.post(f'{BASE_URL}/quick_start_multiclass')
    result = response.json()
    dataset = result['example_dataset']

    results = []

    # Run 5 trials with different shuffles
    for trial in range(5):
        print(f"\nTrial {trial+1}/5...")

        # Shuffle dataset
        shuffled = shuffle_dataset(dataset) if trial > 0 else dataset

        # Reload network
        requests.post(f'{BASE_URL}/quick_start_multiclass')

        # Train with SGD
        train_data = {
            'dataset': shuffled,
            'epochs': 300,
            'learning_rate': 0.5,
            'optimizer': 'sgd',
            'loss_function': 'categorical',
            'batch_size': 2  # Mini-batch
        }
        response = requests.post(f'{BASE_URL}/train', json=train_data)
        result_train = response.json()

        if result_train['success']:
            loss = result_train['final_loss']
            acc = result_train['accuracy']
            print(f"  Final Loss: {loss:.6f}, Accuracy: {acc*100:.2f}%")
            results.append({'loss': loss, 'accuracy': acc})
        else:
            print(f"  [FAIL] {result_train.get('error')}")
            return None

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    losses = [r['loss'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    print(f"\nFinal Loss across 5 trials:")
    for i, loss in enumerate(losses):
        print(f"  Trial {i+1}: {loss:.6f}")
    print(f"  Average: {np.mean(losses):.6f}")
    print(f"  Std Dev: {np.std(losses):.6f}")
    print(f"  Min:     {np.min(losses):.6f}")
    print(f"  Max:     {np.max(losses):.6f}")

    print(f"\nAccuracy across 5 trials:")
    for i, acc in enumerate(accuracies):
        print(f"  Trial {i+1}: {acc*100:.2f}%")
    print(f"  Average: {np.mean(accuracies)*100:.2f}%")
    print(f"  Std Dev: {np.std(accuracies)*100:.2f}%")

    # Check robustness
    all_good = all(loss < 0.5 and acc >= 0.7 for loss, acc in zip(losses, accuracies))
    consistent = np.std(losses) < 0.15

    print(f"\nResults:")
    if all_good:
        print("  [OK] All trials converged well")
    else:
        print("  [WARNING] Some trials struggled")

    if consistent:
        print("  [OK] Reasonably consistent (std < 0.15)")
    else:
        print("  [WARNING] High variance across shuffles")

    return all_good and consistent

def main():
    print("\n")
    print("*" * 70)
    print("SGD ROBUSTNESS TEST: Mini-Batch Training with Shuffled Data")
    print("*" * 70)
    print("\nSGD is more sensitive to data order due to mini-batch updates.")
    print("This test verifies SGD can still train effectively with shuffled data.")
    print()

    random.seed(123)
    np.random.seed(123)

    try:
        result_binary = test_sgd_binary()
        result_multiclass = test_sgd_multiclass()

        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        if result_binary is not None and result_multiclass is not None:
            if result_binary and result_multiclass:
                print("\n[SUCCESS] SGD is robust to shuffled data!")
                print("  - Binary classification: Consistent with SGD")
                print("  - Multiclass classification: Consistent with SGD")
                print("\nConclusion: Both GD and SGD can train effectively")
                print("            regardless of dataset order.")
            else:
                print("\n[PARTIAL SUCCESS] Some variability detected:")
                if not result_binary:
                    print("  - Binary SGD: Higher variance with shuffling")
                if not result_multiclass:
                    print("  - Multiclass SGD: Higher variance with shuffling")
                print("\nNote: Some variance is expected with SGD due to")
                print("      stochastic nature of mini-batch training.")
        else:
            print("\n[FAIL] Some tests could not complete")

        print("\n" + "=" * 70)

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to Flask server")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
