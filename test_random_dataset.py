"""
Test untuk memastikan network bisa train dengan baik
meskipun dataset dirandom/shuffle
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

def test_binary_with_shuffled_data():
    """Test binary classification dengan shuffled dataset"""
    print("=" * 70)
    print("TEST 1: Binary Classification - Shuffled Dataset")
    print("=" * 70)

    # Load binary example
    response = requests.post(f'{BASE_URL}/quick_start_binary')
    result = response.json()

    if not result['success']:
        print(f"[FAIL] Could not load network: {result.get('error')}")
        return None

    dataset = result['example_dataset']
    print("\nOriginal dataset order:")
    for i, line in enumerate(dataset.split('\n')[1:4]):
        print(f"  {i+1}. {line}")

    # Train with original order
    print("\n1. Training with ORIGINAL order...")
    train_data = {
        'dataset': dataset,
        'epochs': 200,
        'learning_rate': 1.0,
        'optimizer': 'gd',
        'loss_function': 'binary'
    }
    response = requests.post(f'{BASE_URL}/train', json=train_data)
    result_original = response.json()

    if result_original['success']:
        loss_original = result_original['final_loss']
        acc_original = result_original['accuracy']
        print(f"   Final Loss: {loss_original:.6f}")
        print(f"   Accuracy:   {acc_original*100:.2f}%")
    else:
        print(f"   [FAIL] {result_original.get('error')}")
        return None

    # Train with shuffled order (3 different shuffles)
    results = []
    for trial in range(3):
        print(f"\n{trial+2}. Training with SHUFFLED order (Trial {trial+1})...")

        # Shuffle dataset
        shuffled = shuffle_dataset(dataset)
        print("   Shuffled dataset (first 3):")
        for i, line in enumerate(shuffled.split('\n')[1:4]):
            print(f"     {i+1}. {line}")

        # Reload network for fresh start
        requests.post(f'{BASE_URL}/quick_start_binary')

        # Train
        train_data['dataset'] = shuffled
        response = requests.post(f'{BASE_URL}/train', json=train_data)
        result_shuffled = response.json()

        if result_shuffled['success']:
            loss_shuffled = result_shuffled['final_loss']
            acc_shuffled = result_shuffled['accuracy']
            print(f"   Final Loss: {loss_shuffled:.6f}")
            print(f"   Accuracy:   {acc_shuffled*100:.2f}%")
            results.append({
                'loss': loss_shuffled,
                'accuracy': acc_shuffled
            })
        else:
            print(f"   [FAIL] {result_shuffled.get('error')}")
            return None

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    losses = [loss_original] + [r['loss'] for r in results]
    accuracies = [acc_original] + [r['accuracy'] for r in results]

    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"\nFinal Loss:")
    print(f"  Original:  {loss_original:.6f}")
    for i, r in enumerate(results):
        print(f"  Shuffle {i+1}: {r['loss']:.6f}")
    print(f"  Average:   {avg_loss:.6f}")
    print(f"  Std Dev:   {std_loss:.6f}")

    print(f"\nAccuracy:")
    print(f"  Original:  {acc_original*100:.2f}%")
    for i, r in enumerate(results):
        print(f"  Shuffle {i+1}: {r['accuracy']*100:.2f}%")
    print(f"  Average:   {avg_acc*100:.2f}%")
    print(f"  Std Dev:   {std_acc*100:.2f}%")

    # Check if all training converged well
    all_good_loss = all(loss < 0.1 for loss in losses)
    all_good_acc = all(acc >= 0.9 for acc in accuracies)
    consistent = std_loss < 0.05  # Loss variation < 0.05

    print(f"\nResults:")
    if all_good_loss:
        print("  [OK] All runs achieved low loss (<0.1)")
    else:
        print("  [WARNING] Some runs have high loss")

    if all_good_acc:
        print("  [OK] All runs achieved high accuracy (>=90%)")
    else:
        print("  [WARNING] Some runs have low accuracy")

    if consistent:
        print("  [OK] Results are consistent (std < 0.05)")
    else:
        print("  [WARNING] Results vary significantly")

    return all_good_loss and all_good_acc and consistent


def test_multiclass_with_shuffled_data():
    """Test multiclass classification dengan shuffled dataset"""
    print("\n" + "=" * 70)
    print("TEST 2: Multiclass Classification - Shuffled Dataset")
    print("=" * 70)

    # Load multiclass example
    response = requests.post(f'{BASE_URL}/quick_start_multiclass')
    result = response.json()

    if not result['success']:
        print(f"[FAIL] Could not load network: {result.get('error')}")
        return None

    dataset = result['example_dataset']
    print("\nOriginal dataset order:")
    for i, line in enumerate(dataset.split('\n')[1:4]):
        print(f"  {i+1}. {line}")

    # Train with original order
    print("\n1. Training with ORIGINAL order...")
    train_data = {
        'dataset': dataset,
        'epochs': 200,
        'learning_rate': 1.0,
        'optimizer': 'gd',
        'loss_function': 'categorical'
    }
    response = requests.post(f'{BASE_URL}/train', json=train_data)
    result_original = response.json()

    if result_original['success']:
        loss_original = result_original['final_loss']
        acc_original = result_original['accuracy']
        print(f"   Final Loss: {loss_original:.6f}")
        print(f"   Accuracy:   {acc_original*100:.2f}%")
    else:
        print(f"   [FAIL] {result_original.get('error')}")
        return None

    # Train with shuffled order (3 different shuffles)
    results = []
    for trial in range(3):
        print(f"\n{trial+2}. Training with SHUFFLED order (Trial {trial+1})...")

        # Shuffle dataset
        shuffled = shuffle_dataset(dataset)
        print("   Shuffled dataset (first 3):")
        for i, line in enumerate(shuffled.split('\n')[1:4]):
            print(f"     {i+1}. {line}")

        # Reload network for fresh start
        requests.post(f'{BASE_URL}/quick_start_multiclass')

        # Train
        train_data['dataset'] = shuffled
        response = requests.post(f'{BASE_URL}/train', json=train_data)
        result_shuffled = response.json()

        if result_shuffled['success']:
            loss_shuffled = result_shuffled['final_loss']
            acc_shuffled = result_shuffled['accuracy']
            print(f"   Final Loss: {loss_shuffled:.6f}")
            print(f"   Accuracy:   {acc_shuffled*100:.2f}%")
            results.append({
                'loss': loss_shuffled,
                'accuracy': acc_shuffled
            })
        else:
            print(f"   [FAIL] {result_shuffled.get('error')}")
            return None

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    losses = [loss_original] + [r['loss'] for r in results]
    accuracies = [acc_original] + [r['accuracy'] for r in results]

    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"\nFinal Loss:")
    print(f"  Original:  {loss_original:.6f}")
    for i, r in enumerate(results):
        print(f"  Shuffle {i+1}: {r['loss']:.6f}")
    print(f"  Average:   {avg_loss:.6f}")
    print(f"  Std Dev:   {std_loss:.6f}")

    print(f"\nAccuracy:")
    print(f"  Original:  {acc_original*100:.2f}%")
    for i, r in enumerate(results):
        print(f"  Shuffle {i+1}: {r['accuracy']*100:.2f}%")
    print(f"  Average:   {avg_acc*100:.2f}%")
    print(f"  Std Dev:   {std_acc*100:.2f}%")

    # Check if all training converged well
    all_good_loss = all(loss < 0.4 for loss in losses)  # More lenient for multiclass
    all_good_acc = all(acc >= 0.8 for acc in accuracies)  # At least 80%
    consistent = std_loss < 0.1  # Loss variation < 0.1

    print(f"\nResults:")
    if all_good_loss:
        print("  [OK] All runs achieved reasonable loss (<0.4)")
    else:
        print("  [WARNING] Some runs have high loss")

    if all_good_acc:
        print("  [OK] All runs achieved good accuracy (>=80%)")
    else:
        print("  [WARNING] Some runs have low accuracy")

    if consistent:
        print("  [OK] Results are consistent (std < 0.1)")
    else:
        print("  [WARNING] Results vary significantly")

    return all_good_loss and all_good_acc and consistent


def main():
    print("\n")
    print("*" * 70)
    print("ROBUSTNESS TEST: Training with Randomized/Shuffled Datasets")
    print("*" * 70)
    print("\nThis test verifies that neural networks can train effectively")
    print("even when the dataset order is randomized/shuffled.")
    print()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    try:
        # Test binary
        result_binary = test_binary_with_shuffled_data()

        # Test multiclass
        result_multiclass = test_multiclass_with_shuffled_data()

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        if result_binary is not None and result_multiclass is not None:
            if result_binary and result_multiclass:
                print("\n[SUCCESS] All tests passed!")
                print("  - Binary classification: ROBUST to shuffled data")
                print("  - Multiclass classification: ROBUST to shuffled data")
                print("\nConclusion: Networks can train effectively regardless of")
                print("            dataset order. Training is robust and consistent.")
            else:
                print("\n[PARTIAL SUCCESS] Some tests had issues:")
                if not result_binary:
                    print("  - Binary classification: Inconsistent results with shuffling")
                if not result_multiclass:
                    print("  - Multiclass classification: Inconsistent results with shuffling")
        else:
            print("\n[FAIL] Some tests could not complete")

        print("\n" + "=" * 70)

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to Flask server")
        print("Please make sure the server is running with: python app.py")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
