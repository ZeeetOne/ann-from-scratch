"""
Test binary classification dengan normalized dataset
"""
import requests
import json

BASE_URL = 'http://localhost:5000'

def test_binary_classification():
    print("=" * 70)
    print("TEST: Binary Classification dengan Normalized Dataset")
    print("=" * 70)

    # Load binary example
    print("\n1. Loading binary example...")
    response = requests.post(f'{BASE_URL}/quick_start_binary')
    result = response.json()

    if result['success']:
        print("   [OK] Network loaded")
        dataset = result['example_dataset']
        print("\n   Example dataset (first 4 lines):")
        for line in dataset.split('\n')[:5]:
            print(f"   {line}")
    else:
        print(f"   [FAIL] {result.get('error')}")
        return

    # Predict before training
    print("\n2. Predictions BEFORE training...")
    pred_data = {
        'dataset': dataset,
        'loss_function': 'binary',
        'threshold': 0.5
    }
    response = requests.post(f'{BASE_URL}/predict', json=pred_data)
    result_before = response.json()

    if result_before['success']:
        print(f"   Loss: {result_before['loss']:.6f}")
    else:
        print(f"   [FAIL] {result_before.get('error')}")
        return

    # Train
    print("\n3. Training network (200 epochs, lr=1.0)...")
    train_data = {
        'dataset': dataset,
        'epochs': 200,
        'learning_rate': 1.0,
        'optimizer': 'gd',
        'loss_function': 'binary'
    }
    response = requests.post(f'{BASE_URL}/train', json=train_data)
    result_train = response.json()

    if result_train['success']:
        initial_loss = result_train['history']['loss'][0]
        final_loss = result_train['final_loss']
        improvement = (initial_loss - final_loss) / initial_loss * 100

        print(f"   Initial Loss: {initial_loss:.6f}")
        print(f"   Final Loss:   {final_loss:.6f}")
        print(f"   Improvement:  {improvement:.2f}%")
        print(f"   Accuracy:     {result_train['accuracy']*100:.2f}%")

        # Show predictions after training
        print("\n   Predictions AFTER training:")
        for i in range(min(10, len(result_train['predictions']))):
            pred = result_train['predictions'][i]
            y_true = pred['y_true']
            y_pred = pred['y_pred']
            y_class = pred['y_pred_classes']

            # Check if correct
            correct = "[OK]" if y_class == y_true else "[FAIL]"

            # Format prediction value
            pred_val = f"{y_pred:.3f}" if isinstance(y_pred, float) else str(y_pred)

            print(f"   Sample {i}: y_true={y_true}, y_pred={pred_val}, class={y_class} {correct}")

        print("\n   [OK] Training successful!")

        # Check if predictions are diverse (not all 0.5)
        diverse = False
        for pred in result_train['predictions']:
            pred_val = pred['y_pred']
            if isinstance(pred_val, float):
                if abs(pred_val - 0.5) > 0.1:
                    diverse = True
                    break
            elif isinstance(pred_val, list) and len(pred_val) > 0:
                if abs(pred_val[0] - 0.5) > 0.1:
                    diverse = True
                    break

        if diverse:
            print("   [OK] Predictions are DIVERSE (not stuck at 0.5)")
        else:
            print("   [WARNING] Predictions still stuck near 0.5")

        # Check accuracy
        if result_train['accuracy'] >= 0.9:
            print("   [OK] High accuracy achieved (>=90%)")
        elif result_train['accuracy'] >= 0.7:
            print("   [WARN] Moderate accuracy (70-90%)")
        else:
            print("   [WARN] Low accuracy (<70%)")

    else:
        print(f"   [FAIL] {result_train.get('error')}")
        if 'traceback' in result_train:
            print("\n   Traceback:")
            print(result_train['traceback'])

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    try:
        test_binary_classification()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Flask server")
        print("Please make sure the server is running")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
