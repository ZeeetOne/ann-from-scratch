"""
Test multiclass classification dengan categorical cross-entropy
Untuk verify bahwa loss berubah dengan benar setelah fix
"""
import requests
import json

BASE_URL = 'http://localhost:5000'

def test_multiclass_training():
    print("=" * 70)
    print("TEST: Multiclass Classification Training")
    print("=" * 70)

    # Step 1: Load example multiclass network
    print("\n1. Loading multiclass example network...")
    response = requests.post(f'{BASE_URL}/quick_start_multiclass')
    result = response.json()

    if result['success']:
        print("   [OK] Network loaded successfully")
        if 'architecture' in result:
            print(f"   Architecture: {result['architecture']}")
    else:
        print(f"   [FAIL] Failed: {result.get('error')}")
        return

    # Step 2: Load dataset
    print("\n2. Loading multiclass dataset...")
    dataset = """suhu,tekanan,kelembapan,hujan,cerah
25,1010,85,1,0
30,1015,45,0,1
22,1005,90,1,0
28,1012,50,0,1
20,1000,95,1,0
32,1018,40,0,1
24,1008,80,1,0
29,1014,48,0,1
21,1003,88,1,0
31,1016,42,0,1"""

    print("   [OK] Dataset loaded (10 samples, 3 features, 2 classes)")

    # Step 3: Make predictions BEFORE training
    print("\n3. Making predictions BEFORE training...")
    pred_data = {
        'dataset': dataset,
        'loss_function': 'categorical',
        'threshold': 0.5
    }
    response = requests.post(f'{BASE_URL}/predict', json=pred_data)
    result_before = response.json()

    if result_before['success']:
        loss_before = result_before['loss']
        print(f"   Loss BEFORE training: {loss_before:.6f}")

        # Print per-sample losses
        if 'predictions' in result_before and len(result_before['predictions']) > 0:
            print("\n   Per-sample losses BEFORE training:")
            for i, pred in enumerate(result_before['predictions'][:3]):  # Show first 3
                print(f"   Sample {i}: y_true={pred['y_true']}, y_pred={[f'{p:.4f}' for p in pred['y_pred']]}, loss={pred.get('loss', 'N/A')}")
    else:
        print(f"   [FAIL] Failed: {result_before.get('error')}")
        return

    # Step 4: Single weight update
    print("\n4. Performing single weight update...")
    update_data = {
        'dataset': dataset,
        'loss_function': 'categorical',
        'learning_rate': 0.5,
        'optimizer': 'gd'
    }
    response = requests.post(f'{BASE_URL}/update_weights', json=update_data)
    result_update = response.json()

    if result_update['success']:
        loss_after_update = result_update['loss_after']
        loss_change = result_update['loss_before'] - loss_after_update
        print(f"   Loss BEFORE update: {result_update['loss_before']:.6f}")
        print(f"   Loss AFTER update:  {loss_after_update:.6f}")
        print(f"   Loss change:        {loss_change:.6f}")

        if abs(loss_change) < 1e-10:
            print("   [WARNING] Loss DID NOT CHANGE! (bug detected)")
        else:
            print("   [OK] Loss changed successfully")
    else:
        print(f"   [FAIL] Failed: {result_update.get('error')}")
        import traceback
        if 'traceback' in result_update:
            print("\n   Traceback:")
            print(result_update['traceback'])
        return

    # Step 5: Full training
    print("\n5. Training network (100 epochs)...")
    train_data = {
        'dataset': dataset,
        'epochs': 100,
        'learning_rate': 0.5,
        'optimizer': 'gd',
        'loss_function': 'categorical'
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

        # Show loss progression
        if 'history' in result_train and 'loss' in result_train['history']:
            print("\n   Loss progression (every 10 epochs):")
            for i in range(0, len(result_train['history']['loss']), 10):
                epoch = result_train['history']['epochs'][i] if 'epochs' in result_train['history'] else i+1
                loss = result_train['history']['loss'][i]
                print(f"   Epoch {epoch:3d}: {loss:.6f}")

        if abs(initial_loss - final_loss) < 1e-10:
            print("\n   [FAIL] Loss did not change during training!")
        else:
            print("\n   [OK] Training completed successfully")

        # Show some predictions
        print("\n   Predictions AFTER training (first 3 samples):")
        for i, pred in enumerate(result_train['predictions'][:3]):
            print(f"   Sample {i}: y_true={pred['y_true']}, y_pred={[f'{p:.4f}' for p in pred['y_pred']]}, class={pred['y_pred_classes']}")
    else:
        print(f"   [FAIL] Failed: {result_train.get('error')}")
        if 'traceback' in result_train:
            print("\n   Traceback:")
            print(result_train['traceback'])
        return

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    try:
        test_multiclass_training()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Flask server at http://localhost:5000")
        print("Please make sure the server is running with: python app.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
