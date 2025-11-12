"""
Test custom network 2-3-1 through API
"""
import requests
import json

BASE_URL = 'http://localhost:5000'

print("=" * 70)
print("TEST: Custom Network 2-3-1 via API")
print("=" * 70)

# Step 1: Build network
print("\n1. Building 2-3-1 network via API...")

build_data = {
    'layers': [
        {'num_nodes': 2, 'activation': 'linear'},
        {'num_nodes': 3, 'activation': 'sigmoid'},
        {'num_nodes': 1, 'activation': 'sigmoid'}
    ],
    'connections': {
        '1': {
            'connections': [[0, 1], [0, 1], [0, 1]],
            'weights': [[0.5, 0.3], [-0.4, 0.6], [0.2, -0.5]],
            'biases': [0.1, -0.2, 0.3]
        },
        '2': {
            'connections': [[0, 1, 2]],
            'weights': [[0.7, -0.4, 0.5]],
            'biases': [0.2]
        }
    }
}

response = requests.post(f'{BASE_URL}/build_network', json=build_data)
result = response.json()

if result['success']:
    print("   [OK] Network built successfully")
else:
    print(f"   [FAIL] {result.get('error')}")
    if 'traceback' in result:
        print("\n   Traceback:")
        print(result['traceback'])
    exit(1)

# Step 2: Make prediction
print("\n2. Making prediction...")

dataset = "x1,x2,y\n0.5,0.8,1\n0.3,0.6,0\n0.7,0.9,1"

pred_data = {
    'dataset': dataset,
    'loss_function': 'binary',
    'threshold': 0.5
}

response = requests.post(f'{BASE_URL}/predict', json=pred_data)
result = response.json()

if result['success']:
    print(f"   [OK] Prediction successful")
    print(f"   Loss: {result['loss']:.6f}")
    print(f"   Number of predictions: {len(result['results'])}")
    for i, res in enumerate(result['results'][:3]):
        print(f"   Sample {i}: {res}")
else:
    print(f"   [FAIL] {result.get('error')}")
    if 'traceback' in result:
        print("\n   Traceback:")
        print(result['traceback'])

# Step 3: Train network
print("\n3. Training network...")

train_data = {
    'dataset': dataset,
    'epochs': 50,
    'learning_rate': 0.5,
    'optimizer': 'gd',
    'loss_function': 'binary'
}

response = requests.post(f'{BASE_URL}/train', json=train_data)
result = response.json()

if result['success']:
    print(f"   [OK] Training successful")
    print(f"   Initial Loss: {result['history']['loss'][0]:.6f}")
    print(f"   Final Loss: {result['final_loss']:.6f}")
    print(f"   Accuracy: {result['accuracy']*100:.2f}%")
else:
    print(f"   [FAIL] {result.get('error')}")
    if 'traceback' in result:
        print("\n   Traceback:")
        print(result['traceback'])

print("\n" + "=" * 70)
print("CONCLUSION: If all tests passed, network 2-3-1 works correctly!")
print("=" * 70)
