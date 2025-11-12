"""
Contoh penggunaan API Training melalui Flask
Script ini menunjukkan cara memanggil endpoint /train
"""

import requests
import json

# Base URL untuk Flask app
BASE_URL = "http://localhost:5000"

print("="*70)
print(" " * 15 + "CONTOH PENGGUNAAN API TRAINING")
print("="*70)

# Step 1: Build network terlebih dahulu
print("\n[1] Membangun Neural Network...")
print("-"*70)

network_config = {
    "layers": [
        {"num_nodes": 2, "activation": "linear"},    # Input
        {"num_nodes": 2, "activation": "sigmoid"},   # Hidden
        {"num_nodes": 1, "activation": "sigmoid"}    # Output
    ],
    "connections": [
        {},  # Layer 0 (input) tidak punya connections
        {
            "layer_idx": 1,
            "connections": [[0, 1], [0, 1]],
            "weights": [[0.5, -0.3], [-0.4, 0.6]],
            "biases": [0.1, -0.2]
        },
        {
            "layer_idx": 2,
            "connections": [[0, 1]],
            "weights": [[0.8, -0.5]],
            "biases": [0.2]
        }
    ]
}

response = requests.post(f"{BASE_URL}/build_network", json=network_config)
result = response.json()

if result['success']:
    print("✓ Network berhasil dibuat!")
    print("\n" + result['summary'])
else:
    print("✗ Error:", result['error'])
    exit(1)

# Step 2: Prepare dataset
print("\n[2] Menyiapkan Dataset...")
print("-"*70)

# Dataset AND gate dalam format CSV
dataset_csv = """x1,x2,y
0,0,0
0,1,0
1,0,0
1,1,1"""

print("Dataset (AND Gate):")
print(dataset_csv)

# Step 3: Train the network
print("\n[3] Melatih Neural Network...")
print("-"*70)

training_config = {
    "dataset": dataset_csv,
    "epochs": 1000,
    "learning_rate": 0.5,
    "optimizer": "gd",          # "gd" atau "sgd"
    "loss_function": "mse",     # "mse" atau "binary"
    "batch_size": None          # None = full batch, atau bisa set integer untuk mini-batch
}

print("Parameter Training:")
print(f"  - Optimizer: {training_config['optimizer'].upper()}")
print(f"  - Learning Rate: {training_config['learning_rate']}")
print(f"  - Epochs: {training_config['epochs']}")
print(f"  - Loss Function: {training_config['loss_function'].upper()}")
print(f"  - Batch Size: {'Full Batch' if training_config['batch_size'] is None else training_config['batch_size']}")

print("\nMemulai training...")
response = requests.post(f"{BASE_URL}/train", json=training_config)
result = response.json()

if not result['success']:
    print("✗ Error:", result['error'])
    if 'traceback' in result:
        print("\nTraceback:")
        print(result['traceback'])
    exit(1)

print("✓ Training selesai!")

# Step 4: Display results
print("\n[4] HASIL TRAINING")
print("="*70)

# Loss history
print("\n" + "-"*70)
print("Loss Per Epoch (setiap 100 epoch)")
print("-"*70)
history = result['history']
print(f"{'Epoch':<10} {'Loss':<15}")
print("-"*30)
for i in range(0, len(history['epochs']), 100):
    print(f"{history['epochs'][i]:<10} {history['loss'][i]:<15.6f}")

if len(history['epochs']) > 0:
    last_idx = len(history['epochs']) - 1
    print(f"{history['epochs'][last_idx]:<10} {history['loss'][last_idx]:<15.6f}")

# Final metrics
print("\n" + "-"*70)
print("Metrik Akhir")
print("-"*70)
print(f"Final Loss: {result['final_loss']:.6f}")
print(f"Accuracy: {result['accuracy'] * 100:.2f}%")

# Updated weights and biases
print("\n" + "-"*70)
print("Bobot & Bias Baru (Setelah Training)")
print("-"*70)

for layer_name, weights in result['updated_weights'].items():
    layer_idx = int(layer_name.split('_')[1])
    print(f"\n{layer_name.upper()}:")
    for node_idx, node_weights in enumerate(weights):
        bias = result['updated_biases'][layer_name][node_idx]
        print(f"  Node {node_idx}:")
        print(f"    Weights: {[f'{w:.4f}' for w in node_weights]}")
        print(f"    Bias: {bias:.4f}")

# Predictions
print("\n" + "-"*70)
print("Prediksi Setelah Training")
print("-"*70)
print(f"{'y_true':<10} {'y_pred':<12} {'y_class':<10} {'Status'}")
print("-"*45)

predictions = result['predictions']
for i in range(len(predictions['y_true'])):
    y_true = predictions['y_true'][i]
    y_pred = predictions['y_pred'][i]
    y_class = predictions['y_pred_classes'][i]
    status = "BENAR ✓" if y_class == y_true else "SALAH ✗"
    print(f"{y_true:<10} {y_pred:<12.4f} {y_class:<10} {status}")

print("\n" + "="*70)
print("DEMO API TRAINING SELESAI")
print("="*70)

# Example dengan SGD
print("\n\n" + "="*70)
print(" " * 10 + "BONUS: Perbandingan dengan SGD Optimizer")
print("="*70)

# Rebuild network dengan weights yang sama
response = requests.post(f"{BASE_URL}/build_network", json=network_config)

# Train dengan SGD
training_config_sgd = {
    "dataset": dataset_csv,
    "epochs": 1000,
    "learning_rate": 0.5,
    "optimizer": "sgd",
    "loss_function": "mse",
    "batch_size": 2  # Mini-batch SGD
}

print("\nTraining dengan SGD...")
response = requests.post(f"{BASE_URL}/train", json=training_config_sgd)
result_sgd = response.json()

if result_sgd['success']:
    print("✓ Training dengan SGD selesai!")
    print(f"\nHasil SGD:")
    print(f"  Final Loss: {result_sgd['final_loss']:.6f}")
    print(f"  Accuracy: {result_sgd['accuracy'] * 100:.2f}%")

    print(f"\nPerbandingan GD vs SGD:")
    print(f"  GD  - Loss: {result['final_loss']:.6f}, Accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"  SGD - Loss: {result_sgd['final_loss']:.6f}, Accuracy: {result_sgd['accuracy'] * 100:.2f}%")
else:
    print("✗ Error:", result_sgd['error'])

print("\n" + "="*70)
