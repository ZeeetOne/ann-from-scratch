"""
Demo script untuk menunjukkan fitur training
Dengan visualisasi hasil yang mudah dibaca
"""

import numpy as np
from ann_core import NeuralNetwork
import pandas as pd

print("="*70)
print(" " * 20 + "ANN TRAINING DEMO")
print("="*70)

# Dataset sederhana: AND gate
print("\n" + "="*70)
print("Dataset: AND Gate")
print("="*70)

X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([[0], [0], [0], [1]])

print("\nInput -> Expected Output")
for i in range(len(X_and)):
    print(f"  {X_and[i]} -> {y_and[i][0]}")

# Buat network 2-2-1
print("\n" + "="*70)
print("Membangun Neural Network")
print("="*70)
print("Architecture: 2 (Input) -> 2 (Hidden, sigmoid) -> 1 (Output, sigmoid)")

nn = NeuralNetwork()
nn.add_layer(2, 'linear')  # Input
nn.add_layer(2, 'sigmoid')  # Hidden
nn.add_layer(1, 'sigmoid')  # Output

# Random initialization
np.random.seed(123)
w1 = [[np.random.randn() * 0.5 for _ in range(2)] for _ in range(2)]
b1 = [np.random.randn() * 0.1 for _ in range(2)]
w2 = [[np.random.randn() * 0.5 for _ in range(2)]]
b2 = [np.random.randn() * 0.1]

nn.set_connections(1, [[0, 1], [0, 1]], w1, b1)
nn.set_connections(2, [[0, 1]], w2, b2)

print("\n" + "-"*70)
print("Bobot Awal (Initial Weights & Biases)")
print("-"*70)
print("\nLayer 1 (Input -> Hidden):")
for i in range(2):
    print(f"  Node {i}: W={[f'{w:.3f}' for w in nn.weights[1][i]]}, b={nn.biases[1][i]:.3f}")
print("\nLayer 2 (Hidden -> Output):")
for i in range(1):
    print(f"  Node {i}: W={[f'{w:.3f}' for w in nn.weights[2][i]]}, b={nn.biases[2][i]:.3f}")

# Prediksi sebelum training
print("\n" + "-"*70)
print("Prediksi SEBELUM Training")
print("-"*70)
y_pred_before = nn.forward(X_and)
loss_before = nn.calculate_loss(y_and, y_pred_before, 'mse')

print(f"\n{'Input':<12} {'Target':<10} {'Prediksi':<10} {'Error':<10}")
print("-"*50)
for i in range(len(X_and)):
    error = abs(y_and[i][0] - y_pred_before[i][0])
    print(f"{str(X_and[i]):<12} {y_and[i][0]:<10} {y_pred_before[i][0]:<10.4f} {error:<10.4f}")
print(f"\nLoss (MSE): {loss_before:.6f}")

# Training
print("\n" + "="*70)
print("TRAINING DIMULAI")
print("="*70)
print("Parameter:")
print(f"  - Optimizer: Gradient Descent (GD)")
print(f"  - Learning Rate: 0.5")
print(f"  - Epochs: 1000")
print(f"  - Loss Function: Mean Squared Error (MSE)")

history = nn.train(
    X_and, y_and,
    epochs=1000,
    learning_rate=0.5,
    optimizer='gd',
    loss_function='mse',
    verbose=True
)

# Prediksi setelah training
print("\n" + "="*70)
print("HASIL TRAINING")
print("="*70)

print("\n" + "-"*70)
print("Bobot Baru (Updated Weights & Biases)")
print("-"*70)
print("\nLayer 1 (Input -> Hidden):")
for i in range(2):
    print(f"  Node {i}: W={[f'{w:.3f}' for w in nn.weights[1][i]]}, b={nn.biases[1][i]:.3f}")
print("\nLayer 2 (Hidden -> Output):")
for i in range(1):
    print(f"  Node {i}: W={[f'{w:.3f}' for w in nn.weights[2][i]]}, b={nn.biases[2][i]:.3f}")

print("\n" + "-"*70)
print("Prediksi SETELAH Training")
print("-"*70)
y_pred_after = nn.forward(X_and)
loss_after = nn.calculate_loss(y_and, y_pred_after, 'mse')

print(f"\n{'Input':<12} {'Target':<10} {'Prediksi':<10} {'Class':<10} {'Status'}")
print("-"*60)
for i in range(len(X_and)):
    pred_class = 1 if y_pred_after[i][0] >= 0.5 else 0
    status = "BENAR" if pred_class == y_and[i][0] else "SALAH"
    print(f"{str(X_and[i]):<12} {y_and[i][0]:<10} {y_pred_after[i][0]:<10.4f} {pred_class:<10} {status}")

print(f"\nLoss (MSE): {loss_after:.6f}")
print(f"Penurunan Loss: {loss_before - loss_after:.6f} ({(loss_before - loss_after)/loss_before * 100:.2f}%)")

# Accuracy
accuracy = np.mean([(1 if y_pred_after[i][0] >= 0.5 else 0) == y_and[i][0] for i in range(len(y_and))])
print(f"Akurasi: {accuracy * 100:.2f}%")

# Loss curve (first and last 10 epochs)
print("\n" + "-"*70)
print("Loss Curve (10 Epoch Pertama dan Terakhir)")
print("-"*70)
print("\nEpoch Pertama:")
print("Epoch\tLoss")
for i in range(min(10, len(history['epoch']))):
    print(f"{history['epoch'][i]}\t{history['loss'][i]:.6f}")

if len(history['epoch']) > 10:
    print("\n...")
    print("\nEpoch Terakhir:")
    print("Epoch\tLoss")
    for i in range(max(len(history['epoch'])-10, 0), len(history['epoch'])):
        print(f"{history['epoch'][i]}\t{history['loss'][i]:.6f}")

print("\n" + "="*70)
print("DEMO SELESAI")
print("="*70)
print("\nKesimpulan:")
print(f"  - Model berhasil dilatih selama {len(history['epoch'])} epoch")
print(f"  - Loss berkurang dari {loss_before:.6f} menjadi {loss_after:.6f}")
print(f"  - Akurasi akhir: {accuracy * 100:.2f}%")
print(f"  - Bobot dan bias telah diupdate melalui backpropagation")
