"""
Demo Multi-Output Neural Network (3-4-2 Architecture)
Prediksi cuaca berdasarkan suhu, tekanan, dan kelembapan
"""

import numpy as np
from ann_core import NeuralNetwork

# Create 3-4-2 network
print("="*60)
print("Building Neural Network (3-4-2 Architecture)")
print("="*60)

nn = NeuralNetwork()

# Layer 0: Input layer (3 nodes - suhu, tekanan, kelembapan)
nn.add_layer(3, 'linear')

# Layer 1: Hidden layer (4 nodes)
nn.add_layer(4, 'sigmoid')

# Layer 2: Output layer (2 nodes - hujan, cerah)
nn.add_layer(2, 'sigmoid')

# Set connections for layer 1 (input to hidden)
connections_layer1 = [
    [0, 1, 2],  # Node 0 connects to all 3 input nodes
    [0, 1, 2],  # Node 1 connects to all 3 input nodes
    [0, 1, 2],  # Node 2 connects to all 3 input nodes
    [0, 1, 2]   # Node 3 connects to all 3 input nodes
]
weights_layer1 = [
    [0.5, 0.3, -0.2],   # Weights for hidden node 0
    [-0.4, 0.6, 0.1],   # Weights for hidden node 1
    [0.2, -0.5, 0.4],   # Weights for hidden node 2
    [0.7, 0.2, -0.3]    # Weights for hidden node 3
]
biases_layer1 = [0.1, -0.2, 0.3, 0.0]

nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

# Set connections for layer 2 (hidden to output)
connections_layer2 = [
    [0, 1, 2, 3],  # Output node 0 (hujan) connects to all 4 hidden nodes
    [0, 1, 2, 3]   # Output node 1 (cerah) connects to all 4 hidden nodes
]
weights_layer2 = [
    [0.8, -0.3, 0.6, 0.4],   # Weights for output node 0 (hujan)
    [-0.5, 0.7, -0.2, 0.3]   # Weights for output node 1 (cerah)
]
biases_layer2 = [0.1, -0.1]

nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print(nn.get_architecture_summary())
print()

# Prepare dataset (weather prediction)
print("="*60)
print("Dataset: Weather Prediction")
print("="*60)
print("Features: Suhu (°C), Tekanan (hPa), Kelembapan (%)")
print("Outputs: Hujan (1=ya, 0=tidak), Cerah (1=ya, 0=tidak)")
print()

# suhu, tekanan, kelembapan
X = np.array([
    [25, 1010, 85],  # Hujan
    [30, 1015, 45],  # Cerah
    [22, 1005, 90],  # Hujan
    [28, 1012, 50],  # Cerah
    [20, 1000, 95],  # Hujan
    [32, 1018, 40],  # Cerah
    [24, 1008, 80],  # Hujan
    [29, 1014, 48],  # Cerah
    [21, 1003, 88],  # Hujan
    [31, 1016, 42],  # Cerah
])

# hujan, cerah
y = np.array([
    [1, 0],  # Hujan
    [0, 1],  # Cerah
    [1, 0],  # Hujan
    [0, 1],  # Cerah
    [1, 0],  # Hujan
    [0, 1],  # Cerah
    [1, 0],  # Hujan
    [0, 1],  # Cerah
    [1, 0],  # Hujan
    [0, 1],  # Cerah
])

# Predictions before training
print("Predictions BEFORE Training:")
print("-"*60)
y_pred_classes, y_pred_probs = nn.predict(X)
initial_loss = nn.calculate_loss(y, y_pred_probs, 'mse')

for i in range(len(X)):
    print(f"Sample {i+1}: Suhu={X[i][0]}°C, Tekanan={X[i][1]}hPa, Kelembapan={X[i][2]}%")
    print(f"  Actual: Hujan={y[i][0]}, Cerah={y[i][1]}")
    print(f"  Predicted: Hujan={y_pred_classes[i][0]} ({y_pred_probs[i][0]:.3f}), Cerah={y_pred_classes[i][1]} ({y_pred_probs[i][1]:.3f})")
    correct = np.array_equal(y[i], y_pred_classes[i])
    print(f"  Status: {'[OK] CORRECT' if correct else '[X] INCORRECT'}")
    print()

print(f"Initial Loss: {initial_loss:.6f}")
print()

# Train the network
print("="*60)
print("Training with SGD...")
print("="*60)

history = nn.train(
    X, y,
    epochs=2000,
    learning_rate=1.0,
    optimizer='sgd',
    loss_function='mse',
    batch_size=2,
    verbose=False
)

print(f"Training completed!")
print(f"Final Loss: {history['loss'][-1]:.6f}")
print(f"Loss Improvement: {((initial_loss - history['loss'][-1]) / initial_loss * 100):.2f}%")
print()

# Predictions after training
print("="*60)
print("Predictions AFTER Training:")
print("="*60)
y_pred_classes, y_pred_probs = nn.predict(X)
final_loss = nn.calculate_loss(y, y_pred_probs, 'mse')

correct_count = 0
for i in range(len(X)):
    print(f"Sample {i+1}: Suhu={X[i][0]}°C, Tekanan={X[i][1]}hPa, Kelembapan={X[i][2]}%")
    print(f"  Actual: Hujan={y[i][0]}, Cerah={y[i][1]}")
    print(f"  Predicted: Hujan={y_pred_classes[i][0]} ({y_pred_probs[i][0]:.3f}), Cerah={y_pred_classes[i][1]} ({y_pred_probs[i][1]:.3f})")
    correct = np.array_equal(y[i], y_pred_classes[i])
    if correct:
        correct_count += 1
    print(f"  Status: {'[OK] CORRECT' if correct else '[X] INCORRECT'}")
    print()

accuracy = correct_count / len(X) * 100
print(f"Final Loss: {final_loss:.6f}")
print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(X)} correct)")
print()

print("="*60)
print("Demo completed successfully!")
print("="*60)
