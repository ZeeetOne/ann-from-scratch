"""
Debug why predictions are stuck at 0.5, 0.5
"""
import numpy as np
from ann_core import NeuralNetwork

# Load example multiclass network (sama seperti di app.py)
nn = NeuralNetwork()
nn.add_layer(3, 'linear')
nn.add_layer(4, 'sigmoid')
nn.add_layer(2, 'softmax')

# Set connections (same as quick_start_multiclass)
connections_layer1 = [
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2]
]
weights_layer1 = [
    [0.5, -0.3, 0.8],
    [-0.4, 0.6, -0.2],
    [0.7, -0.5, 0.3],
    [-0.6, 0.4, -0.7]
]
biases_layer1 = [0.1, -0.2, 0.3, -0.1]
nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

connections_layer2 = [
    [0, 1, 2, 3],
    [0, 1, 2, 3]
]
weights_layer2 = [
    [0.2, -0.4, 0.6, -0.3],
    [-0.5, 0.3, -0.1, 0.7]
]
biases_layer2 = [0.1, -0.1]
nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

print("=" * 70)
print("DEBUGGING PREDICTIONS STUCK AT 0.5, 0.5")
print("=" * 70)

# Example dataset (sama seperti yang digunakan di test)
X = np.array([
    [25, 1010, 85],   # Sample 1
    [30, 1015, 45],   # Sample 2
    [22, 1005, 90],   # Sample 3
])
y_true = np.array([
    [1, 0],  # Class 0
    [0, 1],  # Class 1
    [1, 0]   # Class 0
])

print("\nInput data (RAW - NOT NORMALIZED):")
print("X:")
print(X)
print("\nX statistics:")
print(f"  Min: {X.min(axis=0)}")
print(f"  Max: {X.max(axis=0)}")
print(f"  Mean: {X.mean(axis=0)}")

# Forward pass with raw data
print("\n" + "-" * 70)
print("FORWARD PASS WITH RAW DATA")
print("-" * 70)
y_pred = nn.forward(X)

print("\nLayer outputs:")
for i, layer_out in enumerate(nn.layer_outputs):
    print(f"\nLayer {i}:")
    print(f"  Shape: {layer_out.shape}")
    if i == 1:  # Hidden layer
        print(f"  Output:\n{layer_out}")
        print(f"  Min: {layer_out.min():.6e}, Max: {layer_out.max():.6e}")
        print(f"  Mean: {layer_out.mean():.6e}, Std: {layer_out.std():.6e}")

        # Check saturation
        num_saturated_low = np.sum(layer_out < 0.01)
        num_saturated_high = np.sum(layer_out > 0.99)
        total = layer_out.size
        print(f"  Saturated (< 0.01): {num_saturated_low}/{total} ({num_saturated_low/total*100:.1f}%)")
        print(f"  Saturated (> 0.99): {num_saturated_high}/{total} ({num_saturated_high/total*100:.1f}%)")
    elif i == 2:  # Output layer
        print(f"  Output:\n{y_pred}")

print("\nPredictions (RAW DATA):")
print(y_pred)
print("\nProbability sums per sample:", np.sum(y_pred, axis=1))

# Try with normalized data
print("\n" + "=" * 70)
print("TRY WITH NORMALIZED DATA")
print("=" * 70)

# Normalize using min-max scaling
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)

print("\nNormalized data:")
print(X_normalized)
print("\nNormalized statistics:")
print(f"  Min: {X_normalized.min(axis=0)}")
print(f"  Max: {X_normalized.max(axis=0)}")
print(f"  Mean: {X_normalized.mean(axis=0)}")

# Forward pass with normalized data
y_pred_norm = nn.forward(X_normalized)

print("\nLayer outputs (normalized):")
for i, layer_out in enumerate(nn.layer_outputs):
    if i == 1:  # Hidden layer
        print(f"\nHidden Layer {i}:")
        print(f"  Output:\n{layer_out}")
        print(f"  Min: {layer_out.min():.6f}, Max: {layer_out.max():.6f}")
        print(f"  Mean: {layer_out.mean():.6f}, Std: {layer_out.std():.6f}")

        # Check saturation
        num_saturated_low = np.sum(layer_out < 0.01)
        num_saturated_high = np.sum(layer_out > 0.99)
        total = layer_out.size
        print(f"  Saturated (< 0.01): {num_saturated_low}/{total} ({num_saturated_low/total*100:.1f}%)")
        print(f"  Saturated (> 0.99): {num_saturated_high}/{total} ({num_saturated_high/total*100:.1f}%)")

print("\nPredictions (NORMALIZED DATA):")
print(y_pred_norm)
print("\nProbability sums per sample:", np.sum(y_pred_norm, axis=1))

# Train with normalized data
print("\n" + "=" * 70)
print("TRAINING WITH NORMALIZED DATA")
print("=" * 70)

history = nn.train(
    X_normalized, y_true,
    epochs=200,
    learning_rate=1.0,
    optimizer='gd',
    loss_function='categorical',
    verbose=False
)

print(f"\nInitial Loss: {history['loss'][0]:.6f}")
print(f"Final Loss: {history['loss'][-1]:.6f}")
print(f"Improvement: {(history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100:.2f}%")

# Predictions after training
y_pred_trained = nn.forward(X_normalized)
print("\nPredictions AFTER training (normalized data):")
print(y_pred_trained)

# Calculate accuracy
y_pred_classes = (y_pred_trained > 0.5).astype(int)
accuracy = np.mean(y_pred_classes == y_true)
print(f"\nAccuracy: {accuracy*100:.2f}%")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("\nThe problem is FEATURE SCALING!")
print("Raw data has large values (temperature~1000, pressure~1000)")
print("This causes sigmoid saturation in hidden layer.")
print("\nSOLUTION: Normalize input data to [0, 1] range")
print("=" * 70)
