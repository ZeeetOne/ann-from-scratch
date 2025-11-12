"""
Debug error custom network 2-3-1
Error: index 2 is out of bounds for axis 0 with size 2
"""
import numpy as np
from ann_core import NeuralNetwork

print("=" * 70)
print("DEBUG: Custom Network 2-3-1")
print("=" * 70)

# Create 2-3-1 network
print("\n1. Creating 2-3-1 network...")
nn = NeuralNetwork()
nn.add_layer(2, 'linear')    # Input: 2 nodes
nn.add_layer(3, 'sigmoid')   # Hidden: 3 nodes
nn.add_layer(1, 'sigmoid')   # Output: 1 node

print("   Network created:")
print(f"   Layer 0 (Input):  {nn.layers[0]} nodes, activation: {nn.activations[0]}")
print(f"   Layer 1 (Hidden): {nn.layers[1]} nodes, activation: {nn.activations[1]}")
print(f"   Layer 2 (Output): {nn.layers[2]} nodes, activation: {nn.activations[2]}")

# Set connections for Layer 1 (input to hidden)
print("\n2. Setting connections for Layer 1 (Input -> Hidden)...")
connections_layer1 = [
    [0, 1],  # Hidden node 0 connects to input nodes 0, 1
    [0, 1],  # Hidden node 1 connects to input nodes 0, 1
    [0, 1]   # Hidden node 2 connects to input nodes 0, 1
]
weights_layer1 = [
    [0.5, 0.3],   # Weights for hidden node 0
    [-0.4, 0.6],  # Weights for hidden node 1
    [0.2, -0.5]   # Weights for hidden node 2
]
biases_layer1 = [0.1, -0.2, 0.3]

print(f"   Connections: {connections_layer1}")
print(f"   Weights: {weights_layer1}")
print(f"   Biases: {biases_layer1}")

nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)
print("   [OK] Layer 1 connections set")

# Set connections for Layer 2 (hidden to output)
print("\n3. Setting connections for Layer 2 (Hidden -> Output)...")
connections_layer2 = [
    [0, 1, 2]  # Output node 0 connects to hidden nodes 0, 1, 2
]
weights_layer2 = [
    [0.7, -0.4, 0.5]  # Weights for output node 0
]
biases_layer2 = [0.2]

print(f"   Connections: {connections_layer2}")
print(f"   Weights: {weights_layer2}")
print(f"   Biases: {biases_layer2}")

nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)
print("   [OK] Layer 2 connections set")

# Test input
print("\n4. Testing forward pass...")
X = np.array([[0.5, 0.8]])  # 1 sample, 2 features

print(f"   Input shape: {X.shape}")
print(f"   Input: {X}")

try:
    print("\n   Attempting forward pass...")
    y_pred = nn.forward(X)
    print(f"   [SUCCESS] Forward pass completed!")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Output: {y_pred}")
except Exception as e:
    print(f"\n   [ERROR] Forward pass failed!")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {e}")

    # Print traceback
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()

    # Debug info
    print("\n" + "-" * 70)
    print("DEBUG INFO")
    print("-" * 70)

    print("\nNetwork structure:")
    print(f"  layers: {nn.layers}")
    print(f"  activations: {nn.activations}")

    print("\nConnections:")
    for i, conns in enumerate(nn.connections):
        print(f"  Layer {i}: {conns}")

    print("\nWeights:")
    for i, weights in enumerate(nn.weights):
        print(f"  Layer {i}: {weights}")

    print("\nBiases:")
    for i, biases in enumerate(nn.biases):
        print(f"  Layer {i}: {biases}")

    print("\nLayer outputs (before error):")
    for i, output in enumerate(nn.layer_outputs):
        if output is not None:
            print(f"  Layer {i}: shape={output.shape}, values={output}")
        else:
            print(f"  Layer {i}: None")

print("\n" + "=" * 70)
