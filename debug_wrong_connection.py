"""
Debug error saat user salah input connection index
"""
import numpy as np
from ann_core import NeuralNetwork

print("=" * 70)
print("DEBUG: Testing WRONG Connection Index")
print("=" * 70)

# Create 2-3-1 network
print("\n1. Creating 2-3-1 network...")
nn = NeuralNetwork()
nn.add_layer(2, 'linear')    # Input: 2 nodes (index 0, 1)
nn.add_layer(3, 'sigmoid')   # Hidden: 3 nodes (index 0, 1, 2)
nn.add_layer(1, 'sigmoid')   # Output: 1 node (index 0)

print("   Layer 0 (Input): 2 nodes -> valid indices: 0, 1")
print("   Layer 1 (Hidden): 3 nodes -> valid indices: 0, 1, 2")
print("   Layer 2 (Output): 1 node -> valid index: 0")

# Test 1: WRONG - Connect to node index 2, but input layer only has 2 nodes (0, 1)
print("\n2. TEST 1: WRONG connection (index 2 from 2-node layer)")
connections_layer1_wrong = [
    [0, 1, 2],  # ERROR! Input layer only has nodes 0, 1 (no node 2!)
    [0, 1],
    [0, 1]
]
weights_layer1_wrong = [
    [0.5, 0.3, 0.2],  # 3 weights
    [-0.4, 0.6],      # 2 weights
    [0.2, -0.5]       # 2 weights
]
biases_layer1 = [0.1, -0.2, 0.3]

print(f"   Connections: {connections_layer1_wrong}")
print(f"   Problem: Node 0 tries to connect to input node 2, but input only has nodes 0-1!")

nn.set_connections(1, connections_layer1_wrong, weights_layer1_wrong, biases_layer1)

connections_layer2 = [[0, 1, 2]]
weights_layer2 = [[0.7, -0.4, 0.5]]
biases_layer2 = [0.2]
nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

X = np.array([[0.5, 0.8]])

try:
    print("\n   Attempting forward pass...")
    y_pred = nn.forward(X)
    print(f"   [UNEXPECTED] Forward pass completed: {y_pred}")
except IndexError as e:
    print(f"   [ERROR CAUGHT] IndexError: {e}")
    print("   This is the error user experienced!")
except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")

# Test 2: WRONG - Connect output to node 3, but hidden layer only has 3 nodes (0, 1, 2)
print("\n3. TEST 2: WRONG connection (index 3 from 3-node layer)")
nn2 = NeuralNetwork()
nn2.add_layer(2, 'linear')
nn2.add_layer(3, 'sigmoid')
nn2.add_layer(1, 'sigmoid')

connections_layer1_correct = [[0, 1], [0, 1], [0, 1]]
weights_layer1_correct = [[0.5, 0.3], [-0.4, 0.6], [0.2, -0.5]]
nn2.set_connections(1, connections_layer1_correct, weights_layer1_correct, biases_layer1)

connections_layer2_wrong = [
    [0, 1, 2, 3]  # ERROR! Hidden layer only has nodes 0, 1, 2 (no node 3!)
]
weights_layer2_wrong = [
    [0.7, -0.4, 0.5, 0.3]  # 4 weights
]
biases_layer2 = [0.2]

print(f"   Connections: {connections_layer2_wrong}")
print(f"   Problem: Output tries to connect to hidden node 3, but hidden only has nodes 0-2!")

nn2.set_connections(2, connections_layer2_wrong, weights_layer2_wrong, biases_layer2)

try:
    print("\n   Attempting forward pass...")
    y_pred = nn2.forward(X)
    print(f"   [UNEXPECTED] Forward pass completed: {y_pred}")
except IndexError as e:
    print(f"   [ERROR CAUGHT] IndexError: {e}")
    print("   This is the error user experienced!")
except Exception as e:
    print(f"   [ERROR] {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("\nThe error occurs when user inputs connection index that")
print("exceeds the number of nodes in the previous layer.")
print("\nExample: Layer 1 has 2 nodes (indices 0, 1)")
print("         User tries to connect to index 2 -> ERROR!")
print("\nSOLUTION: Add validation in frontend/backend to check")
print("          connection indices are within valid range.")
print("=" * 70)
