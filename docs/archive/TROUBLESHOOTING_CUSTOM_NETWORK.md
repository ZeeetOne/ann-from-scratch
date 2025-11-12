# Troubleshooting: Custom Network Error "index 2 is out of bounds for axis 0 with size 2"

## Problem
Ketika membuat custom network 2-3-1 (2 input, 3 hidden dengan ReLU, 1 output), terjadi error:
```
Error: index 2 is out of bounds for axis 0 with size 2
```

## Root Causes

Setelah extensive testing, error ini **BUKAN disebabkan oleh**:
- ✅ ReLU activation
- ✅ Network architecture 2-3-1
- ✅ Forward/backward pass implementation

Error **DISEBABKAN oleh salah satu dari**:

### 1. ❌ Connection Index yang Salah

**Masalah:** User menginput connection index yang melebihi jumlah nodes di layer sebelumnya.

**Contoh Error:**
```python
# Layer 0 (Input): 2 nodes -> valid indices: 0, 1 (ONLY!)
# Layer 1 (Hidden): 3 nodes

# WRONG! Trying to connect to node 2, but input only has nodes 0, 1
connections_layer1 = [
    [0, 1, 2],  # ❌ Index 2 tidak ada di layer 0!
    [0, 1],
    [0, 1]
]
```

**Solution:**
Pastikan connection indices sesuai dengan jumlah nodes:
```python
# Layer 0 has 2 nodes (indices: 0, 1)
# Layer 1 has 3 nodes (indices: 0, 1, 2)

# CORRECT connections:
connections_layer1 = [
    [0, 1],  # ✅ Hidden node 0 connects to input nodes 0, 1
    [0, 1],  # ✅ Hidden node 1 connects to input nodes 0, 1
    [0, 1]   # ✅ Hidden node 2 connects to input nodes 0, 1
]

connections_layer2 = [
    [0, 1, 2]  # ✅ Output node connects to hidden nodes 0, 1, 2
]
```

### 2. ❌ Dataset dengan Jumlah Features yang Salah

**Masalah:** Dataset tidak match dengan jumlah input nodes.

**Contoh Error:**
```csv
# Network expects 2 features (2 input nodes)
# But dataset has 3 features:
x1,x2,x3,y
0.5,0.8,0.2,1   # ❌ 3 features, but network expects 2!
```

**Solution:**
```csv
# Network: 2-3-1 (2 input nodes)
# Dataset MUST have 2 features:
x1,x2,y
0.5,0.8,1  # ✅ 2 features
0.3,0.6,0
```

### 3. ❌ Format Input yang Salah di Web UI

**Masalah:** User input format tidak sesuai expected format.

**Solution:**
Di web UI, pastikan format input:
- **Connections**: `0,1` (pisahkan dengan koma, NO SPACES)
- **Weights**: `0.5,0.3` (sesuai jumlah connections)
- **Bias**: Single number (e.g., `0.1`)

## How to Fix

### Checklist untuk Custom Network 2-3-1:

```
✅ Layer 0 (Input): 2 nodes
   - Valid indices: 0, 1

✅ Layer 1 (Hidden): 3 nodes
   - Connection to Layer 0: Use indices 0, 1 ONLY
   - Correct: [0,1] [0,1] [0,1]
   - Wrong: [0,1,2] (no node 2 in input!)

✅ Layer 2 (Output): 1 node
   - Connection to Layer 1: Use indices 0, 1, 2
   - Correct: [0,1,2]
   - Wrong: [0,1,2,3] (no node 3 in hidden!)

✅ Dataset: Must have 2 features + 1 target
   - Correct: x1,x2,y
   - Wrong: x1,x2,x3,y (too many features!)
```

## Example: Correct Configuration

### Network: 2-3-1 dengan ReLU

```python
# Layer Configuration
Layer 0 (Input):  2 nodes, activation: linear
Layer 1 (Hidden): 3 nodes, activation: relu
Layer 2 (Output): 1 node,  activation: sigmoid

# Layer 1 Connections (Input -> Hidden)
Node 0: connects to [0, 1], weights [0.5, 0.3], bias 0.1
Node 1: connects to [0, 1], weights [-0.4, 0.6], bias -0.2
Node 2: connects to [0, 1], weights [0.2, -0.5], bias 0.3

# Layer 2 Connections (Hidden -> Output)
Node 0: connects to [0, 1, 2], weights [0.7, -0.4, 0.5], bias 0.2
```

### Dataset (NORMALIZED!)

```csv
x1,x2,y
0.5,0.8,1
0.3,0.6,0
0.7,0.9,1
```

**Important:** ALWAYS normalize features to [0, 1] range to prevent:
- Sigmoid saturation
- ReLU explosion
- Numerical instability

## Testing

Jika masih error, test dengan script ini:

```python
import numpy as np
from ann_core import NeuralNetwork

# Create 2-3-1 network
nn = NeuralNetwork()
nn.add_layer(2, 'linear')
nn.add_layer(3, 'relu')
nn.add_layer(1, 'sigmoid')

# Set connections (CORRECT!)
connections_layer1 = [[0, 1], [0, 1], [0, 1]]
weights_layer1 = [[0.5, 0.3], [-0.4, 0.6], [0.2, -0.5]]
biases_layer1 = [0.1, -0.2, 0.3]
nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

connections_layer2 = [[0, 1, 2]]
weights_layer2 = [[0.7, -0.4, 0.5]]
biases_layer2 = [0.2]
nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

# Test
X = np.array([[0.5, 0.8], [0.3, 0.6]])
y = np.array([[1], [0]])

y_pred = nn.forward(X)
print("Success! Output:", y_pred)
```

## Summary

**Most Common Cause:** Connection indices melebihi jumlah nodes di layer sebelumnya.

**Quick Fix:**
1. Check connection indices: harus < jumlah nodes di previous layer
2. Check dataset: jumlah features harus = jumlah input nodes
3. Normalize data ke [0, 1] range

Jika masih error, share:
- Exact error message
- Network configuration (layers, activations)
- Connections & weights
- Dataset format
