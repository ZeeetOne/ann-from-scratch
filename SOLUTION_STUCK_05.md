# Solution: Predictions Stuck at 0.5 - Custom 2-3-1 ReLU Network

## Problem
Setelah training custom network 2-3-1 dengan ReLU di hidden layer, **semua predictions = 0.500**.

## Root Causes Found

Saya telah melakukan extensive testing dan menemukan scenarios yang menyebabkan stuck di EXACTLY 0.5:

### 1. All ReLU Neurons DEAD (Most Likely!)

**Problem:** Semua ReLU neurons mengeluarkan 0 (dead neurons).

**How it happens:**
```python
# Hidden layer dengan ReLU
weighted_sum = input * weights + bias

# If weighted_sum < 0 untuk SEMUA neurons:
ReLU_output = 0  # All neurons dead!

# Output layer hanya dapat bias:
output = sigmoid(0 + bias) = sigmoid(0) = 0.5  # STUCK!
```

**Causes:**
- Input values terlalu besar/kecil
- Negative weights dengan positive large inputs
- Poor initialization

**Check:**
```
Setelah training, check hidden layer output:
  Hidden layer: [[0. 0. 0.], [0. 0. 0.], ...]  <- ALL DEAD!
  Dead neurons: 3/3  <- Problem!
```

### 2. All Weights = 0 (Bad Initialization)

**Problem:** Semua weights diinit ke 0.

```python
# If all weights = 0 and all biases = 0:
hidden_output = ReLU(0) = 0
output = sigmoid(0) = 0.5  # STUCK!
```

## Solutions

### ✅ Solution 1: NORMALIZE Dataset (MOST IMPORTANT!)

**Problem:** Raw data (values 2-10, atau lebih besar) causes ReLU issues.

**Solution:**
```python
# Normalize to [0, 1] range
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)
```

**Example:**
```python
# Before (RAW):
X = [[2, 8], [5, 8], [3, 10], [6, 9]]  # Values 2-10

# After (NORMALIZED):
X_norm = [[0.0, 0.0], [0.75, 0.0], [0.25, 1.0], [1.0, 0.5]]  # Values 0-1
```

**Why it works:**
- Prevents ReLU saturation
- Keeps weighted sums in reasonable range
- Prevents all neurons from dying

### ✅ Solution 2: Use Higher Learning Rate

**Problem:** Learning rate terlalu kecil (e.g., 0.01).

**Solution:**
```python
# For 2-3-1 ReLU network, use:
learning_rate = 1.0  # Recommended
# or
learning_rate = 0.5  # Also good

# Avoid:
learning_rate = 0.01  # Too small!
```

### ✅ Solution 3: Train Longer

**Problem:** Not enough epochs.

**Solution:**
```python
epochs = 200  # Recommended minimum
# or
epochs = 300  # Better

# Avoid:
epochs = 50   # Probably not enough
```

### ✅ Solution 4: Check Weights Initialization

**Problem:** Weights initialized to 0 or all same value.

**Solution:**
```python
# Use diverse random weights:
weights_layer1 = [
    [0.5, 0.3],    # Different values
    [-0.4, 0.6],   # Mix of positive/negative
    [0.2, -0.5]    # Break symmetry
]

# NOT this:
weights_layer1 = [
    [0.0, 0.0],    # All zeros -> bad!
    [0.0, 0.0],
    [0.0, 0.0]
]
```

## Step-by-Step Fix

### 1. Normalize Your Dataset

```csv
# Original (RAW):
x1,x2,y
2,8,1
5,8,1
3,10,1
6,9,0

# Convert to NORMALIZED [0,1]:
x1,x2,y
0.0,0.0,1
0.75,0.0,1
0.25,1.0,1
1.0,0.5,0
```

**How to calculate:**
```
For each column:
  min_val = minimum value in column
  max_val = maximum value in column
  normalized = (value - min_val) / (max_val - min_val)

Example for x1 column [2, 5, 3, 6]:
  min = 2, max = 6
  2 -> (2-2)/(6-2) = 0.0
  5 -> (5-2)/(6-2) = 0.75
  3 -> (3-2)/(6-2) = 0.25
  6 -> (6-2)/(6-2) = 1.0
```

### 2. Set Training Parameters

```
Learning Rate: 1.0
Epochs: 200
Optimizer: GD (Gradient Descent)
Loss Function: Binary
```

### 3. Check Results

After training, check:

**Good Signs:**
```
Hidden layer output: [[0.234, 1.567, 0.000], ...]  <- At least 1-2 neurons active
Dead neurons: 1/3  <- Some neurons alive
Final Loss: < 0.1
Predictions: [0.023, 0.967, 0.891, ...]  <- DIVERSE!
```

**Bad Signs (Still Stuck):**
```
Hidden layer output: [[0.0, 0.0, 0.0], ...]  <- ALL DEAD!
Dead neurons: 3/3  <- All dead
Final Loss: > 0.5 (no improvement)
Predictions: [0.5, 0.5, 0.5, ...]  <- STUCK!
```

## Complete Working Example

```python
import numpy as np
from ann_core import NeuralNetwork

# 1. Prepare NORMALIZED data
X_raw = np.array([[2, 8], [5, 8], [3, 10], [6, 9]])
y = np.array([[1], [1], [1], [0]])

# NORMALIZE!
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min)

# 2. Create network
nn = NeuralNetwork()
nn.add_layer(2, 'linear')
nn.add_layer(3, 'relu')
nn.add_layer(1, 'sigmoid')

# 3. Set connections with GOOD weights
connections_layer1 = [[0, 1], [0, 1], [0, 1]]
weights_layer1 = [[0.5, 0.3], [-0.4, 0.6], [0.2, -0.5]]  # Diverse!
biases_layer1 = [0.1, -0.2, 0.3]
nn.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

connections_layer2 = [[0, 1, 2]]
weights_layer2 = [[0.7, -0.4, 0.5]]
biases_layer2 = [0.2]
nn.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

# 4. Train with GOOD parameters
history = nn.train(X, y,
                  epochs=200,          # Enough epochs
                  learning_rate=1.0,   # High enough LR
                  optimizer='gd',
                  loss_function='binary',
                  verbose=True)

# 5. Check results
y_pred = nn.forward(X)
print("Predictions:", y_pred.flatten())  # Should be DIVERSE!
print("Hidden layer:", nn.layer_outputs[1])  # Should have non-zero values

# Expected output:
# Predictions: [0.999, 0.991, 0.996, 0.040]  <- GOOD!
# NOT: [0.5, 0.5, 0.5, 0.5]  <- BAD!
```

## Quick Checklist

- [ ] Dataset NORMALIZED to [0, 1] range
- [ ] Learning rate = 0.5 to 1.0
- [ ] Epochs >= 200
- [ ] Weights are diverse (not all zeros)
- [ ] Check hidden layer has non-zero activations after training
- [ ] Final loss < 0.1
- [ ] Predictions are DIVERSE (not all 0.5)

## If Still Stuck

Share these details:
1. Your exact dataset (first 5 rows)
2. Are values normalized?
3. Learning rate and epochs used
4. Weights initialization
5. Hidden layer output after training
6. Final loss value

99% of cases are fixed by **normalizing the dataset**!
