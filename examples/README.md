# Examples

This directory contains usage examples for ANN from Scratch v2.0.

## Available Examples

### 1. AND Gate Training (`example_and_gate.py`)

Basic example showing how to train a neural network to learn the AND gate logic.

**Usage:**
```bash
python examples/example_and_gate.py
```

**What it demonstrates:**
- Building a simple network (2-2-1)
- Setting custom connections and weights
- Training with Gradient Descent
- Evaluating predictions before/after training
- Calculating accuracy and loss improvement

**Expected Output:**
- Network architecture summary
- Training progress (loss per epoch)
- Predictions comparison
- Final accuracy: 75-100%

---

### 2. Multi-Class Classification (`example_multiclass.py`)

Example showing how to train a multi-class classifier using softmax activation.

**Usage:**
```bash
python examples/example_multiclass.py
```

**What it demonstrates:**
- Building a network with softmax output (3-4-2)
- Multi-class classification (Rain vs Sunny)
- Training with SGD optimizer
- Using categorical cross-entropy loss
- Evaluating class probabilities

**Expected Output:**
- Network with softmax activation
- Training with categorical cross-entropy
- Class probability predictions
- Final accuracy: 90-100%

---

## Creating Your Own Examples

### Template Structure

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core import NeuralNetwork
import numpy as np

def main():
    # 1. Build network
    network = NeuralNetwork()
    network.add_layer(input_size, 'linear')
    network.add_layer(hidden_size, 'sigmoid')
    network.add_layer(output_size, 'sigmoid')

    # 2. Set connections
    network.set_connections(...)

    # 3. Prepare data
    X = np.array([...])
    y = np.array([...])

    # 4. Train
    history = network.train(X, y, epochs=500, learning_rate=0.5)

    # 5. Evaluate
    y_pred, y_prob = network.predict(X)
    accuracy = np.mean((y_pred == y).astype(float))

    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
```

### Tips

1. **Start Simple**: Begin with small networks (2-2-1) before complex ones
2. **Normalize Data**: Scale inputs to [0, 1] range for better training
3. **Monitor Loss**: Check that loss decreases consistently
4. **Adjust Learning Rate**: If loss is unstable, reduce learning rate
5. **Use Right Loss**: Binary for binary, categorical for multi-class
6. **Enough Epochs**: Train long enough for convergence (500-2000 epochs)

### Common Use Cases

**Binary Classification:**
```python
network.add_layer(n_features, 'linear')
network.add_layer(n_hidden, 'sigmoid')
network.add_layer(1, 'sigmoid')  # Single output
# Use: loss_function='binary'
```

**Multi-Class Classification:**
```python
network.add_layer(n_features, 'linear')
network.add_layer(n_hidden, 'sigmoid')
network.add_layer(n_classes, 'softmax')  # Multiple outputs with softmax
# Use: loss_function='categorical'
```

**Multi-Label Classification:**
```python
network.add_layer(n_features, 'linear')
network.add_layer(n_hidden, 'sigmoid')
network.add_layer(n_labels, 'sigmoid')  # Multiple outputs with sigmoid
# Use: loss_function='binary'
```

**Regression:**
```python
network.add_layer(n_features, 'linear')
network.add_layer(n_hidden, 'relu')
network.add_layer(1, 'linear')  # Linear output
# Use: loss_function='mse'
```

---

## More Examples

For more advanced examples, see:
- **[API Examples](../docs/API.md#usage-examples)** - Using REST API
- **[Tests](../tests/integration/)** - Complete workflow tests
- **[Training Guide](../docs/TRAINING_GUIDE.md)** - Best practices

---

**Happy Learning! 🎓**
