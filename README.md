# 🧠 ANN from Scratch v2.0 - Professional Neural Network Framework

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/yourrepo/ann-from-scratch)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

**Interactive web application for building, training, and testing Artificial Neural Networks from scratch using Python and NumPy.**

> 🎉 **v2.0 Major Refactoring**: Completely rewritten with professional software engineering practices! Clean Architecture, SOLID principles, comprehensive testing, and full documentation.

## ✨ What's New in v2.0

### 🏗️ Professional Architecture
- **Clean Architecture** with clear separation of concerns
- **SOLID Principles** applied throughout the codebase
- **Design Patterns**: Strategy, Factory, Facade, Application Factory
- **Modular Structure**: Backend (Core/Services/API), Frontend, Tests, Docs
- **Scalable & Maintainable**: Easy to understand, extend, and test

### 🧪 Comprehensive Testing
- **Unit Tests**: All core components tested in isolation
- **Integration Tests**: End-to-end workflow validation
- **Test Coverage**: Activation functions, loss functions, optimizers, neural network
- **Verified Correctness**: AND gate, multi-class classification tests

### 📚 Full Documentation
- **[Architecture Documentation](docs/ARCHITECTURE.md)**: Design decisions and patterns
- **[API Documentation](docs/API.md)**: Complete REST API reference
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Hyperparameter tuning best practices

### ⚡ Enhanced Features
- **Factory Pattern** for easy extension (add new activations, losses, optimizers)
- **Better Error Handling** with centralized middleware
- **Request Validation** with clear error messages
- **Dependency Injection** for testability
- **Configuration Management** for different environments

## 🎯 Features

### Network Configuration
- **Flexible Architecture**: Define custom layer sizes and connections
- **Multiple Activation Functions**:
  - Sigmoid, ReLU, Linear, Softmax, Threshold
- **Custom Connections**: Manually specify node connections
- **Custom Weights & Biases**: Full control over initialization

### Training & Optimization
- **Backpropagation**: Complete implementation from scratch
- **Multiple Optimizers**:
  - **Gradient Descent (GD)**: Full batch, stable convergence
  - **Stochastic Gradient Descent (SGD)**: Mini-batch support
  - **Momentum**: Accelerated convergence (new!)
- **Multiple Loss Functions**:
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
- **Hyperparameter Control**: Learning rate, epochs, batch size

### Evaluation & Visualization
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Per-class performance analysis
- **Training History**: Loss curve visualization
- **Detailed Forward Pass**: Layer-by-layer activation inspection

### User Interface
- **Interactive Web UI**: Modern, user-friendly interface
- **Real-time Results**: Instant feedback during training
- **Responsive Design**: Works on desktop and mobile
- **API Access**: Full REST API for programmatic use

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Steps

1. **Clone or Download Repository**
   ```bash
   git clone https://github.com/yourrepo/ann-from-scratch.git
   cd ann-from-scratch
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Option 1: Web Interface

1. **Run the application:**
   ```bash
   python run.py
   ```

2. **Open browser:**
   ```
   http://localhost:5000
   ```

3. **Try examples:**
   - Click "Load Example Network" for quick start
   - Modify parameters and train
   - See results instantly

### Option 2: Python API

```python
from backend.core import NeuralNetwork
import numpy as np

# Build network (2-2-1 for AND gate)
network = NeuralNetwork()
network.add_layer(2, 'linear')    # Input
network.add_layer(2, 'sigmoid')   # Hidden
network.add_layer(1, 'sigmoid')   # Output

# Set connections
network.set_connections(
    1, [[0,1], [0,1]],
    [[0.5,-0.3], [-0.4,0.6]],
    [0.1,-0.2]
)
network.set_connections(
    2, [[0,1]],
    [[0.8,-0.5]],
    [0.2]
)

# Prepare AND gate data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [0], [0], [1]])

# Train
history = network.train(
    X, y,
    epochs=500,
    learning_rate=0.5,
    optimizer='gd',
    loss_function='mse'
)

# Predict
y_pred_classes, y_pred_probs = network.predict(X)
print(f"Final Loss: {history['loss'][-1]:.6f}")
print(f"Predictions: {y_pred_probs.flatten()}")
```

### Option 3: REST API

```python
import requests

# Quick start with example network
response = requests.post('http://localhost:5000/quick_start_binary')
data = response.json()
dataset = data['example_dataset']

# Train the network
train_response = requests.post('http://localhost:5000/train', json={
    'dataset': dataset,
    'epochs': 1000,
    'learning_rate': 0.5,
    'optimizer': 'gd',
    'loss_function': 'binary'
})

result = train_response.json()
print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
```

## 📁 Project Structure

```
ann-from-scratch/
├── backend/                    # Backend Application
│   ├── core/                   # Core ML Algorithms
│   │   ├── activation_functions.py  # Sigmoid, ReLU, Softmax, etc.
│   │   ├── loss_functions.py        # MSE, Cross-Entropy, etc.
│   │   ├── optimizers.py            # GD, SGD, Momentum
│   │   └── neural_network.py        # Main Network Class
│   ├── services/               # Business Logic
│   │   ├── network_service.py       # Network management
│   │   ├── training_service.py      # Training orchestration
│   │   └── data_service.py          # Data processing
│   ├── api/                    # REST API
│   │   ├── routes/             # Endpoint definitions
│   │   ├── middleware/         # Error handling
│   │   └── app.py              # Flask application
│   ├── utils/                  # Utilities
│   └── config.py               # Configuration
├── frontend/                   # Frontend Application
│   ├── static/                 # CSS & JavaScript
│   └── templates/              # HTML templates
├── tests/                      # Test Suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── examples/                   # Usage examples
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # Design & patterns
│   ├── API.md                  # API reference
│   └── TRAINING_GUIDE.md       # Training tips
├── requirements.txt            # Dependencies
└── run.py                      # Application entry point
```

## 🎓 Usage Examples

### Binary Classification

```python
# Load example binary classification network (3-4-1)
response = requests.post('http://localhost:5000/quick_start_binary')
# Returns: Student pass/fail prediction network
```

### Multi-Class Classification

```python
# Load example multi-class network (3-4-2 with softmax)
response = requests.post('http://localhost:5000/quick_start_multiclass')
# Returns: Weather prediction network (rain/sunny)
```

### Custom Network

```python
# Build your own network
network = NeuralNetwork()
network.add_layer(3, 'linear')      # 3 input features
network.add_layer(4, 'sigmoid')     # 4 hidden neurons
network.add_layer(2, 'softmax')     # 2 classes

# Fully connected layers
network.set_full_connections(
    1,
    weight_matrix=np.random.randn(4, 3) * 0.5,
    biases=np.zeros(4)
)
network.set_full_connections(
    2,
    weight_matrix=np.random.randn(2, 4) * 0.5,
    biases=np.zeros(2)
)

# Train with your data
history = network.train(X_train, y_train, epochs=1000)
```

## 🧪 Testing

### Run Unit Tests
```bash
python tests/unit/test_activation_functions.py
```

### Run Integration Tests
```bash
python tests/integration/test_complete_workflow.py
```

### Run API Tests
```bash
# First, start the server:
python run.py

# In another terminal:
python tests/integration/test_api_manual.py
```

## 📊 Performance

Validated on standard datasets:

- **AND Gate**: 75-100% accuracy (500 epochs)
- **Multi-Class Classification**: 100% accuracy (200 epochs)
- **Loss Reduction**: Consistent decrease with proper hyperparameters

## 🔧 Configuration

### Environment Variables

```bash
# Development mode
DEBUG=True
HOST=localhost
PORT=5000

# Production mode
DEBUG=False
HOST=0.0.0.0
PORT=80
```

### Configuration Files

```python
# Use different configs
python run.py --config development  # Default
python run.py --config production
python run.py --config testing
```

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Design decisions, patterns, and extension points
- **[API Reference](docs/API.md)** - Complete REST API documentation
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Best practices for training networks
- **[Examples](examples/)** - Code examples for common tasks

## 🛠️ Technologies

- **Backend**: Python 3.8+, Flask
- **Computation**: NumPy
- **Data Processing**: Pandas
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Architecture**: Clean Architecture, SOLID Principles
- **Patterns**: Strategy, Factory, Facade, Application Factory

## 🎯 Key Design Patterns

### Strategy Pattern
```python
# Easy to swap algorithms
network.train(X, y, optimizer='sgd')   # Use SGD
network.train(X, y, optimizer='gd')    # Use GD
network.train(X, y, optimizer='momentum')  # Use Momentum
```

### Factory Pattern
```python
# Create instances by name
activation = ActivationFactory.create('sigmoid')
loss = LossFunctionFactory.create('binary')
optimizer = OptimizerFactory.create('sgd', learning_rate=0.01)
```

### Facade Pattern
```python
# Services simplify complex operations
results = TrainingService.train_network(network, X, y, config)
predictions = DataService.process_predictions(network, X, y)
```

## 🚀 Extending the Framework

### Add New Activation Function

```python
from backend.core import ActivationFunction, ActivationFactory

class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2

    @property
    def name(self):
        return "tanh"

# Register
ActivationFactory.register('tanh', Tanh)

# Use
network.add_layer(4, 'tanh')
```

### Add New Optimizer

```python
from backend.core import Optimizer, OptimizerFactory

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        # ... initialize momentum and velocity

    def update(self, params, gradients):
        # ... Adam update logic
        return updated_params

    @property
    def name(self):
        return "adam"

# Register
OptimizerFactory.register('adam', Adam)

# Use
network.train(X, y, optimizer='adam')
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Adam optimizer
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] L1/L2 regularization
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] Model save/load
- [ ] More activation functions (tanh, leaky ReLU, ELU)
- [ ] Visualization improvements
- [ ] Performance optimizations

## 📝 License

MIT License - Free to use for educational and commercial purposes.

## 🙏 Acknowledgments

Built with best practices from:

- **Martin Fowler** - Refactoring
- **Robert C. Martin** - Clean Architecture, SOLID Principles
- **Gang of Four** - Design Patterns
- **Miguel Grinberg** - Flask Web Development

## 📧 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: See [docs/](docs/) directory
- **Examples**: See [examples/](examples/) directory

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

**Built with ❤️ for Deep Learning Education**

*Learn by building. Understand by implementing. Master by teaching.*

**Version 2.0.0** - Professional Edition with Clean Architecture
