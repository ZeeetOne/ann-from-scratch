# ANN from Scratch v2.0 - Documentation

## Welcome

This is the comprehensive documentation for ANN from Scratch v2.0, a refactored, scalable implementation of neural networks built from scratch using Python and NumPy.

## Documentation Structure

- **[README.md](../README.md)** - Main project documentation (start here!)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture and design decisions
- **[API.md](API.md)** - Complete API reference
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training tips and best practices

## Quick Links

### Getting Started
- [Installation](#installation) - Set up the project
- [Quick Start](#quick-start) - Run your first network
- [Basic Usage](#basic-usage) - Common operations

### Advanced
- [Architecture](ARCHITECTURE.md) - System design and patterns
- [API Reference](API.md) - Complete endpoint documentation
- [Training Guide](TRAINING_GUIDE.md) - Hyperparameter tuning

### Development
- [Testing](#testing) - Running tests
- [Contributing](#contributing) - How to contribute
- [Extending](#extending) - Adding new features

## What's New in v2.0

### Major Refactoring

🎉 **v2.0 is a complete rewrite with professional software engineering practices!**

**Key Improvements:**

1. **Clean Architecture** - Proper separation of concerns
2. **SOLID Principles** - Single responsibility, open/closed, etc.
3. **Design Patterns** - Strategy, Factory, Facade patterns
4. **Modular Structure** - Easy to understand and extend
5. **Comprehensive Testing** - Unit and integration tests
6. **Better Documentation** - Architecture docs, API docs

### Structure Changes

**Before (v1.0):**
```
ann-from-scratch/
├── ann_core.py          # 600+ lines monolithic file
├── app.py               # 1000+ lines Flask routes
├── static/
└── templates/
```

**After (v2.0):**
```
ann-from-scratch/
├── backend/
│   ├── core/           # ML algorithms (4 files)
│   ├── services/       # Business logic (3 files)
│   ├── api/            # Routes & middleware (4 files)
│   └── utils/          # Utilities (2 files)
├── frontend/           # Presentation layer
├── tests/              # Comprehensive tests
├── examples/           # Usage examples
└── docs/               # Full documentation
```

### Benefits

- **Scalable**: Easy to add new features
- **Maintainable**: Clear code organization
- **Testable**: Comprehensive test coverage
- **Extensible**: Plugin architecture for new algorithms
- **Professional**: Industry-standard practices

## Navigation Guide

### For Users

1. Start with **[Main README](../README.md)** for features and installation
2. Try **[Quick Start Examples](../README.md#quick-start)**
3. Read **[Training Guide](TRAINING_GUIDE.md)** for best practices
4. Refer to **[API Docs](API.md)** for programmatic access

### For Developers

1. Read **[Architecture Docs](ARCHITECTURE.md)** to understand design
2. Check **[Extension Guide](ARCHITECTURE.md#extension-points)** to add features
3. Review **[Testing Strategy](ARCHITECTURE.md#testing-strategy)** for tests
4. Follow **[SOLID Principles](ARCHITECTURE.md#solid-principles)** when contributing

### For Researchers

1. Understand **[Core Algorithms](ARCHITECTURE.md#core-layer)** implementation
2. Review **[Forward/Backward Pass](ARCHITECTURE.md#data-flow)** logic
3. Study **[Optimization Methods](ARCHITECTURE.md#optimizers)** details
4. Check **[Test Results](../tests/integration/test_complete_workflow.py)** for validation

## Directory Reference

### Backend

- `backend/core/` - Neural network algorithms
  - `activation_functions.py` - Sigmoid, ReLU, Softmax, etc.
  - `loss_functions.py` - MSE, Cross-Entropy, etc.
  - `optimizers.py` - GD, SGD, Momentum
  - `neural_network.py` - Main network class

- `backend/services/` - Business logic
  - `network_service.py` - Network management
  - `training_service.py` - Training orchestration
  - `data_service.py` - Data processing

- `backend/api/` - Web API
  - `routes/` - Endpoint definitions
  - `middleware/` - Error handling, CORS
  - `app.py` - Flask application factory

- `backend/utils/` - Utilities
  - `validators.py` - Request validation
  - `data_processor.py` - Data formatting

### Frontend

- `frontend/static/` - CSS and JavaScript
- `frontend/templates/` - HTML templates

### Tests

- `tests/unit/` - Unit tests for components
- `tests/integration/` - End-to-end tests

### Documentation

- `docs/` - This directory
- All markdown documentation files

## Examples

### Python Script Example

```python
from backend.core import NeuralNetwork

# Build network
network = NeuralNetwork()
network.add_layer(2, 'linear')
network.add_layer(2, 'sigmoid')
network.add_layer(1, 'sigmoid')

# Set connections
network.set_connections(1, [[0,1], [0,1]], [[0.5,-0.3], [-0.4,0.6]], [0.1,-0.2])
network.set_connections(2, [[0,1]], [[0.8,-0.5]], [0.2])

# Train
import numpy as np
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [0], [0], [1]])

history = network.train(X, y, epochs=500, learning_rate=0.5)
print(f"Final loss: {history['loss'][-1]}")
```

### API Example

```python
import requests

# Build network
response = requests.post('http://localhost:5000/build_network', json={
    'layers': [...],
    'connections': [...]
})

# Train
response = requests.post('http://localhost:5000/train', json={
    'dataset': 'x1,x2,y\n...',
    'epochs': 1000,
    'learning_rate': 0.5
})
```

## Support

- **Issues**: [GitHub Issues](https://github.com/anthropics/ann-from-scratch/issues)
- **Documentation**: You're here!
- **Examples**: See `examples/` directory
- **Tests**: See `tests/` directory

## License

MIT License - Free to use for educational purposes.

## Acknowledgments

Built with best practices from:
- Martin Fowler - Refactoring
- Robert C. Martin - Clean Architecture
- Gang of Four - Design Patterns
- Miguel Grinberg - Flask Web Development

---

**Happy Learning! 🎓**
