# Architecture Documentation

## Overview

ANN from Scratch v2.0 follows **Clean Architecture** principles with clear separation of concerns. The refactored codebase is modular, scalable, and follows industry best practices.

## Design Principles

### SOLID Principles

1. **Single Responsibility Principle (SRP)**
   - Each class has one reason to change
   - Example: `Sigmoid` class only handles sigmoid activation

2. **Open/Closed Principle (OCP)**
   - Open for extension, closed for modification
   - Example: Add new activation functions by extending `ActivationFunction` base class

3. **Liskov Substitution Principle (LSP)**
   - Derived classes can substitute base classes
   - Example: Any `ActivationFunction` can be used interchangeably

4. **Interface Segregation Principle (ISP)**
   - Small, focused interfaces
   - Example: `ActivationFunction` has only `forward()` and `derivative()`

5. **Dependency Inversion Principle (DIP)**
   - Depend on abstractions, not concretions
   - Example: `NeuralNetwork` depends on `ActivationFunction` interface

### Design Patterns

1. **Strategy Pattern**
   - Used for: Activation functions, Loss functions, Optimizers
   - Benefit: Easy to switch algorithms at runtime

2. **Factory Pattern**
   - Used for: Creating activation, loss, and optimizer instances
   - Benefit: Centralized object creation, easy to extend

3. **Facade Pattern**
   - Used for: Service layer (NetworkService, TrainingService)
   - Benefit: Simplifies complex subsystem interactions

4. **Application Factory Pattern**
   - Used for: Flask app creation
   - Benefit: Testability, multiple configurations

## Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│           (HTML, CSS, JavaScript - Frontend)             │
└─────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────┐
│                       API Layer                          │
│        (Flask Routes, Middleware, Error Handling)        │
│                                                           │
│  - network_routes.py   - Build network, get info         │
│  - training_routes.py  - Training, backprop, gradients   │
│  - prediction_routes.py - Predictions, forward pass      │
│  - example_routes.py   - Quick start examples            │
└─────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────┐
│                    Services Layer                        │
│            (Business Logic, Orchestration)               │
│                                                           │
│  - NetworkService      - Network management              │
│  - TrainingService     - Training orchestration          │
│  - DataService         - Data processing                 │
└─────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────┐
│                      Core Layer                          │
│          (ML Algorithms, Neural Network Core)            │
│                                                           │
│  - NeuralNetwork       - Main network class              │
│  - ActivationFunction  - Activation implementations      │
│  - LossFunction        - Loss implementations            │
│  - Optimizer           - Optimization algorithms         │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ann-from-scratch/
├── backend/
│   ├── api/                    # API Layer
│   │   ├── routes/             # Endpoint routes
│   │   ├── middleware/         # Error handling, CORS
│   │   └── app.py              # Flask app factory
│   ├── core/                   # Core ML Layer
│   │   ├── activation_functions.py
│   │   ├── loss_functions.py
│   │   ├── optimizers.py
│   │   └── neural_network.py
│   ├── services/               # Services Layer
│   │   ├── network_service.py
│   │   ├── training_service.py
│   │   └── data_service.py
│   ├── utils/                  # Utilities
│   │   ├── validators.py
│   │   └── data_processor.py
│   └── config.py               # Configuration
├── frontend/                   # Presentation Layer
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
├── tests/                      # Test Suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── examples/                   # Usage examples
├── docs/                       # Documentation
└── run.py                      # Entry point
```

## Component Responsibilities

### Core Layer

**Activation Functions (`activation_functions.py`)**
- Implements various activation functions (Sigmoid, ReLU, Linear, Softmax, Threshold)
- Each activation is a separate class implementing `ActivationFunction` interface
- Factory pattern for creating activation instances

**Loss Functions (`loss_functions.py`)**
- Implements loss functions (MSE, Binary Cross-Entropy, Categorical Cross-Entropy)
- Each loss is a separate class implementing `LossFunction` interface
- Includes both forward (calculate) and backward (derivative) methods

**Optimizers (`optimizers.py`)**
- Implements optimization algorithms (GD, SGD, Momentum)
- Each optimizer is a separate class implementing `Optimizer` interface
- Handles parameter updates during training

**Neural Network (`neural_network.py`)**
- Main network class orchestrating everything
- Manages layers, connections, weights, biases
- Implements forward propagation
- Implements backpropagation
- Provides training loop

### Services Layer

**Network Service**
- Building networks from configuration
- Managing network state
- Creating example networks
- Extracting network information

**Training Service**
- Training orchestration
- Calculating evaluation metrics (accuracy, precision, recall, F1)
- Computing gradients
- Tracking training history

**Data Service**
- Parsing CSV datasets
- Validating data
- Processing predictions
- Generating forward pass details

### API Layer

**Routes**
- Modular route definitions
- Each route file handles related endpoints
- Uses Blueprint pattern for organization

**Middleware**
- Centralized error handling
- CORS configuration
- Request/response hooks

### Utils

**Validators**
- Request validation before processing
- Fail-fast error detection
- Consistent error messages

**Data Processor**
- Response formatting
- Data type conversions
- Safe type casting

## Data Flow

### 1. Building Network

```
User Request → API Route → Validator → NetworkService → NeuralNetwork → Response
```

### 2. Making Predictions

```
User Request + Dataset → API Route → Validator → DataService (parse)
    → NeuralNetwork.forward() → DataService (format) → Response
```

### 3. Training

```
User Request + Config → API Route → Validator → DataService (parse)
    → TrainingService.train() → NeuralNetwork.train()
        → Loop: Forward → Loss → Backward → Update
    → TrainingService (metrics) → Response
```

## Extension Points

### Adding New Activation Function

```python
from backend.core import ActivationFunction

class NewActivation(ActivationFunction):
    def forward(self, x):
        # Implementation
        pass

    def derivative(self, x):
        # Implementation
        pass

    @property
    def name(self):
        return "new_activation"

# Register with factory
ActivationFactory.register('new_activation', NewActivation)
```

### Adding New Loss Function

```python
from backend.core import LossFunction

class NewLoss(LossFunction):
    def calculate(self, y_true, y_pred):
        # Implementation
        pass

    def derivative(self, y_true, y_pred):
        # Implementation
        pass

    @property
    def name(self):
        return "new_loss"

# Register with factory
LossFunctionFactory.register('new_loss', NewLoss)
```

### Adding New Optimizer

```python
from backend.core import Optimizer

class NewOptimizer(Optimizer):
    def update(self, params, gradients):
        # Implementation
        return updated_params

    @property
    def name(self):
        return "new_optimizer"

# Register with factory
OptimizerFactory.register('new_optimizer', NewOptimizer)
```

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies
- Fast execution
- High coverage

### Integration Tests
- Test complete workflows
- End-to-end scenarios
- Verify correctness (AND gate, multi-class classification)
- Performance validation

### Test Categories

1. **Activation Functions** - Forward, backward, edge cases
2. **Loss Functions** - Calculation, derivatives, edge cases
3. **Optimizers** - Parameter updates, convergence
4. **Neural Network** - Forward pass, backward pass, training
5. **Services** - Business logic, data processing
6. **API** - Endpoints, validation, error handling

## Configuration

Environment-based configuration:
- `development` - Debug mode, verbose logging
- `production` - Optimized, minimal logging
- `testing` - Test-specific settings

## Security Considerations

1. **Input Validation** - All inputs validated before processing
2. **Error Handling** - Sensitive information not exposed in production
3. **CORS** - Configurable CORS settings
4. **Rate Limiting** - (Future enhancement)

## Performance Optimizations

1. **NumPy Vectorization** - All operations vectorized
2. **Batch Processing** - Mini-batch support in training
3. **Memory Efficiency** - Minimal memory footprint
4. **Lazy Evaluation** - Computations only when needed

## Future Enhancements

1. **Advanced Optimizers** - Adam, RMSprop, AdaGrad
2. **Regularization** - L1, L2, Dropout
3. **Learning Rate Scheduling** - Step decay, exponential decay
4. **Early Stopping** - Prevent overfitting
5. **Model Serialization** - Save/load models
6. **Batch Normalization** - Faster convergence
7. **More Activations** - Tanh, Leaky ReLU, ELU
8. **Visualization** - Network architecture graphs, training curves
9. **REST API** - Full RESTful API with authentication
10. **Database Integration** - Store models and training history

## References

- Martin Fowler - Refactoring
- Robert C. Martin - Clean Architecture, SOLID Principles
- Gang of Four - Design Patterns
- Miguel Grinberg - Flask Web Development
