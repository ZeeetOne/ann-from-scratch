# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ANN from Scratch is a professional neural network framework built from scratch using Python and NumPy. It features an interactive web interface for building, training, and testing artificial neural networks with custom architectures.

**Tech Stack:**
- Backend: Flask, NumPy, Pandas
- Frontend: HTML5, CSS3, JavaScript (ES6 Modules), Tailwind CSS + DaisyUI
- Build Tools: PostCSS for CSS compilation, npm for dependency management
- Architecture: Clean Architecture with SOLID principles
- Design Patterns: Strategy, Factory, Facade

## Quick Start

### First Time Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for CSS build)
npm install

# Build CSS
npm run build:css

# Run application
python run.py
```

Application will be available at http://localhost:5000

## Development Commands

### Running the Application

```bash
# Start development server (default: localhost:5000)
python run.py

# Windows batch file
run.bat

# Run with specific configuration
python run.py --config development    # Development mode (default, debug=True)
python run.py --config production     # Production mode (debug=False)
python run.py --config testing        # Testing mode
```

### Frontend Development

#### CSS Development
The project uses PostCSS to compile modular CSS files:

```bash
# Install dependencies (first time only)
npm install

# Build CSS for production (one-time build)
npm run build:css

# Watch CSS files and auto-rebuild on changes (recommended for development)
npm run watch:css
```

**CSS File Structure:**
- **Source files**: `frontend/static/css/src/`
  - `base/variables.css` - CSS custom properties (colors, spacing)
  - `base/reset.css` - CSS reset and base styles
  - `base/layout.css` - Page layout and structure
  - `base/utilities.css` - Animations, modals, utilities
  - `components/network-canvas.css` - Network visualization styles
  - `features/app-features.css` - Application-specific UI styles
  - `main.css` - Imports all CSS modules
- **Compiled output**: `frontend/static/css/dist/style.css` (minified, single file)

**Important**: The HTML templates reference `css/dist/style.css`, so you must run `npm run build:css` before the app will display correctly.

#### JavaScript Development
JavaScript uses native ES6 modules - no build step required:

- **Entry point**: `frontend/static/js/app.js` (loaded as `<script type="module">`)
- **Modules**: `frontend/static/js/modules/` - Feature-specific modules
- **Utilities**: `frontend/static/js/utils/` - Shared utilities (API client, formatters)
- **Config**: `frontend/static/js/config/constants.js` - Application constants

Simply edit `.js` files and refresh the browser. Modules are automatically loaded via ES6 import statements.

### Dependencies

**Python Dependencies** (requirements.txt):
```bash
pip install -r requirements.txt

# Includes: Flask, numpy, pandas
```

**Node.js Dependencies** (package.json):
```bash
npm install

# Includes: postcss, postcss-cli, postcss-import, autoprefixer, cssnano
```

## Project Structure

```
ann-from-scratch/
├── backend/                          # Python/Flask Backend
│   ├── core/                         # ⚙️ ML Algorithms (Pure Python/NumPy)
│   │   ├── neural_network.py         # Main NeuralNetwork class
│   │   ├── activation_functions.py   # Sigmoid, ReLU, Softmax, Linear, Threshold
│   │   ├── loss_functions.py         # MSE, Binary/Categorical Cross-Entropy
│   │   ├── optimizers.py             # GD, SGD, Momentum
│   │   └── __init__.py
│   │
│   ├── services/                     # 🔧 Business Logic Layer
│   │   ├── network_service.py        # Network building & management
│   │   ├── training_service.py       # Training orchestration & metrics
│   │   ├── data_service.py           # Data processing & validation
│   │   └── __init__.py
│   │
│   ├── api/                          # 🌐 REST API Layer
│   │   ├── routes/                   # API Endpoints
│   │   │   ├── network_routes.py     # /build_network, /network_info
│   │   │   ├── training_routes.py    # /train, /backpropagation, /update_weights
│   │   │   ├── prediction_routes.py  # /predict, /forward_pass
│   │   │   ├── example_routes.py     # /quick_start_binary, /quick_start_multiclass
│   │   │   └── __init__.py
│   │   ├── middleware/               # Error Handling
│   │   │   ├── error_handler.py
│   │   │   └── __init__.py
│   │   ├── app.py                    # Flask Application Factory
│   │   └── __init__.py
│   │
│   ├── utils/                        # 🛠️ Utilities
│   │   ├── validators.py             # Input validation
│   │   ├── data_processor.py         # Data transformations
│   │   └── __init__.py
│   │
│   ├── config.py                     # Configuration (Dev, Prod, Test)
│   └── __init__.py
│
├── frontend/                         # HTML/CSS/JavaScript Frontend
│   ├── static/
│   │   ├── css/
│   │   │   ├── src/                  # 📝 Source CSS (Modular)
│   │   │   │   ├── base/
│   │   │   │   │   ├── variables.css      # CSS custom properties
│   │   │   │   │   ├── reset.css          # CSS reset & base styles
│   │   │   │   │   ├── layout.css         # Page layout
│   │   │   │   │   └── utilities.css      # Animations & modals
│   │   │   │   ├── components/
│   │   │   │   │   └── network-canvas.css # Network visualization
│   │   │   │   ├── features/
│   │   │   │   │   └── app-features.css   # App-specific styles
│   │   │   │   └── main.css          # Imports all CSS modules
│   │   │   └── dist/                 # ⚡ Built CSS (PostCSS output)
│   │   │       └── style.css         # Compiled & minified
│   │   │
│   │   └── js/                       # 📦 JavaScript (ES6 Modules)
│   │       ├── modules/              # Feature Modules
│   │       │   └── network/
│   │       │       └── network-builder.js  # Interactive network builder
│   │       ├── utils/                # Shared Utilities
│   │       │   ├── api-client.js     # Centralized API communication
│   │       │   └── formatters.js     # Number formatting
│   │       ├── config/
│   │       │   └── constants.js      # App constants
│   │       ├── app.js                # 🚀 Main Entry Point (ES6 Module)
│   │       └── app.js.backup         # Backup of original file
│   │
│   └── templates/                    # Jinja2 Templates
│       ├── partials/
│       │   ├── head.html             # <head> section (CSS links)
│       │   ├── navbar.html           # Navigation bar
│       │   ├── hero.html             # Hero section
│       │   ├── steps.html            # Progress indicator
│       │   ├── footer.html           # Footer
│       │   ├── modal.html            # Modal dialogs
│       │   └── scripts.html          # JavaScript includes
│       └── index.html                # Main page (7 interactive tabs)
│
├── .gitignore                        # Git ignore rules
├── package.json                      # 📦 Node.js dependencies & build scripts
├── postcss.config.js                 # PostCSS configuration
├── requirements.txt                  # 🐍 Python dependencies
├── run.py                            # Application entry point
├── run.bat                           # Windows launcher
├── README.md                         # User documentation
└── CLAUDE.md                         # This file (AI assistant guidance)
```

## Architecture

### Backend Architecture (Clean Architecture)

The backend follows a layered architecture with clear separation of concerns:

```
Layer 1: Core (Pure Business Logic - No External Dependencies)
├── neural_network.py        # Main NeuralNetwork class
├── activation_functions.py  # Activation function implementations
├── loss_functions.py        # Loss function implementations
└── optimizers.py            # Optimizer implementations

Layer 2: Services (Business Logic - Depends on Core)
├── network_service.py       # Network building & management
├── training_service.py      # Training orchestration & metrics
└── data_service.py          # Data processing & validation

Layer 3: API (Web Layer - Depends on Services)
├── routes/                  # Flask blueprints for endpoints
├── middleware/              # Error handling
└── app.py                   # Flask application factory

Layer 4: Utils (Cross-cutting concerns)
├── validators.py            # Input validation
└── data_processor.py        # Data transformations
```

**Dependency Rule**: Inner layers (Core) never depend on outer layers (API). Dependencies point inward.

### Frontend Architecture (ES6 Modules)

```
app.js (Main Entry Point)
├── imports modules/network/network-builder.js
├── imports utils/api-client.js
├── imports utils/formatters.js
└── imports config/constants.js
```

**Module Loading**: Uses native ES6 modules with `<script type="module">`. No bundler required.

### Key Design Patterns

**1. Strategy Pattern** - Swappable algorithms:
```python
# Easily swap optimizers
network.train(X, y, optimizer='sgd')
network.train(X, y, optimizer='momentum')

# Easily swap loss functions
network.train(X, y, loss_function='mse')
network.train(X, y, loss_function='binary_crossentropy')
```

**2. Factory Pattern** - Create instances by name:
```python
activation = ActivationFactory.create('sigmoid')
loss = LossFunctionFactory.create('mse')
optimizer = OptimizerFactory.create('sgd', learning_rate=0.01)
```

**3. Facade Pattern** - Simplify complex operations:
```python
# Services provide simple interfaces to complex operations
results = TrainingService.train_network(network, X, y, config)
predictions = DataService.process_predictions(network, X, y)
```

**4. Application Factory Pattern** (Flask):
```python
# backend/api/app.py
def create_app(config_name='development'):
    app = Flask(__name__)
    # Configure and initialize
    return app
```

## API Endpoints

### Network Building
- `POST /build_network` - Build custom network from layer configuration
- `POST /quick_start_binary` - Load binary classification example (3-4-1 network)
- `POST /quick_start_multiclass` - Load multi-class example (3-4-2 network)
- `GET /network_info` - Get current network information

### Predictions
- `POST /predict` - Make predictions on a dataset
- `POST /forward_pass` - Get detailed layer-by-layer activations

### Training
- `POST /train` - Train network with dataset (full training loop)
- `POST /calculate_loss` - Calculate current loss (educational)
- `POST /backpropagation` - Calculate gradients (educational demonstration)
- `POST /update_weights` - Single weight update step (educational demonstration)

## Common Development Tasks

### Adding a New Activation Function

1. **Create class** in `backend/core/activation_functions.py`:
```python
class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2

    @property
    def name(self):
        return "tanh"
```

2. **Register with factory**:
```python
ActivationFactory.register('tanh', Tanh)
```

3. **Use in network**:
```python
network.add_layer(num_nodes=4, activation='tanh')
```

### Adding a New Optimizer

1. **Create class** in `backend/core/optimizers.py`:
```python
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0

    def update(self, params, gradients):
        self.t += 1
        # Implement Adam update logic here
        return updated_params

    @property
    def name(self):
        return "adam"
```

2. **Register with factory**:
```python
OptimizerFactory.register('adam', Adam)
```

3. **Use in training**:
```python
network.train(X, y, optimizer='adam', learning_rate=0.001)
```

### Adding a New Loss Function

1. **Create class** in `backend/core/loss_functions.py`:
```python
class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        self.delta = delta

    def calculate(self, y_true, y_pred):
        # Implement Huber loss calculation
        pass

    def derivative(self, y_true, y_pred):
        # Implement derivative
        pass

    @property
    def name(self):
        return "huber"
```

2. **Register with factory**:
```python
LossFunctionFactory.register('huber', HuberLoss)
```

3. **Use in training**:
```python
network.train(X, y, loss_function='huber')
```

## Important Implementation Details

### Custom Connections

This framework supports **custom connections** between nodes (not just fully connected layers). Each node can connect to a subset of nodes in the previous layer:

```python
# Example: Partial connections
network.set_connections(
    layer_idx=1,
    connections=[[0,1], [0,1]],  # Each node's input connections
    weights=[[0.5,-0.3], [-0.4,0.6]],  # Corresponding weights
    biases=[0.1, -0.2]
)
```

### Forward Pass Caching

The network caches both pre-activation (`z_values`) and post-activation (`layer_outputs`) during forward pass, which is essential for backpropagation:

```python
self.layer_z_values[i] = z  # Before activation
self.layer_outputs[i] = a   # After activation
```

### Backpropagation Implementation

Backpropagation uses the chain rule to compute gradients layer-by-layer in reverse:

1. **Output layer**: `δ = loss_derivative * activation_derivative`
2. **Hidden layers**: `δ = (W^T @ δ_next) * activation_derivative`
3. **Gradients**: `∂W = δ @ a_prev^T`, `∂b = δ`

The implementation handles custom connections (not just fully connected).

### Smart Weight Initialization

- **Xavier initialization** for sigmoid/tanh: Prevents vanishing gradients
- **He initialization** for ReLU: Prevents dying neurons

This was critical to fixing the "probability stuck at 0.300" issue.

## Configuration

### Environment Configurations

Located in `backend/config.py`:

- **DevelopmentConfig**: DEBUG=True, localhost:5000, verbose logging
- **ProductionConfig**: DEBUG=False, 0.0.0.0:80, minimal logging
- **TestingConfig**: DEBUG=True, TESTING=True, test settings

Switch configurations:
```bash
python run.py --config development
python run.py --config production
python run.py --config testing
```

## Frontend UI Components

### 7-Tab Interface

The web interface uses a tab-based workflow:

1. **Build Network** - Visual network builder with drag-and-drop connections
2. **Dataset** - Upload CSV or generate example data
3. **Forward Pass** - Run predictions and inspect layer activations
4. **Loss** - Calculate and view loss
5. **Epoch** - Run single epoch (forward → backprop → update)
6. **Train** - Automated training for multiple epochs
7. **Results** - View charts, metrics, confusion matrix

### Technologies Used

- **Tailwind CSS + DaisyUI**: Component framework
- **Chart.js**: Loss curve visualization
- **Font Awesome**: Icons
- **Vanilla JavaScript (ES6)**: No framework needed, modern modules

## Best Practices for AI Assistants

When working with this codebase:

1. **Backend Changes**: Maintain Clean Architecture - never have Core depend on Services or API
2. **Frontend CSS**: Edit source files in `frontend/static/css/src/`, then run `npm run build:css`
3. **Frontend JS**: Edit `.js` files directly, browser will reload modules automatically
4. **New Endpoints**: Add to appropriate routes file, register blueprint in `app.py`
5. **Testing Changes**: Always test in browser after modifications
6. **Documentation**: Update README.md and CLAUDE.md when adding major features

## Future Enhancements

Potential additions (not yet implemented):

- [ ] Adam optimizer
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] L1/L2 regularization
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] Model save/load functionality
- [ ] Additional activation functions (Tanh, Leaky ReLU, ELU)

---

**Last Updated**: After comprehensive restructuring to ES6 modules and PostCSS build system.
