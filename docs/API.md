# API Documentation

Complete API reference for ANN from Scratch v2.0.

## Base URL

```
http://localhost:5000
```

## Response Format

All API responses follow this consistent format:

**Success Response:**
```json
{
    "success": true,
    "message": "Operation successful",
    "data": { ... }
}
```

**Error Response:**
```json
{
    "success": false,
    "error": "Error message",
    "error_type": "ValueError"
}
```

## Endpoints

### Health Check

**GET** `/health`

Check if server is running.

**Response:**
```json
{
    "status": "healthy",
    "version": "2.0.0"
}
```

---

### Build Network

**POST** `/build_network`

Build a neural network from configuration.

**Request Body:**
```json
{
    "layers": [
        {"num_nodes": 2, "activation": "linear"},
        {"num_nodes": 3, "activation": "sigmoid"},
        {"num_nodes": 1, "activation": "sigmoid"}
    ],
    "connections": [
        {
            "layer_idx": 1,
            "connections": [[0,1], [0,1], [0,1]],
            "weights": [[0.5,-0.3], [-0.4,0.6], [0.2,-0.5]],
            "biases": [0.1, -0.2, 0.3]
        },
        {
            "layer_idx": 2,
            "connections": [[0,1,2]],
            "weights": [[0.8,-0.5,0.3]],
            "biases": [0.2]
        }
    ]
}
```

**Parameters:**
- `layers` (array, required): Layer configurations
  - `num_nodes` (int, required): Number of nodes in layer
  - `activation` (string, required): Activation function name
    - Options: `sigmoid`, `relu`, `linear`, `softmax`, `threshold`
- `connections` (array, required): Connection configurations
  - `layer_idx` (int, required): Target layer index (1-based)
  - `connections` (array, required): Connection indices per node
  - `weights` (array, required): Weight values per node
  - `biases` (array, optional): Bias values per node

**Response:**
```json
{
    "success": true,
    "message": "Network built successfully",
    "summary": "Network architecture summary...",
    "classification_type": "binary",
    "recommended_loss": "binary",
    "network_info": {
        "num_layers": 3,
        "layer_sizes": [2, 3, 1],
        "activations": ["linear", "sigmoid", "sigmoid"],
        "total_parameters": 13
    },
    "connections": [ ... ]
}
```

---

### Quick Start - Binary Classification

**POST** `/quick_start_binary`

Load example binary classification network (3-4-1).

**Response:**
```json
{
    "success": true,
    "message": "Binary Example: 3-4-1 network with Sigmoid",
    "example_dataset": "x1,x2,x3,y1\n0.875,0.8,0.9,1\n...",
    "summary": "Network architecture...",
    "classification_type": "binary",
    "recommended_loss": "binary",
    "layers": [...],
    "connections": [...],
    "network_info": {...}
}
```

---

### Quick Start - Multi-Class Classification

**POST** `/quick_start_multiclass`

Load example multi-class classification network (3-4-2).

**Response:**
```json
{
    "success": true,
    "message": "Multi-Class Example: 3-4-2 network with Softmax",
    "example_dataset": "x1,x2,x3,y1,y2\n0.417,0.556,0.818,1,0\n...",
    "summary": "Network architecture...",
    "classification_type": "multi-class",
    "recommended_loss": "categorical",
    "layers": [...],
    "connections": [...],
    "network_info": {...}
}
```

---

### Make Predictions

**POST** `/predict`

Make predictions on a dataset.

**Request Body:**
```json
{
    "dataset": "x1,x2,y\n0,0,0\n0,1,0\n1,0,0\n1,1,1",
    "loss_function": "binary",
    "threshold": 0.5
}
```

**Parameters:**
- `dataset` (string, required): CSV dataset with features and labels
- `loss_function` (string, optional): Loss function to use
  - Options: `binary`, `mse`, `categorical`
  - Default: `mse`
- `threshold` (float, optional): Classification threshold (0.0-1.0)
  - Default: 0.5
  - Only used for binary/multi-label classification

**Response:**
```json
{
    "success": true,
    "results": [
        {
            "index": 0,
            "features": [0, 0],
            "y_true": 0,
            "y_pred_prob": 0.234,
            "y_pred_class": "No",
            "correct": true
        },
        ...
    ],
    "loss": 0.171582,
    "accuracy": 0.75,
    "feature_names": ["x1", "x2"],
    "target_names": ["y"],
    "num_outputs": 1
}
```

---

### Forward Pass (Detailed)

**POST** `/forward_pass`

Get detailed layer-by-layer forward pass information.

**Request Body:**
```json
{
    "dataset": "x1,x2,y\n0,0,0\n0,1,0\n1,0,0\n1,1,1"
}
```

**Response:**
```json
{
    "success": true,
    "feature_names": ["x1", "x2"],
    "target_names": ["y"],
    "num_samples": 4,
    "samples": [
        {
            "sample_index": 0,
            "input": [0, 0],
            "target": [0],
            "prediction": [0.234],
            "layer_outputs": [
                {
                    "layer_index": 0,
                    "layer_type": "input",
                    "num_nodes": 2,
                    "activation_function": "linear",
                    "outputs": [[0, 0]]
                },
                {
                    "layer_index": 1,
                    "layer_type": "hidden",
                    "num_nodes": 2,
                    "activation_function": "sigmoid",
                    "nodes": [
                        {
                            "node_index": 0,
                            "bias": 0.1,
                            "weighted_sum": 0.1,
                            "activated_value": 0.525,
                            "input_contributions": [
                                {"from_node": 0, "input_value": 0, "weight": 0.5, "contribution": 0},
                                {"from_node": 1, "input_value": 0, "weight": -0.3, "contribution": 0}
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
}
```

---

### Train Network

**POST** `/train`

Train the neural network.

**Request Body:**
```json
{
    "dataset": "x1,x2,y\n0,0,0\n0,1,0\n1,0,0\n1,1,1",
    "epochs": 1000,
    "learning_rate": 0.5,
    "optimizer": "gd",
    "loss_function": "mse",
    "batch_size": null
}
```

**Parameters:**
- `dataset` (string, required): CSV dataset
- `epochs` (int, optional): Number of training epochs (1-100000)
  - Default: 100
- `learning_rate` (float, optional): Learning rate (0.0001-100.0)
  - Default: 0.01
- `optimizer` (string, optional): Optimizer algorithm
  - Options: `gd` (Gradient Descent), `sgd` (Stochastic GD), `momentum`
  - Default: `gd`
- `loss_function` (string, optional): Loss function
  - Options: `binary`, `mse`, `categorical`
  - Default: `mse`
- `batch_size` (int, optional): Mini-batch size for SGD
  - Default: Full batch

**Response:**
```json
{
    "success": true,
    "message": "Training completed: 1000 epochs",
    "history": {
        "epochs": [1, 2, ..., 1000],
        "loss": [0.308, 0.295, ..., 0.171]
    },
    "final_loss": 0.171582,
    "accuracy": 0.75,
    "num_outputs": 1,
    "evaluation": {
        "confusion_matrix": {
            "0": {"0": 3, "1": 0},
            "1": {"0": 1, "1": 0}
        },
        "metrics_per_class": {
            "0": {
                "precision": 0.75,
                "recall": 1.0,
                "f1_score": 0.857,
                "support": 3,
                "true_positives": 3,
                "false_positives": 1,
                "false_negatives": 0,
                "true_negatives": 0
            },
            "1": {...}
        },
        "macro_avg": {
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.429
        }
    },
    "updated_weights": {...},
    "updated_biases": {...},
    "predictions": [...]
}
```

---

### Calculate Loss

**POST** `/calculate_loss`

Calculate loss for current network on dataset.

**Request Body:**
```json
{
    "dataset": "x1,x2,y\n0,0,0\n0,1,0\n1,0,0\n1,1,1",
    "loss_function": "mse"
}
```

**Response:**
```json
{
    "success": true,
    "loss_function": "mse",
    "total_loss": 0.171582,
    "sample_losses": [
        {
            "sample_index": 0,
            "loss": 0.055,
            "y_true": [0],
            "y_pred": [0.234]
        },
        ...
    ],
    "num_samples": 4
}
```

---

### Backpropagation (Demo)

**POST** `/backpropagation`

Calculate gradients using backpropagation (demonstration).

**Request Body:**
```json
{
    "dataset": "x1,x2,y\n0,0,0\n0,1,0\n1,0,0\n1,1,1",
    "loss_function": "mse"
}
```

**Response:**
```json
{
    "success": true,
    "loss_function": "mse",
    "sample_index": 0,
    "layers": [
        {
            "layer_index": 1,
            "layer_type": "hidden",
            "gradients": [
                {
                    "weight_gradients": [0.0123, -0.0087],
                    "bias_gradient": 0.0156
                },
                ...
            ]
        },
        ...
    ]
}
```

---

### Update Weights (Single Step)

**POST** `/update_weights`

Perform single weight update step (demonstration).

**Request Body:**
```json
{
    "dataset": "x1,x2,y\n0,0,0\n0,1,0\n1,0,0\n1,1,1",
    "loss_function": "mse",
    "learning_rate": 0.01,
    "optimizer": "gd"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Weights updated using GD (1 epoch completed)",
    "optimizer": "gd",
    "learning_rate": 0.01,
    "loss_before": 0.308552,
    "loss_after": 0.305123,
    "loss_reduction": 0.003429,
    "old_weights": {...},
    "old_biases": {...},
    "new_weights": {...},
    "new_biases": {...},
    "weight_changes": {...},
    "bias_changes": {...}
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request (validation error, invalid parameters) |
| 404  | Not Found (invalid endpoint) |
| 500  | Internal Server Error |

## Common Errors

### ValidationError
```json
{
    "success": false,
    "error": "epochs must be between 1 and 100000",
    "error_type": "ValidationError"
}
```

### ValueError
```json
{
    "success": false,
    "error": "No network has been built. Please build a network first.",
    "error_type": "ValueError"
}
```

### KeyError
```json
{
    "success": false,
    "error": "Missing required field: 'dataset'",
    "error_type": "KeyError"
}
```

## Usage Examples

### Python (requests)

```python
import requests

# Build network
payload = {
    "layers": [
        {"num_nodes": 2, "activation": "linear"},
        {"num_nodes": 2, "activation": "sigmoid"},
        {"num_nodes": 1, "activation": "sigmoid"}
    ],
    "connections": [...]
}

response = requests.post("http://localhost:5000/build_network", json=payload)
print(response.json())

# Train
train_payload = {
    "dataset": "x1,x2,y\n...",
    "epochs": 1000,
    "learning_rate": 0.5,
    "optimizer": "gd"
}

response = requests.post("http://localhost:5000/train", json=train_payload)
print(response.json())
```

### JavaScript (fetch)

```javascript
// Build network
const payload = {
    layers: [
        {num_nodes: 2, activation: "linear"},
        {num_nodes: 2, activation: "sigmoid"},
        {num_nodes: 1, activation: "sigmoid"}
    ],
    connections: [...]
};

fetch('http://localhost:5000/build_network', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
})
.then(response => response.json())
.then(data => console.log(data));
```

### curl

```bash
# Build network
curl -X POST http://localhost:5000/build_network \
  -H "Content-Type: application/json" \
  -d '{"layers": [...], "connections": [...]}'

# Quick start
curl -X POST http://localhost:5000/quick_start_binary

# Train
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset": "...", "epochs": 1000, "learning_rate": 0.5}'
```

## Rate Limiting

Currently no rate limiting. Consider implementing for production use.

## Authentication

Currently no authentication. Consider implementing for production use.

## Versioning

Current version: **v2.0.0**

API version is returned in `/health` endpoint.
