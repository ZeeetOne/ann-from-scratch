"""
Neural Network Module

Implements the main Neural Network class with forward/backward propagation.
Uses composition and dependency injection for flexibility and testability.

Author: ANN from Scratch Team
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from .activation_functions import ActivationFunction, ActivationFactory
from .loss_functions import LossFunction, LossFunctionFactory
from .optimizers import Optimizer, OptimizerFactory


class NeuralNetwork:
    """
    Custom Neural Network with flexible architecture.

    Supports:
    - Custom connections between nodes
    - Multiple activation functions per layer
    - Various loss functions and optimizers
    - Backpropagation training

    Design Principles:
    - Composition over Inheritance
    - Dependency Injection for flexibility
    - Single Responsibility Principle
    """

    def __init__(self):
        """Initialize empty neural network"""
        # Network architecture
        self.layers: List[int] = []  # Number of nodes per layer
        self.activations: List[ActivationFunction] = []  # Activation per layer

        # Network parameters
        self.connections: List[List[List[int]]] = []  # Connection indices
        self.weights: List[List[List[float]]] = []  # Weight values
        self.biases: List[List[float]] = []  # Bias values

        # Forward pass cache (for backpropagation)
        self.layer_outputs: List[Optional[np.ndarray]] = []  # After activation
        self.layer_z_values: List[Optional[np.ndarray]] = []  # Before activation

        # Training history
        self.training_history: Dict[str, List] = {
            'loss': [],
            'epoch': []
        }

    def add_layer(
        self,
        num_nodes: int,
        activation: Union[str, ActivationFunction] = 'sigmoid'
    ):
        """
        Add a layer to the network

        Args:
            num_nodes: Number of nodes in this layer
            activation: Activation function (name or instance)

        Raises:
            ValueError: If num_nodes is not positive
        """
        if num_nodes <= 0:
            raise ValueError("Number of nodes must be positive")

        # Store layer size
        self.layers.append(num_nodes)

        # Store activation function
        if isinstance(activation, str):
            activation_fn = ActivationFactory.create(activation)
        elif isinstance(activation, ActivationFunction):
            activation_fn = activation
        else:
            raise TypeError("activation must be string or ActivationFunction")

        self.activations.append(activation_fn)

        # Initialize output caches
        self.layer_outputs.append(None)
        self.layer_z_values.append(None)

    def set_connections(
        self,
        layer_idx: int,
        connections: List[List[int]],
        weights: List[List[float]],
        biases: Optional[List[float]] = None
    ):
        """
        Set connections between layers

        Args:
            layer_idx: Target layer index (connections FROM previous TO this)
            connections: List where connections[i] = indices of previous layer nodes
            weights: List where weights[i] = weight values for node i
            biases: Bias values for each node (default: zeros)

        Raises:
            ValueError: If layer_idx is invalid or data is inconsistent
        """
        if layer_idx <= 0 or layer_idx >= len(self.layers):
            raise ValueError(
                f"Invalid layer_idx: {layer_idx}. "
                f"Must be between 1 and {len(self.layers) - 1}"
            )

        if len(connections) != self.layers[layer_idx]:
            raise ValueError(
                f"Number of connection lists ({len(connections)}) "
                f"doesn't match layer size ({self.layers[layer_idx]})"
            )

        if len(weights) != len(connections):
            raise ValueError("Length of weights must match connections")

        # Validate each node's connections
        for i, (node_conns, node_weights) in enumerate(zip(connections, weights)):
            if len(node_conns) != len(node_weights):
                raise ValueError(
                    f"Node {i}: number of connections ({len(node_conns)}) "
                    f"doesn't match number of weights ({len(node_weights)})"
                )

        # Ensure space in lists
        while len(self.connections) <= layer_idx:
            self.connections.append([])
            self.weights.append([])
            self.biases.append([])

        # Store connections and weights
        self.connections[layer_idx] = connections
        self.weights[layer_idx] = weights

        # Store biases (default to zeros)
        if biases is None:
            biases = [0.0] * len(connections)

        if len(biases) != len(connections):
            raise ValueError("Length of biases must match connections")

        self.biases[layer_idx] = biases

    def set_full_connections(
        self,
        layer_idx: int,
        weight_matrix: np.ndarray,
        biases: Optional[np.ndarray] = None
    ):
        """
        Set fully connected weights between layers (convenience method)

        Args:
            layer_idx: Target layer index
            weight_matrix: Matrix of shape (num_nodes_current, num_nodes_previous)
            biases: Bias array for current layer

        Raises:
            ValueError: If dimensions don't match
        """
        if layer_idx <= 0 or layer_idx >= len(self.layers):
            raise ValueError(f"Invalid layer_idx: {layer_idx}")

        num_current = self.layers[layer_idx]
        num_previous = self.layers[layer_idx - 1]

        if weight_matrix.shape != (num_current, num_previous):
            raise ValueError(
                f"Weight matrix shape {weight_matrix.shape} "
                f"doesn't match expected ({num_current}, {num_previous})"
            )

        # Convert matrix to connection list format
        connections = []
        weights = []

        for i in range(num_current):
            # Each node connects to all nodes in previous layer
            connections.append(list(range(num_previous)))
            weights.append(weight_matrix[i].tolist())

        if biases is None:
            biases = np.zeros(num_current)

        self.set_connections(layer_idx, connections, weights, biases.tolist())

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation through network

        Args:
            inputs: Input array of shape (batch_size, input_features)

        Returns:
            Output predictions of shape (batch_size, output_features)

        Raises:
            ValueError: If input shape doesn't match network
        """
        # Ensure 2D array (batch_size, features)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        if inputs.shape[1] != self.layers[0]:
            raise ValueError(
                f"Input features ({inputs.shape[1]}) "
                f"don't match network input layer ({self.layers[0]})"
            )

        batch_size = inputs.shape[0]

        # Input layer (no activation)
        self.layer_outputs[0] = inputs
        self.layer_z_values[0] = inputs

        # Propagate through each layer
        for layer_idx in range(1, len(self.layers)):
            num_nodes = self.layers[layer_idx]
            prev_output = self.layer_outputs[layer_idx - 1]

            # Initialize z values for this layer
            layer_z = np.zeros((batch_size, num_nodes))

            # Calculate weighted sum for each node
            for node_idx in range(num_nodes):
                if (layer_idx < len(self.connections) and
                    node_idx < len(self.connections[layer_idx])):

                    node_connections = self.connections[layer_idx][node_idx]
                    node_weights = self.weights[layer_idx][node_idx]
                    node_bias = self.biases[layer_idx][node_idx]

                    # z = bias + Σ(weight * input)
                    weighted_sum = node_bias
                    for conn_idx, weight in zip(node_connections, node_weights):
                        if conn_idx < prev_output.shape[1]:
                            weighted_sum += prev_output[:, conn_idx] * weight

                    layer_z[:, node_idx] = weighted_sum

            # Store z values
            self.layer_z_values[layer_idx] = layer_z

            # Apply activation function
            activation_fn = self.activations[layer_idx]

            # Softmax needs special handling (applied to entire layer)
            if activation_fn.name == 'softmax':
                layer_output = activation_fn.forward(layer_z)
            else:
                layer_output = activation_fn.forward(layer_z)

            self.layer_outputs[layer_idx] = layer_output

        # Return output of last layer
        return self.layer_outputs[-1]

    def backward(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        loss_function: LossFunction
    ) -> Tuple[List, List]:
        """
        Backpropagation to calculate gradients

        Args:
            y_true: True labels
            y_pred: Predicted values
            loss_function: Loss function instance

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        batch_size = y_true.shape[0]

        # Initialize gradient storage
        weight_gradients = [[] for _ in range(len(self.layers))]
        bias_gradients = [[] for _ in range(len(self.layers))]

        # Calculate output layer error (delta)
        output_layer_idx = len(self.layers) - 1

        # Get loss derivative: dL/da
        dL_da = loss_function.derivative(y_true, y_pred)

        # Get activation derivative: da/dz
        activation_fn = self.activations[output_layer_idx]
        da_dz = activation_fn.derivative(self.layer_z_values[output_layer_idx])

        # delta = dL/dz = dL/da * da/dz
        delta = dL_da * da_dz

        # Store deltas for each layer (working backwards)
        deltas = [None] * len(self.layers)
        deltas[output_layer_idx] = delta

        # Backpropagate through hidden layers
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            activation_fn = self.activations[layer_idx]
            da_dz = activation_fn.derivative(self.layer_z_values[layer_idx])

            next_layer_idx = layer_idx + 1
            num_nodes = self.layers[layer_idx]
            layer_delta = np.zeros((batch_size, num_nodes))

            # For each node in current layer
            for node_idx in range(num_nodes):
                weighted_delta_sum = 0

                # Sum weighted deltas from next layer
                if next_layer_idx < len(self.connections):
                    for next_node_idx in range(len(self.connections[next_layer_idx])):
                        node_connections = self.connections[next_layer_idx][next_node_idx]
                        node_weights = self.weights[next_layer_idx][next_node_idx]

                        # If this node connects to next node
                        if node_idx in node_connections:
                            weight_idx = node_connections.index(node_idx)
                            weight = node_weights[weight_idx]
                            weighted_delta_sum += deltas[next_layer_idx][:, next_node_idx] * weight

                layer_delta[:, node_idx] = weighted_delta_sum

            deltas[layer_idx] = layer_delta * da_dz

        # Calculate gradients for weights and biases
        for layer_idx in range(1, len(self.layers)):
            num_nodes = self.layers[layer_idx]
            prev_output = self.layer_outputs[layer_idx - 1]

            layer_weight_gradients = []
            layer_bias_gradients = []

            for node_idx in range(num_nodes):
                if (layer_idx < len(self.connections) and
                    node_idx < len(self.connections[layer_idx])):

                    node_connections = self.connections[layer_idx][node_idx]
                    node_delta = deltas[layer_idx][:, node_idx]

                    # Weight gradients: dL/dw = delta * input
                    node_weight_gradients = []
                    for conn_idx in node_connections:
                        if conn_idx < prev_output.shape[1]:
                            dL_dw = np.mean(node_delta * prev_output[:, conn_idx])
                            node_weight_gradients.append(dL_dw)
                        else:
                            node_weight_gradients.append(0.0)

                    layer_weight_gradients.append(node_weight_gradients)

                    # Bias gradient: dL/db = delta
                    dL_db = np.mean(node_delta)
                    layer_bias_gradients.append(dL_db)

            weight_gradients[layer_idx] = layer_weight_gradients
            bias_gradients[layer_idx] = layer_bias_gradients

        return weight_gradients, bias_gradients

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        optimizer: Union[str, Optimizer] = 'gd',
        loss_function: Union[str, LossFunction] = 'mse',
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the neural network

        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            optimizer: Optimizer (name or instance)
            loss_function: Loss function (name or instance)
            batch_size: Batch size for SGD (None = full batch)
            verbose: Print training progress

        Returns:
            Training history dictionary

        Raises:
            ValueError: If parameters are invalid
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        # Create optimizer if string
        if isinstance(optimizer, str):
            opt = OptimizerFactory.create(optimizer, learning_rate=learning_rate)
        elif isinstance(optimizer, Optimizer):
            opt = optimizer
            opt.set_learning_rate(learning_rate)
        else:
            raise TypeError("optimizer must be string or Optimizer")

        # Create loss function if string
        if isinstance(loss_function, str):
            loss_fn = LossFunctionFactory.create(loss_function)
        elif isinstance(loss_function, LossFunction):
            loss_fn = loss_function
        else:
            raise TypeError("loss_function must be string or LossFunction")

        # Default batch size = full batch
        if batch_size is None:
            batch_size = X.shape[0]

        # Reset training history
        self.training_history = {'loss': [], 'epoch': []}

        # Training loop
        for epoch in range(epochs):
            # Shuffle data for SGD
            if opt.name == 'sgd':
                indices = np.random.permutation(X.shape[0])
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Calculate loss
                batch_loss = loss_fn.calculate(y_batch, y_pred)
                epoch_loss += batch_loss
                num_batches += 1

                # Backward pass
                weight_grads, bias_grads = self.backward(y_batch, y_pred, loss_fn)

                # Update weights and biases
                for layer_idx in range(1, len(self.layers)):
                    if layer_idx < len(self.weights) and layer_idx < len(weight_grads):
                        for node_idx in range(len(self.weights[layer_idx])):
                            if node_idx < len(weight_grads[layer_idx]):
                                # Update weights
                                for w_idx in range(len(self.weights[layer_idx][node_idx])):
                                    if w_idx < len(weight_grads[layer_idx][node_idx]):
                                        self.weights[layer_idx][node_idx][w_idx] = opt.update(
                                            self.weights[layer_idx][node_idx][w_idx],
                                            weight_grads[layer_idx][node_idx][w_idx]
                                        )

                                # Update biases
                                if node_idx < len(bias_grads[layer_idx]):
                                    self.biases[layer_idx][node_idx] = opt.update(
                                        self.biases[layer_idx][node_idx],
                                        bias_grads[layer_idx][node_idx]
                                    )

            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            self.training_history['loss'].append(avg_loss)
            self.training_history['epoch'].append(epoch + 1)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return self.training_history

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data

        Args:
            X: Input data
            threshold: Threshold for binary/multi-label (ignored for multi-class)

        Returns:
            Tuple of (predicted_classes, predicted_probabilities)
        """
        probabilities = self.forward(X)
        classification_type = self.get_classification_type()

        if classification_type == 'multi-class':
            # Multi-class: use argmax, convert to one-hot
            predicted_classes = np.zeros_like(probabilities)
            max_indices = np.argmax(probabilities, axis=1)
            predicted_classes[np.arange(len(predicted_classes)), max_indices] = 1
        else:
            # Binary or multi-label: use threshold
            predicted_classes = (probabilities >= threshold).astype(int)

        return predicted_classes, probabilities

    def get_classification_type(self) -> str:
        """
        Determine classification type based on network architecture

        Returns:
            'binary', 'multi-label', or 'multi-class'
        """
        num_outputs = self.layers[-1]
        output_activation = self.activations[-1].name

        if num_outputs == 1:
            return 'binary'
        elif output_activation == 'softmax':
            return 'multi-class'
        else:
            return 'multi-label'

    def get_recommended_loss(self) -> str:
        """Get recommended loss function based on classification type"""
        classification_type = self.get_classification_type()

        if classification_type == 'binary':
            return 'binary'
        elif classification_type == 'multi-class':
            return 'categorical'
        else:
            return 'binary'

    def get_architecture_summary(self) -> str:
        """Get a summary of the network architecture"""
        summary = "Neural Network Architecture:\n"
        summary += "=" * 50 + "\n"

        for i, (num_nodes, activation) in enumerate(zip(self.layers, self.activations)):
            layer_type = "Input" if i == 0 else "Hidden" if i < len(self.layers) - 1 else "Output"
            summary += f"Layer {i} ({layer_type}): {num_nodes} nodes, Activation: {activation.name}\n"

            if i > 0 and i < len(self.connections):
                summary += f"  Connections from Layer {i-1}:\n"
                for node_idx in range(min(len(self.connections[i]), num_nodes)):
                    conns = self.connections[i][node_idx]
                    weights = self.weights[i][node_idx]
                    bias = self.biases[i][node_idx]
                    summary += f"    Node {node_idx}: connections={conns}, weights={weights}, bias={bias:.3f}\n"

        summary += "=" * 50
        return summary
