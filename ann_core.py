"""
ANN from Scratch - Core Library
Implements basic Artificial Neural Network with customizable architecture
"""

import numpy as np
from typing import List, Dict, Tuple, Callable


class ActivationFunctions:
    """Collection of activation functions"""

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)

    @staticmethod
    def threshold(x, threshold=0.5):
        """Threshold activation function"""
        return (x > threshold).astype(float)

    @staticmethod
    def linear(x):
        """Linear activation (no activation)"""
        return x

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid derivative"""
        sig = ActivationFunctions.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def relu_derivative(x):
        """ReLU derivative"""
        return (x > 0).astype(float)

    @staticmethod
    def linear_derivative(x):
        """Linear derivative"""
        return np.ones_like(x)

    @staticmethod
    def get_activation(name: str) -> Callable:
        """Get activation function by name"""
        activations = {
            'sigmoid': ActivationFunctions.sigmoid,
            'relu': ActivationFunctions.relu,
            'threshold': ActivationFunctions.threshold,
            'linear': ActivationFunctions.linear,
            'softmax': ActivationFunctions.softmax
        }
        return activations.get(name.lower(), ActivationFunctions.sigmoid)

    @staticmethod
    def get_activation_derivative(name: str) -> Callable:
        """Get activation derivative by name"""
        derivatives = {
            'sigmoid': ActivationFunctions.sigmoid_derivative,
            'relu': ActivationFunctions.relu_derivative,
            'threshold': ActivationFunctions.linear_derivative,  # Not differentiable, use linear
            'linear': ActivationFunctions.linear_derivative,
            'softmax': ActivationFunctions.linear_derivative  # Softmax derivative handled separately
        }
        return derivatives.get(name.lower(), ActivationFunctions.sigmoid_derivative)


class LossFunctions:
    """Collection of loss functions"""

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """Binary Cross-Entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """Categorical Cross-Entropy loss (for multi-class with softmax)"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # y_true is one-hot encoded, so we only sum over true class
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        """Binary Cross-Entropy derivative"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

    @staticmethod
    def mse_derivative(y_true, y_pred):
        """MSE derivative"""
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred):
        """
        Categorical Cross-Entropy derivative (for softmax output)
        When using softmax + CCE together, derivative simplifies to: y_pred - y_true
        This is a special property of softmax + CCE combination
        """
        return (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def get_loss(name: str) -> Callable:
        """Get loss function by name"""
        losses = {
            'binary': LossFunctions.binary_cross_entropy,
            'mse': LossFunctions.mse,
            'categorical': LossFunctions.categorical_cross_entropy
        }
        return losses.get(name.lower(), LossFunctions.mse)

    @staticmethod
    def get_loss_derivative(name: str) -> Callable:
        """Get loss derivative by name"""
        derivatives = {
            'binary': LossFunctions.binary_cross_entropy_derivative,
            'mse': LossFunctions.mse_derivative,
            'categorical': LossFunctions.categorical_cross_entropy_derivative
        }
        return derivatives.get(name.lower(), LossFunctions.mse_derivative)


class Optimizer:
    """Base class for optimizers"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        """Update weights based on gradients"""
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Batch Gradient Descent optimizer"""

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def update(self, weights, gradients):
        """Update weights using gradient descent"""
        return weights - self.learning_rate * gradients


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer (with mini-batch support)"""

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def update(self, weights, gradients):
        """Update weights using SGD"""
        return weights - self.learning_rate * gradients


class NeuralNetwork:
    """
    Custom Neural Network with user-defined architecture
    Supports custom connections, weights, and activation functions per layer
    """

    def __init__(self):
        self.layers = []
        self.connections = []
        self.weights = []
        self.biases = []
        self.activations = []
        self.layer_outputs = []
        self.layer_z_values = []  # Store pre-activation values for backprop
        self.training_history = {
            'loss': [],
            'epoch': []
        }

    def add_layer(self, num_nodes: int, activation: str = 'sigmoid'):
        """Add a layer to the network"""
        self.layers.append(num_nodes)
        self.activations.append(activation)
        self.layer_outputs.append(None)
        self.layer_z_values.append(None)

    def set_connections(self, layer_idx: int, connections: List[List[int]], weights: List[List[float]], biases: List[float] = None):
        """
        Set connections between layers

        Args:
            layer_idx: Index of the target layer (connections FROM previous layer TO this layer)
            connections: List of lists, where connections[i] contains indices of nodes in previous layer that connect to node i
            weights: List of lists, where weights[i] contains weights for connections to node i
            biases: List of bias values for each node in this layer
        """
        if layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        # Ensure we have enough space in our lists
        while len(self.connections) < layer_idx + 1:
            self.connections.append([])
            self.weights.append([])
            self.biases.append([])

        self.connections[layer_idx] = connections
        self.weights[layer_idx] = weights

        # Set biases or default to 0
        if biases is None:
            biases = [0.0] * len(connections)
        self.biases[layer_idx] = biases

    def set_full_connections(self, layer_idx: int, weight_matrix: np.ndarray, biases: np.ndarray = None):
        """
        Set fully connected weights between layers

        Args:
            layer_idx: Index of the target layer
            weight_matrix: Matrix of shape (num_nodes_current, num_nodes_previous)
            biases: Array of bias values for each node in this layer
        """
        if layer_idx == 0 or layer_idx >= len(self.layers):
            raise ValueError(f"Invalid layer index {layer_idx}")

        num_current = self.layers[layer_idx]
        num_previous = self.layers[layer_idx - 1]

        if weight_matrix.shape != (num_current, num_previous):
            raise ValueError(f"Weight matrix shape {weight_matrix.shape} doesn't match expected ({num_current}, {num_previous})")

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
        Forward propagation through the network

        Args:
            inputs: Input array of shape (batch_size, input_features)

        Returns:
            Output predictions
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        batch_size = inputs.shape[0]

        # First layer is input
        self.layer_outputs[0] = inputs
        self.layer_z_values[0] = inputs  # Input layer has no z values

        # Propagate through each layer
        for layer_idx in range(1, len(self.layers)):
            num_nodes = self.layers[layer_idx]
            prev_output = self.layer_outputs[layer_idx - 1]

            # Initialize output and z values for this layer
            layer_output = np.zeros((batch_size, num_nodes))
            layer_z = np.zeros((batch_size, num_nodes))

            # Calculate output for each node in current layer
            for node_idx in range(num_nodes):
                if layer_idx < len(self.connections) and node_idx < len(self.connections[layer_idx]):
                    # Get connections and weights for this node
                    node_connections = self.connections[layer_idx][node_idx]
                    node_weights = self.weights[layer_idx][node_idx]
                    node_bias = self.biases[layer_idx][node_idx] if self.biases[layer_idx] else 0.0

                    # Calculate weighted sum (z)
                    weighted_sum = node_bias
                    for conn_idx, weight in zip(node_connections, node_weights):
                        if conn_idx < prev_output.shape[1]:
                            weighted_sum += prev_output[:, conn_idx] * weight

                    # Store z value
                    layer_z[:, node_idx] = weighted_sum

            # Apply activation function
            # For softmax, apply to entire layer at once; for others, apply per node
            if self.activations[layer_idx] == 'softmax':
                # Softmax must be applied across all nodes in the layer
                activation_fn = ActivationFunctions.get_activation('softmax')
                layer_output = activation_fn(layer_z)
            else:
                # For other activations, apply per node
                activation_fn = ActivationFunctions.get_activation(self.activations[layer_idx])
                for node_idx in range(num_nodes):
                    layer_output[:, node_idx] = activation_fn(layer_z[:, node_idx])

            self.layer_outputs[layer_idx] = layer_output
            self.layer_z_values[layer_idx] = layer_z

        # Return output of last layer
        return self.layer_outputs[-1]

    def get_classification_type(self) -> str:
        """
        Determine classification type based on network architecture

        Returns:
            'binary': Single output neuron (binary classification)
            'multi-label': Multiple outputs with sigmoid (multi-label)
            'multi-class': Multiple outputs with softmax (multi-class)
        """
        num_outputs = self.layers[-1]
        output_activation = self.activations[-1]

        if num_outputs == 1:
            return 'binary'
        elif output_activation == 'softmax':
            return 'multi-class'
        else:
            # Multiple outputs with sigmoid/linear = multi-label
            return 'multi-label'

    def get_recommended_loss(self) -> str:
        """Get recommended loss function based on classification type"""
        classification_type = self.get_classification_type()

        if classification_type == 'binary':
            return 'binary'  # Binary Cross Entropy
        elif classification_type == 'multi-class':
            return 'categorical'  # Categorical Cross Entropy
        else:  # multi-label
            return 'binary'  # Binary Cross Entropy (per output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
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

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = 'mse') -> float:
        """Calculate loss between predictions and true values"""
        loss_fn = LossFunctions.get_loss(loss_function)
        return loss_fn(y_true, y_pred)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = 'mse') -> Tuple[List, List]:
        """
        Backpropagation to calculate gradients

        Args:
            y_true: True labels
            y_pred: Predicted values
            loss_function: Loss function to use

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        batch_size = y_true.shape[0]

        # Initialize gradients storage
        weight_gradients = [[] for _ in range(len(self.layers))]
        bias_gradients = [[] for _ in range(len(self.layers))]

        # Get loss derivative
        loss_derivative_fn = LossFunctions.get_loss_derivative(loss_function)

        # Calculate output layer error (delta)
        # dL/dz = dL/da * da/dz
        output_layer_idx = len(self.layers) - 1
        activation_derivative_fn = ActivationFunctions.get_activation_derivative(
            self.activations[output_layer_idx]
        )

        # dL/da (loss derivative with respect to output)
        dL_da = loss_derivative_fn(y_true, y_pred)

        # da/dz (activation derivative)
        da_dz = activation_derivative_fn(self.layer_z_values[output_layer_idx])

        # delta = dL/dz
        delta = dL_da * da_dz

        # Store deltas for each layer (working backwards)
        deltas = [None] * len(self.layers)
        deltas[output_layer_idx] = delta

        # Backpropagate through hidden layers
        for layer_idx in range(len(self.layers) - 2, 0, -1):
            # Get activation derivative for this layer
            activation_derivative_fn = ActivationFunctions.get_activation_derivative(
                self.activations[layer_idx]
            )
            da_dz = activation_derivative_fn(self.layer_z_values[layer_idx])

            # Calculate delta for this layer
            next_layer_idx = layer_idx + 1
            num_nodes = self.layers[layer_idx]
            layer_delta = np.zeros((batch_size, num_nodes))

            # For each node in current layer
            for node_idx in range(num_nodes):
                # Sum the weighted deltas from next layer
                weighted_delta_sum = 0

                # Check each node in next layer that connects to this node
                if next_layer_idx < len(self.connections):
                    for next_node_idx in range(len(self.connections[next_layer_idx])):
                        node_connections = self.connections[next_layer_idx][next_node_idx]
                        node_weights = self.weights[next_layer_idx][next_node_idx]

                        # If this node is connected to the next node
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
                if layer_idx < len(self.connections) and node_idx < len(self.connections[layer_idx]):
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

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100,
              learning_rate: float = 0.01,
              optimizer: str = 'gd',
              loss_function: str = 'mse',
              batch_size: int = None,
              verbose: bool = True) -> Dict:
        """
        Train the neural network

        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            optimizer: Optimizer to use ('gd' or 'sgd')
            loss_function: Loss function to use
            batch_size: Batch size for SGD (None = full batch)
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        # Initialize optimizer
        if optimizer.lower() == 'gd':
            opt = GradientDescent(learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate)
        else:
            opt = GradientDescent(learning_rate)

        # If batch_size not specified, use full batch
        if batch_size is None:
            batch_size = X.shape[0]

        # Reset training history
        self.training_history = {'loss': [], 'epoch': []}

        # Training loop
        for epoch in range(epochs):
            # Shuffle data for SGD
            if optimizer.lower() == 'sgd':
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
                batch_loss = self.calculate_loss(y_batch, y_pred, loss_function)
                epoch_loss += batch_loss
                num_batches += 1

                # Backward pass
                weight_grads, bias_grads = self.backward(y_batch, y_pred, loss_function)

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

    def get_architecture_summary(self) -> str:
        """Get a summary of the network architecture"""
        summary = "Neural Network Architecture:\n"
        summary += "=" * 50 + "\n"

        for i, (num_nodes, activation) in enumerate(zip(self.layers, self.activations)):
            layer_type = "Input" if i == 0 else "Hidden" if i < len(self.layers) - 1 else "Output"
            summary += f"Layer {i} ({layer_type}): {num_nodes} nodes, Activation: {activation}\n"

            if i > 0 and i < len(self.connections):
                summary += f"  Connections from Layer {i-1}:\n"
                for node_idx in range(min(len(self.connections[i]), num_nodes)):
                    conns = self.connections[i][node_idx]
                    weights = self.weights[i][node_idx]
                    bias = self.biases[i][node_idx] if i < len(self.biases) else 0
                    summary += f"    Node {node_idx}: connections={conns}, weights={weights}, bias={bias:.3f}\n"

        summary += "=" * 50
        return summary
