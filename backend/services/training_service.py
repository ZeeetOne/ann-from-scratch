"""
Training Service

Handles neural network training, evaluation, and metrics calculation.
Provides facade for complex training operations.

Author: ANN from Scratch Team
"""

import numpy as np
from typing import Dict, Tuple, Optional
from ..core import NeuralNetwork, LossFunctionFactory


class TrainingService:
    """
    Service for training and evaluating neural networks.

    Responsibilities:
    - Train networks with various configurations
    - Calculate evaluation metrics
    - Compute gradients
    - Track training history
    """

    @staticmethod
    def train_network(
        network: NeuralNetwork,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict
    ) -> Dict:
        """
        Train neural network

        Args:
            network: NeuralNetwork instance
            X: Training features
            y: Training labels
            config: Training configuration dict with keys:
                - epochs: Number of epochs
                - learning_rate: Learning rate
                - optimizer: Optimizer name ('gd' or 'sgd')
                - loss_function: Loss function name
                - batch_size: Batch size (optional)

        Returns:
            Dict with training results
        """
        epochs = config.get('epochs', 100)
        learning_rate = config.get('learning_rate', 0.01)
        optimizer = config.get('optimizer', 'gd')
        loss_function = config.get('loss_function', 'mse')
        batch_size = config.get('batch_size', None)

        # Train network
        history = network.train(
            X, y,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_function=loss_function,
            batch_size=batch_size,
            verbose=False
        )

        # Make predictions after training
        y_pred_classes, y_pred_probs = network.predict(X)

        # Calculate final metrics
        loss_fn = LossFunctionFactory.create(loss_function)
        final_loss = loss_fn.calculate(y, y_pred_probs)

        # Calculate accuracy
        accuracy = TrainingService._calculate_accuracy(y, y_pred_classes)

        # Calculate detailed metrics
        metrics = TrainingService.calculate_evaluation_metrics(y, y_pred_classes, y_pred_probs)

        return {
            'history': {
                'epochs': history['epoch'],
                'loss': history['loss']
            },
            'final_loss': float(final_loss),
            'accuracy': float(accuracy),
            'metrics': metrics,
            'predictions': {
                'classes': y_pred_classes.tolist(),
                'probabilities': y_pred_probs.tolist()
            }
        }

    @staticmethod
    def _calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean((y_pred == y_true.astype(int)).astype(float))

    @staticmethod
    def calculate_evaluation_metrics(
        y_true: np.ndarray,
        y_pred_classes: np.ndarray,
        y_pred_probs: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics

        Args:
            y_true: True labels
            y_pred_classes: Predicted classes
            y_pred_probs: Predicted probabilities

        Returns:
            Dict with metrics (confusion matrix, precision, recall, F1)
        """
        y_true_flat = y_true.flatten().astype(int)
        y_pred_flat = y_pred_classes.flatten().astype(int)

        # Get unique classes
        unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))

        # Calculate confusion matrix
        confusion_matrix = {}
        for true_class in unique_classes:
            confusion_matrix[int(true_class)] = {}
            for pred_class in unique_classes:
                count = np.sum((y_true_flat == true_class) & (y_pred_flat == pred_class))
                confusion_matrix[int(true_class)][int(pred_class)] = int(count)

        # Calculate per-class metrics
        metrics_per_class = {}
        for cls in unique_classes:
            tp = np.sum((y_true_flat == cls) & (y_pred_flat == cls))
            fp = np.sum((y_true_flat != cls) & (y_pred_flat == cls))
            fn = np.sum((y_true_flat == cls) & (y_pred_flat != cls))
            tn = np.sum((y_true_flat != cls) & (y_pred_flat != cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics_per_class[int(cls)] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(np.sum(y_true_flat == cls)),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }

        # Calculate macro averages
        avg_precision = np.mean([m['precision'] for m in metrics_per_class.values()])
        avg_recall = np.mean([m['recall'] for m in metrics_per_class.values()])
        avg_f1 = np.mean([m['f1_score'] for m in metrics_per_class.values()])

        return {
            'confusion_matrix': confusion_matrix,
            'metrics_per_class': metrics_per_class,
            'macro_avg': {
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1_score': float(avg_f1)
            }
        }

    @staticmethod
    def calculate_loss(
        network: NeuralNetwork,
        X: np.ndarray,
        y: np.ndarray,
        loss_function: str = 'mse'
    ) -> Dict:
        """
        Calculate loss for given data

        Args:
            network: NeuralNetwork instance
            X: Input features
            y: True labels
            loss_function: Loss function name

        Returns:
            Dict with loss information
        """
        # Forward pass
        y_pred = network.forward(X)

        # Calculate loss
        loss_fn = LossFunctionFactory.create(loss_function)
        total_loss = loss_fn.calculate(y, y_pred)

        # Calculate per-sample losses
        sample_losses = []
        for i in range(len(X)):
            sample_loss = loss_fn.calculate(
                y[i:i+1],
                y_pred[i:i+1]
            )
            sample_losses.append({
                'sample_index': i,
                'loss': float(sample_loss),
                'y_true': y[i].tolist(),
                'y_pred': y_pred[i].tolist()
            })

        return {
            'loss_function': loss_function,
            'total_loss': float(total_loss),
            'sample_losses': sample_losses,
            'num_samples': len(X)
        }

    @staticmethod
    def calculate_gradients(
        network: NeuralNetwork,
        X: np.ndarray,
        y: np.ndarray,
        loss_function: str = 'mse'
    ) -> Dict:
        """
        Calculate gradients using backpropagation (for demonstration)

        Args:
            network: NeuralNetwork instance
            X: Input features (single sample for demo)
            y: True label
            loss_function: Loss function name

        Returns:
            Dict with gradient information
        """
        # Use first sample for demonstration
        X_sample = X[0:1]
        y_sample = y[0:1]

        # Forward pass
        y_pred = network.forward(X_sample)

        # Backward pass
        loss_fn = LossFunctionFactory.create(loss_function)
        weight_grads, bias_grads = network.backward(y_sample, y_pred, loss_fn)

        # Format gradients by layer
        layers_data = []
        for layer_idx in range(1, len(network.layers)):
            if layer_idx < len(weight_grads) and weight_grads[layer_idx]:
                node_gradients = []

                for node_idx in range(len(weight_grads[layer_idx])):
                    node_gradients.append({
                        'weight_gradients': weight_grads[layer_idx][node_idx],
                        'bias_gradient': float(bias_grads[layer_idx][node_idx])
                    })

                layer_type = 'output' if layer_idx == len(network.layers) - 1 else 'hidden'

                layers_data.append({
                    'layer_index': layer_idx,
                    'layer_type': layer_type,
                    'gradients': node_gradients
                })

        return {
            'loss_function': loss_function,
            'sample_index': 0,
            'layers': layers_data
        }

    @staticmethod
    def perform_single_update(
        network: NeuralNetwork,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        optimizer: str,
        loss_function: str
    ) -> Dict:
        """
        Perform single weight update (1 epoch) for demonstration

        Args:
            network: NeuralNetwork instance
            X: Training features
            y: Training labels
            learning_rate: Learning rate
            optimizer: Optimizer name
            loss_function: Loss function name

        Returns:
            Dict with update information (old/new weights, loss change)
        """
        # Store old weights
        old_weights = {}
        old_biases = {}
        for layer_idx in range(1, len(network.layers)):
            old_weights[f'layer_{layer_idx}'] = [
                w[:] for w in network.weights[layer_idx]
            ]
            old_biases[f'layer_{layer_idx}'] = network.biases[layer_idx][:]

        # Calculate loss before
        loss_fn = LossFunctionFactory.create(loss_function)
        y_pred_before = network.forward(X)
        loss_before = loss_fn.calculate(y, y_pred_before)

        # Perform 1 epoch training
        network.train(
            X, y,
            epochs=1,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_function=loss_function,
            verbose=False
        )

        # Calculate loss after
        y_pred_after = network.forward(X)
        loss_after = loss_fn.calculate(y, y_pred_after)

        # Get new weights
        new_weights = {}
        new_biases = {}
        weight_changes = {}
        bias_changes = {}

        for layer_idx in range(1, len(network.layers)):
            layer_key = f'layer_{layer_idx}'
            new_weights[layer_key] = network.weights[layer_idx]
            new_biases[layer_key] = network.biases[layer_idx]

            # Calculate changes
            weight_changes[layer_key] = []
            bias_changes[layer_key] = []

            for node_idx in range(len(network.weights[layer_idx])):
                node_weight_changes = []
                for weight_idx in range(len(network.weights[layer_idx][node_idx])):
                    old_w = old_weights[layer_key][node_idx][weight_idx]
                    new_w = new_weights[layer_key][node_idx][weight_idx]
                    node_weight_changes.append(new_w - old_w)
                weight_changes[layer_key].append(node_weight_changes)

                old_b = old_biases[layer_key][node_idx]
                new_b = new_biases[layer_key][node_idx]
                bias_changes[layer_key].append(new_b - old_b)

        return {
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'loss_before': float(loss_before),
            'loss_after': float(loss_after),
            'loss_reduction': float(loss_before - loss_after),
            'old_weights': old_weights,
            'old_biases': old_biases,
            'new_weights': new_weights,
            'new_biases': new_biases,
            'weight_changes': weight_changes,
            'bias_changes': bias_changes
        }
