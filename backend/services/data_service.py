"""
Data Service

Handles data processing, validation, and transformation.
Provides utilities for dataset handling.

Author: ANN from Scratch Team
"""

import numpy as np
import pandas as pd
from io import StringIO
from typing import Tuple, Dict, List, Optional
from ..core import NeuralNetwork


class DataService:
    """
    Service for data processing and validation.

    Responsibilities:
    - Parse and validate datasets
    - Split features and labels
    - Process predictions
    - Generate forward pass details
    """

    @staticmethod
    def parse_dataset(
        dataset_str: str,
        num_outputs: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Parse CSV dataset string

        Args:
            dataset_str: CSV string
            num_outputs: Number of output neurons (to determine label columns)

        Returns:
            Tuple of (X, y, feature_names, target_names)

        Raises:
            ValueError: If dataset is invalid
        """
        if not dataset_str or dataset_str.strip() == '':
            raise ValueError("Dataset is empty")

        # Parse CSV
        try:
            df = pd.read_csv(StringIO(dataset_str))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {str(e)}")

        if len(df.columns) < 2:
            raise ValueError("Dataset must have at least 2 columns (features and target)")

        # Extract features and targets
        if num_outputs > 1:
            # Multi-output: last num_outputs columns are targets
            if len(df.columns) < num_outputs + 1:
                raise ValueError(
                    f"Dataset has {len(df.columns)} columns but "
                    f"network expects {num_outputs} outputs. "
                    f"Need at least {num_outputs + 1} columns."
                )

            X = df.iloc[:, :-num_outputs].values
            y = df.iloc[:, -num_outputs:].values
            feature_names = df.columns[:-num_outputs].tolist()
            target_names = df.columns[-num_outputs:].tolist()
        else:
            # Single output: last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.reshape(-1, 1)
            feature_names = df.columns[:-1].tolist()
            target_names = [df.columns[-1]]

        # Validate data
        if len(X) == 0:
            raise ValueError("Dataset has no samples")

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Dataset contains NaN values")

        return X, y, feature_names, target_names

    @staticmethod
    def process_predictions(
        network: NeuralNetwork,
        X: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Process predictions and calculate metrics

        Args:
            network: NeuralNetwork instance
            X: Input features
            y_true: True labels
            threshold: Classification threshold

        Returns:
            Dict with prediction results
        """
        # Make predictions
        y_pred_classes, y_pred_probs = network.predict(X, threshold)

        # Calculate accuracy
        num_outputs = network.layers[-1]

        if num_outputs > 1:
            accuracy = np.mean((y_pred_classes == y_true.astype(int)).astype(float))
        else:
            accuracy = np.mean(
                (y_pred_classes.flatten() == y_true.flatten().astype(int)).astype(float)
            )

        # Prepare results per sample
        results = []
        for i in range(len(X)):
            if num_outputs == 1:
                result = {
                    'index': i,
                    'features': X[i].tolist(),
                    'y_true': int(y_true[i][0]),
                    'y_pred_prob': float(y_pred_probs[i][0]),
                    'y_pred_class': 'Yes' if y_pred_classes[i][0] == 1 else 'No',
                    'correct': int(y_true[i][0]) == int(y_pred_classes[i][0])
                }
            else:
                result = {
                    'index': i,
                    'features': X[i].tolist(),
                    'y_true': y_true[i].tolist(),
                    'y_pred_prob': y_pred_probs[i].tolist(),
                    'y_pred_class': y_pred_classes[i].tolist(),
                    'correct': np.array_equal(y_true[i], y_pred_classes[i])
                }
            results.append(result)

        return {
            'results': results,
            'accuracy': float(accuracy),
            'num_samples': len(X),
            'num_outputs': num_outputs
        }

    @staticmethod
    def generate_forward_pass_details(
        network: NeuralNetwork,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """
        Generate detailed forward pass information for all samples

        Args:
            network: NeuralNetwork instance
            X: Input features
            y_true: True labels

        Returns:
            Dict with detailed layer-by-layer activations
        """
        num_outputs = network.layers[-1]
        all_samples = []

        for sample_idx in range(len(X)):
            # Run forward pass for this sample
            sample_input = X[sample_idx:sample_idx+1]
            network.forward(sample_input)

            # Collect layer outputs
            layer_outputs = []

            # Input layer
            layer_outputs.append({
                'layer_index': 0,
                'layer_type': 'input',
                'num_nodes': network.layers[0],
                'activation_function': network.activations[0].name,
                'outputs': network.layer_outputs[0].tolist()
            })

            # Hidden and output layers
            for layer_idx in range(1, len(network.layers)):
                layer_type = 'hidden' if layer_idx < len(network.layers) - 1 else 'output'

                # Get layer details
                nodes_detail = []
                for node_idx in range(network.layers[layer_idx]):
                    if (layer_idx < len(network.connections) and
                        node_idx < len(network.connections[layer_idx])):

                        connected_nodes = network.connections[layer_idx][node_idx]
                        node_weights = network.weights[layer_idx][node_idx]
                        node_bias = network.biases[layer_idx][node_idx]

                        # Calculate contributions
                        input_contributions = []
                        weighted_sum = node_bias
                        prev_output = network.layer_outputs[layer_idx - 1]

                        for conn_idx, prev_node_idx in enumerate(connected_nodes):
                            weight = node_weights[conn_idx]
                            input_value = prev_output[0][prev_node_idx]
                            contribution = input_value * weight
                            weighted_sum += contribution

                            input_contributions.append({
                                'from_node': prev_node_idx,
                                'input_value': float(input_value),
                                'weight': float(weight),
                                'contribution': float(contribution)
                            })

                        activated_value = float(network.layer_outputs[layer_idx][0][node_idx])

                        nodes_detail.append({
                            'node_index': node_idx,
                            'bias': float(node_bias),
                            'weighted_sum': float(weighted_sum),
                            'activated_value': activated_value,
                            'input_contributions': input_contributions
                        })

                layer_outputs.append({
                    'layer_index': layer_idx,
                    'layer_type': layer_type,
                    'num_nodes': network.layers[layer_idx],
                    'activation_function': network.activations[layer_idx].name,
                    'nodes': nodes_detail
                })

            # Get prediction for this sample
            prediction = [node['activated_value'] for node in layer_outputs[-1]['nodes']]

            all_samples.append({
                'sample_index': sample_idx,
                'input': X[sample_idx].tolist(),
                'target': y_true[sample_idx].tolist() if num_outputs > 1 else [float(y_true[sample_idx][0])],
                'prediction': prediction,
                'layer_outputs': layer_outputs
            })

        return {
            'num_samples': len(X),
            'samples': all_samples
        }

    @staticmethod
    def validate_input_shape(
        X: np.ndarray,
        expected_features: int
    ) -> bool:
        """
        Validate input shape matches network

        Args:
            X: Input array
            expected_features: Expected number of features

        Returns:
            True if valid

        Raises:
            ValueError: If shape is invalid
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != expected_features:
            raise ValueError(
                f"Input features ({X.shape[1]}) "
                f"don't match network input layer ({expected_features})"
            )

        return True

    @staticmethod
    def format_training_predictions(
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        y_pred_classes: np.ndarray
    ) -> List[Dict]:
        """
        Format predictions for training output

        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities
            y_pred_classes: Predicted classes

        Returns:
            List of prediction dictionaries
        """
        num_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
        predictions_list = []

        for i in range(len(y_true)):
            if num_outputs > 1:
                predictions_list.append({
                    'y_true': y_true[i].tolist(),
                    'y_pred': y_pred_probs[i].tolist(),
                    'y_pred_classes': y_pred_classes[i].tolist()
                })
            else:
                predictions_list.append({
                    'y_true': int(y_true[i][0]) if y_true.ndim > 1 else int(y_true[i]),
                    'y_pred': float(y_pred_probs[i][0]) if y_pred_probs.ndim > 1 else float(y_pred_probs[i]),
                    'y_pred_classes': int(y_pred_classes[i][0]) if y_pred_classes.ndim > 1 else int(y_pred_classes[i])
                })

        return predictions_list
