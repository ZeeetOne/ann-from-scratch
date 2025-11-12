"""
Data Processor

Utilities for data transformation and formatting.

Author: ANN from Scratch Team
"""

import numpy as np
from typing import Dict, Any, List


class DataProcessor:
    """
    Processor for data transformation and formatting.

    Provides static methods for common data processing tasks.
    """

    @staticmethod
    def format_response(
        success: bool,
        data: Dict[str, Any] = None,
        error: str = None
    ) -> Dict[str, Any]:
        """
        Format API response consistently

        Args:
            success: Whether operation was successful
            data: Response data (if successful)
            error: Error message (if failed)

        Returns:
            Formatted response dict
        """
        response = {'success': success}

        if success:
            if data:
                response.update(data)
        else:
            response['error'] = error or 'Unknown error occurred'

        return response

    @staticmethod
    def numpy_to_python(obj: Any) -> Any:
        """
        Convert numpy types to Python native types for JSON serialization

        Args:
            obj: Object to convert

        Returns:
            Python native type
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: DataProcessor.numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DataProcessor.numpy_to_python(item) for item in obj]
        else:
            return obj

    @staticmethod
    def extract_weights_and_biases(network) -> Dict[str, Any]:
        """
        Extract weights and biases from network for serialization

        Args:
            network: NeuralNetwork instance

        Returns:
            Dict with weights and biases
        """
        weights_dict = {}
        biases_dict = {}

        for layer_idx in range(1, len(network.layers)):
            if layer_idx < len(network.weights):
                weights_dict[f'layer_{layer_idx}'] = network.weights[layer_idx]
                biases_dict[f'layer_{layer_idx}'] = network.biases[layer_idx]

        return {
            'weights': weights_dict,
            'biases': biases_dict
        }

    @staticmethod
    def safe_float_conversion(value: Any, default: float = 0.0) -> float:
        """
        Safely convert value to float

        Args:
            value: Value to convert
            default: Default value if conversion fails

        Returns:
            Float value
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_int_conversion(value: Any, default: int = 0) -> int:
        """
        Safely convert value to int

        Args:
            value: Value to convert
            default: Default value if conversion fails

        Returns:
            Int value
        """
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
