"""
Request Validators

Validates API request data before processing.
Follows Fail-Fast principle for early error detection.

Author: ANN from Scratch Team
"""

from typing import Dict, Any, List


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class RequestValidator:
    """
    Validator for API request data.

    Provides static methods for validating different types of requests.
    """

    @staticmethod
    def validate_build_network_request(data: Dict[str, Any]) -> None:
        """
        Validate build network request

        Args:
            data: Request data dict

        Raises:
            ValidationError: If validation fails
        """
        if not data:
            raise ValidationError("Request data is empty")

        # Validate layers
        layers = data.get('layers', [])
        if not layers:
            raise ValidationError("No layers specified")

        if not isinstance(layers, list):
            raise ValidationError("'layers' must be a list")

        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValidationError(f"Layer {i} must be a dict")

            num_nodes = layer.get('num_nodes')
            if num_nodes is None:
                raise ValidationError(f"Layer {i}: 'num_nodes' is required")

            if not isinstance(num_nodes, int) or num_nodes <= 0:
                raise ValidationError(
                    f"Layer {i}: 'num_nodes' must be a positive integer, got {num_nodes}"
                )

            activation = layer.get('activation', 'sigmoid')
            valid_activations = ['sigmoid', 'relu', 'linear', 'softmax', 'threshold']
            if activation not in valid_activations:
                raise ValidationError(
                    f"Layer {i}: Invalid activation '{activation}'. "
                    f"Must be one of {valid_activations}"
                )

        # Validate connections
        connections = data.get('connections', [])
        if not isinstance(connections, list):
            raise ValidationError("'connections' must be a list")

        for i, conn in enumerate(connections):
            if not isinstance(conn, dict):
                raise ValidationError(f"Connection {i} must be a dict")

            layer_idx = conn.get('layer_idx')
            if layer_idx is None:
                raise ValidationError(f"Connection {i}: 'layer_idx' is required")

            if not isinstance(layer_idx, int) or layer_idx <= 0:
                raise ValidationError(
                    f"Connection {i}: 'layer_idx' must be a positive integer"
                )

            if 'connections' not in conn:
                raise ValidationError(f"Connection {i}: 'connections' is required")

            if 'weights' not in conn:
                raise ValidationError(f"Connection {i}: 'weights' is required")

    @staticmethod
    def validate_training_request(data: Dict[str, Any]) -> None:
        """
        Validate training request

        Args:
            data: Request data dict

        Raises:
            ValidationError: If validation fails
        """
        if not data:
            raise ValidationError("Request data is empty")

        # Validate dataset
        dataset = data.get('dataset')
        if not dataset or not isinstance(dataset, str) or dataset.strip() == '':
            raise ValidationError("'dataset' is required and must be a non-empty string")

        # Validate epochs
        epochs = data.get('epochs', 100)
        if not isinstance(epochs, int) or epochs <= 0 or epochs > 100000:
            raise ValidationError("'epochs' must be an integer between 1 and 100000")

        # Validate learning_rate
        learning_rate = data.get('learning_rate', 0.01)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 100:
            raise ValidationError("'learning_rate' must be a number between 0 and 100")

        # Validate optimizer
        optimizer = data.get('optimizer', 'gd')
        valid_optimizers = ['gd', 'sgd', 'momentum']
        if optimizer not in valid_optimizers:
            raise ValidationError(
                f"Invalid optimizer '{optimizer}'. Must be one of {valid_optimizers}"
            )

        # Validate loss_function
        loss_function = data.get('loss_function', 'mse')
        valid_losses = ['mse', 'binary', 'categorical']
        if loss_function not in valid_losses:
            raise ValidationError(
                f"Invalid loss_function '{loss_function}'. Must be one of {valid_losses}"
            )

        # Validate batch_size (optional)
        batch_size = data.get('batch_size')
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValidationError("'batch_size' must be a positive integer")

    @staticmethod
    def validate_prediction_request(data: Dict[str, Any]) -> None:
        """
        Validate prediction request

        Args:
            data: Request data dict

        Raises:
            ValidationError: If validation fails
        """
        if not data:
            raise ValidationError("Request data is empty")

        # Validate dataset
        dataset = data.get('dataset')
        if not dataset or not isinstance(dataset, str) or dataset.strip() == '':
            raise ValidationError("'dataset' is required and must be a non-empty string")

        # Validate loss_function (optional)
        loss_function = data.get('loss_function', 'mse')
        valid_losses = ['mse', 'binary', 'categorical']
        if loss_function not in valid_losses:
            raise ValidationError(
                f"Invalid loss_function '{loss_function}'. Must be one of {valid_losses}"
            )

        # Validate threshold (optional)
        threshold = data.get('threshold', 0.5)
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise ValidationError("'threshold' must be a number between 0 and 1")

    @staticmethod
    def validate_forward_pass_request(data: Dict[str, Any]) -> None:
        """
        Validate forward pass request

        Args:
            data: Request data dict

        Raises:
            ValidationError: If validation fails
        """
        if not data:
            raise ValidationError("Request data is empty")

        # Validate dataset
        dataset = data.get('dataset')
        if not dataset or not isinstance(dataset, str) or dataset.strip() == '':
            raise ValidationError("'dataset' is required and must be a non-empty string")

    @staticmethod
    def validate_loss_calculation_request(data: Dict[str, Any]) -> None:
        """
        Validate loss calculation request

        Args:
            data: Request data dict

        Raises:
            ValidationError: If validation fails
        """
        RequestValidator.validate_forward_pass_request(data)

        # Validate loss_function
        loss_function = data.get('loss_function', 'mse')
        valid_losses = ['mse', 'binary', 'categorical']
        if loss_function not in valid_losses:
            raise ValidationError(
                f"Invalid loss_function '{loss_function}'. Must be one of {valid_losses}"
            )
