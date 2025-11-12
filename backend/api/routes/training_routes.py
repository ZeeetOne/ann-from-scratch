"""
Training Routes

API endpoints for training operations.

Author: ANN from Scratch Team
"""

from flask import Blueprint, request, jsonify, current_app
from ...services import TrainingService, DataService
from ...utils.validators import RequestValidator
from ...utils.data_processor import DataProcessor

training_bp = Blueprint('training', __name__)


def get_current_network():
    """Helper to get current network from app context"""
    if not hasattr(current_app, 'current_network') or current_app.current_network is None:
        raise ValueError('No network has been built. Please build a network first.')
    return current_app.current_network


@training_bp.route('/train', methods=['POST'])
def train():
    """
    Train the neural network

    Request JSON:
        {
            "dataset": "csv_string",
            "epochs": 1000,
            "learning_rate": 0.5,
            "optimizer": "gd",
            "loss_function": "mse",
            "batch_size": null  // optional
        }

    Returns:
        JSON response with training history and metrics
    """
    try:
        network = get_current_network()
        data = request.json

        # Validate request
        RequestValidator.validate_training_request(data)

        # Parse dataset
        dataset_str = data.get('dataset')
        num_outputs = network.layers[-1]
        X, y, feature_names, target_names = DataService.parse_dataset(dataset_str, num_outputs)

        # Training configuration
        config = {
            'epochs': int(data.get('epochs', 100)),
            'learning_rate': float(data.get('learning_rate', 0.01)),
            'optimizer': data.get('optimizer', 'gd'),
            'loss_function': data.get('loss_function', 'mse'),
            'batch_size': data.get('batch_size', None)
        }

        if config['batch_size'] is not None:
            config['batch_size'] = int(config['batch_size'])

        # Train network
        results = TrainingService.train_network(network, X, y, config)

        # Get updated weights and biases
        updated_params = DataProcessor.extract_weights_and_biases(network)

        # Format predictions - get fresh predictions as numpy arrays
        y_pred_classes, y_pred_probs = network.predict(X)
        predictions = DataService.format_training_predictions(
            y,
            y_pred_probs,
            y_pred_classes
        )

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'message': f"Training completed: {config['epochs']} epochs",
                'history': results['history'],
                'final_loss': results['final_loss'],
                'accuracy': results['accuracy'],
                'num_outputs': num_outputs,
                'evaluation': results['metrics'],
                'updated_weights': updated_params['weights'],
                'updated_biases': updated_params['biases'],
                'predictions': predictions
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400


@training_bp.route('/calculate_loss', methods=['POST'])
def calculate_loss():
    """
    Calculate loss for current network

    Request JSON:
        {
            "dataset": "csv_string",
            "loss_function": "mse"
        }

    Returns:
        JSON response with loss information
    """
    try:
        network = get_current_network()
        data = request.json

        # Validate request
        RequestValidator.validate_loss_calculation_request(data)

        # Parse dataset
        dataset_str = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')
        num_outputs = network.layers[-1]

        X, y, _, _ = DataService.parse_dataset(dataset_str, num_outputs)

        # Calculate loss
        loss_info = TrainingService.calculate_loss(network, X, y, loss_function)

        return jsonify(DataProcessor.format_response(
            success=True,
            data=loss_info
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400


@training_bp.route('/backpropagation', methods=['POST'])
def backpropagation():
    """
    Calculate gradients using backpropagation (for demonstration)

    Request JSON:
        {
            "dataset": "csv_string",
            "loss_function": "mse"
        }

    Returns:
        JSON response with gradient information
    """
    try:
        network = get_current_network()
        data = request.json

        # Validate request
        RequestValidator.validate_loss_calculation_request(data)

        # Parse dataset
        dataset_str = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')
        num_outputs = network.layers[-1]

        X, y, _, _ = DataService.parse_dataset(dataset_str, num_outputs)

        # Calculate gradients
        gradients_info = TrainingService.calculate_gradients(network, X, y, loss_function)

        return jsonify(DataProcessor.format_response(
            success=True,
            data=gradients_info
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400


@training_bp.route('/update_weights', methods=['POST'])
def update_weights():
    """
    Update weights using gradients (single step demonstration)

    Request JSON:
        {
            "dataset": "csv_string",
            "loss_function": "mse",
            "learning_rate": 0.01,
            "optimizer": "gd"
        }

    Returns:
        JSON response with weight update information
    """
    try:
        network = get_current_network()
        data = request.json

        # Validate request
        RequestValidator.validate_training_request(data)

        # Parse dataset
        dataset_str = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')
        learning_rate = float(data.get('learning_rate', 0.01))
        optimizer = data.get('optimizer', 'gd')
        num_outputs = network.layers[-1]

        X, y, _, _ = DataService.parse_dataset(dataset_str, num_outputs)

        # Perform single update
        update_info = TrainingService.perform_single_update(
            network, X, y, learning_rate, optimizer, loss_function
        )

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'message': f'Weights updated using {optimizer.upper()} (1 epoch completed)',
                **update_info
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400
