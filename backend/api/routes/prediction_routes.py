"""
Prediction Routes

API endpoints for predictions and forward pass.

Author: ANN from Scratch Team
"""

from flask import Blueprint, request, jsonify, current_app
from ...services import DataService
from ...utils.validators import RequestValidator
from ...utils.data_processor import DataProcessor

prediction_bp = Blueprint('prediction', __name__)


def get_current_network():
    """Helper to get current network from app context"""
    if not hasattr(current_app, 'current_network') or current_app.current_network is None:
        raise ValueError('No network has been built. Please build a network first.')
    return current_app.current_network


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions on uploaded dataset

    Request JSON:
        {
            "dataset": "csv_string",
            "loss_function": "mse",
            "threshold": 0.5
        }

    Returns:
        JSON response with predictions and metrics
    """
    try:
        network = get_current_network()
        data = request.json

        # Validate request
        RequestValidator.validate_prediction_request(data)

        # Parse dataset
        dataset_str = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')
        threshold = float(data.get('threshold', 0.5))
        num_outputs = network.layers[-1]

        X, y, feature_names, target_names = DataService.parse_dataset(dataset_str, num_outputs)

        # Make predictions
        prediction_results = DataService.process_predictions(network, X, y, threshold)

        # Calculate loss
        from ...core import LossFunctionFactory
        loss_fn = LossFunctionFactory.create(loss_function)
        y_pred_probs = network.forward(X)
        loss = loss_fn.calculate(y, y_pred_probs)

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'results': prediction_results['results'],
                'loss': float(loss),
                'accuracy': prediction_results['accuracy'],
                'feature_names': feature_names,
                'target_names': target_names,
                'num_outputs': num_outputs
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400


@prediction_bp.route('/forward_pass', methods=['POST'])
def forward_pass():
    """
    Run forward pass and show layer-by-layer activations

    Request JSON:
        {
            "dataset": "csv_string"
        }

    Returns:
        JSON response with detailed forward pass information
    """
    try:
        network = get_current_network()
        data = request.json

        # Validate request
        RequestValidator.validate_forward_pass_request(data)

        # Parse dataset
        dataset_str = data.get('dataset')
        num_outputs = network.layers[-1]

        X, y, feature_names, target_names = DataService.parse_dataset(dataset_str, num_outputs)

        # Generate forward pass details
        forward_details = DataService.generate_forward_pass_details(network, X, y)

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'feature_names': feature_names,
                'target_names': target_names,
                'num_samples': forward_details['num_samples'],
                'samples': forward_details['samples']
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400
