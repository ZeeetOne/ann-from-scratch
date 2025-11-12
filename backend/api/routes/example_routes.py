"""
Example Routes

API endpoints for quick start examples.

Author: ANN from Scratch Team
"""

from flask import Blueprint, jsonify, current_app, request
from ...services import NetworkService
from ...utils.data_processor import DataProcessor

example_bp = Blueprint('example', __name__)


@example_bp.route('/quick_start_multiclass', methods=['POST'])
def quick_start_multiclass():
    """
    Quick start with multi-class classification example (3-4-2 with softmax)

    Returns:
        JSON response with example network and dataset
    """
    try:
        # Create example network
        network, dataset, description = NetworkService.create_example_multiclass_network()

        # Store network in app context
        current_app.current_network = network

        # Get network info
        network_info = NetworkService.get_network_info(network)
        connections_data = NetworkService.get_connections_data(network)
        summary = network.get_architecture_summary()

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'message': description,
                'summary': summary,
                'classification_type': network_info['classification_type'],
                'recommended_loss': network_info['recommended_loss'],
                'example_dataset': dataset,
                'layers': [
                    {'num_nodes': 3, 'activation': 'linear'},
                    {'num_nodes': 4, 'activation': 'sigmoid'},
                    {'num_nodes': 2, 'activation': 'softmax'}
                ],
                'connections': connections_data,
                'network_info': network_info
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400


@example_bp.route('/quick_start_binary', methods=['POST'])
def quick_start_binary():
    """
    Quick start with binary classification example (3-4-1 with sigmoid)

    Returns:
        JSON response with example network and dataset
    """
    try:
        # Create example network
        network, dataset, description = NetworkService.create_example_binary_network()

        # Store network in app context
        current_app.current_network = network

        # Get network info
        network_info = NetworkService.get_network_info(network)
        connections_data = NetworkService.get_connections_data(network)
        summary = network.get_architecture_summary()

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'message': description,
                'summary': summary,
                'classification_type': network_info['classification_type'],
                'recommended_loss': network_info['recommended_loss'],
                'example_dataset': dataset,
                'layers': [
                    {'num_nodes': 3, 'activation': 'linear'},
                    {'num_nodes': 4, 'activation': 'sigmoid'},
                    {'num_nodes': 1, 'activation': 'sigmoid'}
                ],
                'connections': connections_data,
                'network_info': network_info
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400


@example_bp.route('/generate_random_dataset', methods=['POST'])
def generate_random_dataset():
    """
    Generate random dataset matching current network architecture

    Request JSON:
        {
            "num_samples": 10  // optional, default 10
        }

    Returns:
        JSON response with random dataset
    """
    try:
        # Get current network
        if not hasattr(current_app, 'current_network') or current_app.current_network is None:
            raise ValueError('No network has been built. Please build a network first.')

        network = current_app.current_network
        data = request.json or {}

        # Get parameters
        num_samples = int(data.get('num_samples', 10))

        if num_samples < 1 or num_samples > 1000:
            raise ValueError('num_samples must be between 1 and 1000')

        # Generate random dataset (always truly random, no seed)
        dataset = NetworkService.generate_random_dataset(network, num_samples, seed=None)

        # Get network info
        network_info = NetworkService.get_network_info(network)

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'message': f'Generated {num_samples} random samples',
                'dataset': dataset,
                'num_samples': num_samples,
                'num_inputs': network.layers[0],
                'num_outputs': network.layers[-1],
                'classification_type': network_info['classification_type']
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400
