"""
Example Routes

API endpoints for quick start examples.

Author: ANN from Scratch Team
"""

from flask import Blueprint, jsonify, current_app
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
