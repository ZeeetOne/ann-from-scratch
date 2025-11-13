"""
Network Routes

API endpoints for network building and management.

Author: ANN from Scratch Team
"""

from flask import Blueprint, request, jsonify, current_app
from ...services import NetworkService
from ...utils.validators import RequestValidator
from ...utils.data_processor import DataProcessor

network_bp = Blueprint('network', __name__)


@network_bp.route('/build_network', methods=['POST'])
def build_network():
    """
    Build neural network from configuration

    Request JSON:
        {
            "layers": [
                {"num_nodes": 3, "activation": "linear"},
                {"num_nodes": 4, "activation": "sigmoid"},
                {"num_nodes": 2, "activation": "softmax"}
            ],
            "connections": [
                {
                    "layer_idx": 1,
                    "connections": [[0,1,2], [0,1,2], [0,1,2], [0,1,2]],
                    "weights": [[...], [...], [...], [...]],
                    "biases": [0.1, -0.2, 0.3, 0.0]
                },
                ...
            ]
        }

    Returns:
        JSON response with network summary and info
    """
    try:
        data = request.json

        print("=" * 60)
        print("BACKEND: Received build_network request")
        print("=" * 60)
        print("Layers:", data.get('layers'))
        print("\nConnections data:")
        for conn_layer in data.get('connections', []):
            layer_idx = conn_layer.get('layer_idx')
            print(f"\n  Layer {layer_idx}:")
            print(f"    Connections: {conn_layer.get('connections')}")
            print(f"    Weights: {conn_layer.get('weights')}")
            print(f"    Biases: {conn_layer.get('biases')}")
        print("=" * 60)

        # Validate request
        RequestValidator.validate_build_network_request(data)

        # Build network
        network, summary = NetworkService.build_network(data)

        # Store network in app context
        current_app.current_network = network

        # Get network info
        network_info = NetworkService.get_network_info(network)
        connections_data = NetworkService.get_connections_data(network)

        return jsonify(DataProcessor.format_response(
            success=True,
            data={
                'message': 'Network built successfully',
                'summary': summary,
                'classification_type': network_info['classification_type'],
                'recommended_loss': network_info['recommended_loss'],
                'network_info': network_info,
                'connections': connections_data
            }
        ))

    except Exception as e:
        return jsonify(DataProcessor.format_response(
            success=False,
            error=str(e)
        )), 400
