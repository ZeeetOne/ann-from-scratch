"""
Network Service

Handles neural network building, configuration, and management.
Provides facade for complex network operations.

Author: ANN from Scratch Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..core import NeuralNetwork


class NetworkService:
    """
    Service for managing neural network lifecycle.

    Responsibilities:
    - Build networks from configuration
    - Manage network state
    - Provide network information
    - Create example networks
    """

    @staticmethod
    def build_network(config: Dict) -> Tuple[NeuralNetwork, str]:
        """
        Build neural network from configuration

        Args:
            config: Network configuration dict with keys:
                - layers: List of layer configs (num_nodes, activation)
                - connections: List of connection configs per layer

        Returns:
            Tuple of (network, summary_message)

        Raises:
            ValueError: If configuration is invalid
        """
        layers_config = config.get('layers', [])
        connections_config = config.get('connections', [])

        if not layers_config:
            raise ValueError("No layers specified in configuration")

        # Create network
        network = NeuralNetwork()

        # Add layers
        for layer_info in layers_config:
            num_nodes = layer_info.get('num_nodes')
            activation = layer_info.get('activation', 'sigmoid')

            if not num_nodes or num_nodes <= 0:
                raise ValueError(f"Invalid num_nodes: {num_nodes}")

            network.add_layer(num_nodes, activation)

        # Set connections
        for conn_info in connections_config:
            layer_idx = conn_info.get('layer_idx')
            layer_connections = conn_info.get('connections')
            layer_weights = conn_info.get('weights')
            layer_biases = conn_info.get('biases', None)

            if layer_idx is None:
                raise ValueError("layer_idx is required in connections")

            network.set_connections(
                layer_idx,
                layer_connections,
                layer_weights,
                layer_biases
            )

        summary = network.get_architecture_summary()
        return network, summary

    @staticmethod
    def get_network_info(network: NeuralNetwork) -> Dict:
        """
        Get information about network

        Args:
            network: NeuralNetwork instance

        Returns:
            Dict with network information
        """
        return {
            'num_layers': len(network.layers),
            'layer_sizes': network.layers,
            'activations': [act.name for act in network.activations],
            'classification_type': network.get_classification_type(),
            'recommended_loss': network.get_recommended_loss(),
            'total_parameters': NetworkService._count_parameters(network)
        }

    @staticmethod
    def _count_parameters(network: NeuralNetwork) -> int:
        """Count total number of trainable parameters"""
        total = 0

        for layer_idx in range(1, len(network.layers)):
            if layer_idx < len(network.weights):
                # Count weights
                for node_weights in network.weights[layer_idx]:
                    total += len(node_weights)

                # Count biases
                total += len(network.biases[layer_idx])

        return total

    @staticmethod
    def create_example_multiclass_network() -> Tuple[NeuralNetwork, str, str]:
        """
        Create example multi-class classification network (3-4-2 with softmax)

        Returns:
            Tuple of (network, dataset, description)
        """
        network = NeuralNetwork()

        # Layer 0: Input (3 nodes)
        network.add_layer(3, 'linear')

        # Layer 1: Hidden (4 nodes)
        network.add_layer(4, 'sigmoid')

        # Layer 2: Output (2 nodes with softmax)
        network.add_layer(2, 'softmax')

        # Connections Layer 1 (input -> hidden)
        connections_layer1 = [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ]
        weights_layer1 = [
            [0.5, 0.3, -0.2],
            [-0.4, 0.6, 0.1],
            [0.2, -0.5, 0.4],
            [0.7, 0.2, -0.3]
        ]
        biases_layer1 = [0.1, -0.2, 0.3, 0.0]

        network.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

        # Connections Layer 2 (hidden -> output)
        connections_layer2 = [
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]
        weights_layer2 = [
            [0.8, -0.3, 0.6, 0.4],
            [-0.5, 0.7, -0.2, 0.3]
        ]
        biases_layer2 = [0.1, -0.1]

        network.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

        # Example dataset (normalized weather data)
        dataset = "x1,x2,x3,y1,y2\n"
        dataset += "0.417,0.556,0.818,1,0\n"
        dataset += "0.833,0.833,0.091,0,1\n"
        dataset += "0.167,0.278,0.909,1,0\n"
        dataset += "0.667,0.667,0.182,0,1\n"
        dataset += "0.0,0.0,1.0,1,0\n"
        dataset += "1.0,1.0,0.0,0,1\n"
        dataset += "0.333,0.444,0.873,1,0\n"
        dataset += "0.75,0.778,0.145,0,1\n"
        dataset += "0.083,0.167,0.945,1,0\n"
        dataset += "0.917,0.889,0.036,0,1"

        description = "Multi-Class Example: 3-4-2 network with Softmax (Weather Prediction)"

        return network, dataset, description

    @staticmethod
    def create_example_binary_network() -> Tuple[NeuralNetwork, str, str]:
        """
        Create example binary classification network (3-4-1 with sigmoid)

        Returns:
            Tuple of (network, dataset, description)
        """
        network = NeuralNetwork()

        # Layer 0: Input (3 nodes)
        network.add_layer(3, 'linear')

        # Layer 1: Hidden (4 nodes)
        network.add_layer(4, 'sigmoid')

        # Layer 2: Output (1 node with sigmoid)
        network.add_layer(1, 'sigmoid')

        # Connections Layer 1
        connections_layer1 = [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ]
        weights_layer1 = [
            [0.6, 0.4, 0.3],
            [-0.3, 0.5, 0.2],
            [0.4, -0.2, 0.6],
            [0.3, 0.3, -0.4]
        ]
        biases_layer1 = [0.2, -0.1, 0.15, 0.0]

        network.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

        # Connections Layer 2
        connections_layer2 = [[0, 1, 2, 3]]
        weights_layer2 = [[0.7, -0.4, 0.5, 0.6]]
        biases_layer2 = [0.2]

        network.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

        # Example dataset (normalized student data)
        dataset = "x1,x2,x3,y1\n"
        dataset += "0.875,0.8,0.9,1\n"
        dataset += "0.125,0.2,0.1,0\n"
        dataset += "0.75,0.6,0.8,1\n"
        dataset += "0.25,0.4,0.2,0\n"
        dataset += "1.0,1.0,1.0,1\n"
        dataset += "0.0,0.0,0.0,0\n"
        dataset += "0.625,0.8,0.7,1\n"
        dataset += "0.375,0.4,0.3,0\n"
        dataset += "0.875,0.6,0.84,1\n"
        dataset += "0.125,0.2,0.16,0"

        description = "Binary Example: 3-4-1 network with Sigmoid (Student Pass/Fail)"

        return network, dataset, description

    @staticmethod
    def get_connections_data(network: NeuralNetwork) -> List[Dict]:
        """
        Extract connection data from network for serialization

        Args:
            network: NeuralNetwork instance

        Returns:
            List of connection dictionaries
        """
        connection_data = []

        for layer_idx in range(1, len(network.layers)):
            if layer_idx < len(network.connections):
                connection_data.append({
                    'layer_idx': layer_idx,
                    'connections': network.connections[layer_idx],
                    'weights': network.weights[layer_idx],
                    'biases': network.biases[layer_idx]
                })

        return connection_data
