"""
Flask Web Application for ANN from Scratch
Interactive interface for building and testing neural networks
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
from io import StringIO
from ann_core import NeuralNetwork, ActivationFunctions, LossFunctions

app = Flask(__name__)

# Global variable to store the current network
current_network = None


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/build_network', methods=['POST'])
def build_network():
    """Build neural network from user configuration"""
    global current_network

    try:
        data = request.json
        layers = data.get('layers', [])
        connections = data.get('connections', [])

        if not layers:
            return jsonify({'success': False, 'error': 'No layers specified'})

        # Create new network
        current_network = NeuralNetwork()

        # Add layers
        for layer_info in layers:
            num_nodes = layer_info.get('num_nodes')
            activation = layer_info.get('activation', 'sigmoid')
            current_network.add_layer(num_nodes, activation)

        # Set connections for each layer (starting from layer 1, as layer 0 is input)
        for conn_info in connections:
            layer_idx = conn_info.get('layer_idx')
            layer_connections = conn_info.get('connections')
            layer_weights = conn_info.get('weights')
            layer_biases = conn_info.get('biases', None)

            current_network.set_connections(layer_idx, layer_connections, layer_weights, layer_biases)

        # Get architecture summary
        summary = current_network.get_architecture_summary()

        # Get classification type and recommended loss
        classification_type = current_network.get_classification_type()
        recommended_loss = current_network.get_recommended_loss()

        return jsonify({
            'success': True,
            'message': 'Network built successfully',
            'summary': summary,
            'classification_type': classification_type,
            'recommended_loss': recommended_loss
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/forward_pass', methods=['POST'])
def forward_pass():
    """Run forward pass and show layer-by-layer activations"""
    global current_network

    try:
        if current_network is None:
            return jsonify({'success': False, 'error': 'No network has been built. Please build a network first.'})

        data = request.json
        dataset = data.get('dataset')

        if not dataset:
            return jsonify({'success': False, 'error': 'No dataset provided'})

        # Parse dataset
        df = pd.read_csv(StringIO(dataset))

        # Determine number of output neurons from network
        num_outputs = current_network.layers[-1]

        # Extract features (X) and target (y)
        if num_outputs > 1:
            X = df.iloc[:, :-num_outputs].values
            y_true = df.iloc[:, -num_outputs:].values
            feature_names = df.columns[:-num_outputs].tolist()
            target_names = df.columns[-num_outputs:].tolist()
        else:
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values.reshape(-1, 1)
            feature_names = df.columns[:-1].tolist()
            target_names = [df.columns[-1]]

        # Store all samples for selection
        all_samples = []

        for sample_idx in range(len(X)):
            # Run forward pass with detailed layer outputs for this sample
            layer_outputs = []
            activations = X[sample_idx:sample_idx+1].copy()

            # Store input layer
            layer_outputs.append({
                'layer_index': 0,
                'layer_type': 'input',
                'num_nodes': current_network.layers[0],
                'activation_function': current_network.activations[0],
                'outputs': activations.tolist()
            })

            # Forward pass through each layer
            for layer_idx in range(1, len(current_network.layers)):
                # Get connections, weights, and biases for this layer
                connections = current_network.connections[layer_idx]
                weights = current_network.weights[layer_idx]
                biases = current_network.biases[layer_idx]
                activation_fn = current_network.activations[layer_idx]

                # Calculate weighted sum for each node in this layer
                layer_outputs_detail = []
                weighted_sums = []

                for node_idx in range(len(connections)):
                    connected_nodes = connections[node_idx]
                    node_weights = weights[node_idx]
                    node_bias = biases[node_idx]

                    # Calculate weighted sum
                    weighted_sum = node_bias
                    input_contributions = []

                    for conn_idx, prev_node_idx in enumerate(connected_nodes):
                        weight = node_weights[conn_idx]
                        input_value = activations[0][prev_node_idx]
                        contribution = input_value * weight
                        weighted_sum += contribution

                        input_contributions.append({
                            'from_node': prev_node_idx,
                            'input_value': float(input_value),
                            'weight': float(weight),
                            'contribution': float(contribution)
                        })

                    weighted_sums.append(weighted_sum)

                    layer_outputs_detail.append({
                        'node_index': node_idx,
                        'bias': float(node_bias),
                        'weighted_sum': float(weighted_sum),
                        'input_contributions': input_contributions
                    })

                # Apply activation function to all weighted sums
                weighted_sums_array = np.array([weighted_sums])
                if activation_fn == 'sigmoid':
                    activated_values = 1 / (1 + np.exp(-weighted_sums_array))
                elif activation_fn == 'relu':
                    activated_values = np.maximum(0, weighted_sums_array)
                elif activation_fn == 'threshold':
                    activated_values = (weighted_sums_array >= 0).astype(float)
                elif activation_fn == 'softmax':
                    # Softmax activation
                    exp_values = np.exp(weighted_sums_array - np.max(weighted_sums_array))
                    activated_values = exp_values / np.sum(exp_values)
                else:  # linear
                    activated_values = weighted_sums_array

                # Add activated values to layer outputs
                for node_idx, detail in enumerate(layer_outputs_detail):
                    detail['activated_value'] = float(activated_values[0][node_idx])

                # Prepare activations for next layer
                activations = activated_values

                layer_type = 'hidden' if layer_idx < len(current_network.layers) - 1 else 'output'
                layer_outputs.append({
                    'layer_index': layer_idx,
                    'layer_type': layer_type,
                    'num_nodes': current_network.layers[layer_idx],
                    'activation_function': activation_fn,
                    'nodes': layer_outputs_detail
                })

            all_samples.append({
                'sample_index': sample_idx,
                'input': X[sample_idx].tolist(),
                'target': y_true[sample_idx].tolist() if num_outputs > 1 else [float(y_true[sample_idx][0])],
                'prediction': [node['activated_value'] for node in layer_outputs[-1]['nodes']],
                'layer_outputs': layer_outputs
            })

        return jsonify({
            'success': True,
            'feature_names': feature_names,
            'target_names': target_names,
            'num_samples': len(X),
            'samples': all_samples
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/calculate_loss', methods=['POST'])
def calculate_loss():
    """Calculate loss function"""
    global current_network

    try:
        if current_network is None:
            return jsonify({'success': False, 'error': 'No network has been built. Please build a network first.'})

        data = request.json
        dataset = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')

        if not dataset:
            return jsonify({'success': False, 'error': 'No dataset provided'})

        # Parse dataset
        df = pd.read_csv(StringIO(dataset))

        # Determine number of output neurons from network
        num_outputs = current_network.layers[-1]

        # Extract features (X) and target (y)
        if num_outputs > 1:
            X = df.iloc[:, :-num_outputs].values
            y_true = df.iloc[:, -num_outputs:].values
        else:
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values.reshape(-1, 1)

        # Make predictions
        y_pred_classes, y_pred = current_network.predict(X)

        # Calculate loss for each sample
        sample_losses = []
        for i in range(len(X)):
            if loss_function == 'mse':
                loss = np.mean((y_true[i] - y_pred[i]) ** 2)
            else:  # binary cross-entropy
                epsilon = 1e-15
                y_pred_clipped = np.clip(y_pred[i], epsilon, 1 - epsilon)
                loss = -np.mean(y_true[i] * np.log(y_pred_clipped) + (1 - y_true[i]) * np.log(1 - y_pred_clipped))

            sample_losses.append({
                'sample_index': i,
                'loss': float(loss),
                'y_true': y_true[i].tolist() if num_outputs > 1 else [float(y_true[i][0])],
                'y_pred': y_pred[i].tolist()
            })

        # Calculate total loss
        total_loss = current_network.calculate_loss(y_true, y_pred, loss_function)

        return jsonify({
            'success': True,
            'loss_function': loss_function,
            'total_loss': float(total_loss),
            'sample_losses': sample_losses,
            'num_samples': len(X)
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/backpropagation', methods=['POST'])
def backpropagation():
    """Calculate gradients using backpropagation"""
    global current_network

    try:
        if current_network is None:
            return jsonify({'success': False, 'error': 'No network has been built. Please build a network first.'})

        data = request.json
        dataset = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')

        if not dataset:
            return jsonify({'success': False, 'error': 'No dataset provided'})

        # Parse dataset
        df = pd.read_csv(StringIO(dataset))

        # Determine number of output neurons from network
        num_outputs = current_network.layers[-1]

        # Extract features (X) and target (y)
        if num_outputs > 1:
            X = df.iloc[:, :-num_outputs].values
            y_true = df.iloc[:, -num_outputs:].values
        else:
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values.reshape(-1, 1)

        # Use first sample for demonstration
        X_sample = X[0:1]
        y_sample = y_true[0:1]

        # Forward pass to get activations
        current_network.forward(X_sample)

        # Calculate gradients manually for demonstration
        gradients = []

        # Get output from network
        y_pred = current_network.layer_outputs[-1]

        # Calculate output layer error (∂L/∂a)
        if loss_function == 'mse':
            output_error = 2 * (y_pred - y_sample) / y_sample.size
        else:  # binary cross-entropy
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            output_error = -(y_sample / y_pred_clipped - (1 - y_sample) / (1 - y_pred_clipped))

        # Backpropagate through layers
        layer_errors = [output_error]
        layers_data = []

        for layer_idx in range(len(current_network.layers) - 1, 0, -1):
            activation_fn = current_network.activations[layer_idx]
            z_values = current_network.layer_z_values[layer_idx]

            # Calculate activation derivative
            if activation_fn == 'sigmoid':
                activation_derivative = current_network.layer_outputs[layer_idx] * (1 - current_network.layer_outputs[layer_idx])
            elif activation_fn == 'relu':
                activation_derivative = (z_values > 0).astype(float)
            else:
                activation_derivative = np.ones_like(z_values)

            # Calculate delta (∂L/∂z)
            delta = layer_errors[-1] * activation_derivative

            # Calculate weight gradients (∂L/∂W)
            prev_activation = current_network.layer_outputs[layer_idx - 1]
            weight_gradients = np.dot(delta.T, prev_activation)
            bias_gradients = np.sum(delta, axis=0, keepdims=True)

            # Prepare gradients per node for this layer
            node_gradients = []
            for node_idx in range(current_network.layers[layer_idx]):
                node_gradients.append({
                    'weight_gradients': weight_gradients[node_idx].tolist(),
                    'bias_gradient': float(bias_gradients[0][node_idx])
                })

            # Determine layer type
            layer_type = 'output' if layer_idx == len(current_network.layers) - 1 else 'hidden'

            layers_data.insert(0, {
                'layer_index': layer_idx,
                'layer_type': layer_type,
                'gradients': node_gradients
            })

            # Propagate error to previous layer
            if layer_idx > 1:
                weights = np.array(current_network.weights[layer_idx])
                error = np.dot(delta, weights)
                layer_errors.append(error)

        return jsonify({
            'success': True,
            'loss_function': loss_function,
            'sample_index': 0,  # Using first sample
            'layers': layers_data
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/update_weights', methods=['POST'])
def update_weights():
    """Update weights using gradients (single step) - demonstrating 1 epoch"""
    global current_network

    try:
        if current_network is None:
            return jsonify({'success': False, 'error': 'No network has been built. Please build a network first.'})

        data = request.json
        dataset = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')
        learning_rate = float(data.get('learning_rate', 0.01))
        optimizer = data.get('optimizer', 'gd')

        if not dataset:
            return jsonify({'success': False, 'error': 'No dataset provided'})

        # Parse dataset
        df = pd.read_csv(StringIO(dataset))

        # Determine number of output neurons from network
        num_outputs = current_network.layers[-1]

        # Extract features (X) and target (y)
        if num_outputs > 1:
            X = df.iloc[:, :-num_outputs].values
            y_true = df.iloc[:, -num_outputs:].values
        else:
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values.reshape(-1, 1)

        # Store old weights and biases for comparison
        old_weights = {}
        old_biases = {}
        for layer_idx in range(1, len(current_network.layers)):
            old_weights[f'layer_{layer_idx}'] = [w[:] for w in current_network.weights[layer_idx]]
            old_biases[f'layer_{layer_idx}'] = current_network.biases[layer_idx][:]

        # Calculate loss before update
        y_pred_before = current_network.forward(X)
        loss_before = current_network.calculate_loss(y_true, y_pred_before, loss_function)

        # Perform single weight update (1 epoch on entire dataset)
        if optimizer == 'gd':
            # Gradient Descent: full batch update
            # Forward pass
            y_pred = current_network.forward(X)

            # Calculate output error
            if loss_function == 'mse':
                output_error = 2 * (y_pred - y_true) / y_true.size
            elif loss_function == 'categorical':
                # Categorical cross-entropy derivative (for multi-class)
                output_error = (y_pred - y_true) / y_true.shape[0]
            else:  # binary cross-entropy
                epsilon = 1e-15
                y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
                output_error = -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped))

            # Backpropagate and update
            layer_errors = [output_error]

            for layer_idx in range(len(current_network.layers) - 1, 0, -1):
                activation_fn = current_network.activations[layer_idx]
                z_values = current_network.layer_z_values[layer_idx]

                # Calculate activation derivative
                if activation_fn == 'sigmoid':
                    activation_derivative = current_network.layer_outputs[layer_idx] * (1 - current_network.layer_outputs[layer_idx])
                elif activation_fn == 'relu':
                    activation_derivative = (z_values > 0).astype(float)
                else:
                    activation_derivative = np.ones_like(z_values)

                # Calculate delta
                delta = layer_errors[-1] * activation_derivative

                # Calculate gradients
                prev_activation = current_network.layer_outputs[layer_idx - 1]
                weight_gradients = np.dot(delta.T, prev_activation) / len(X)
                bias_gradients = np.sum(delta, axis=0, keepdims=True) / len(X)

                # Update weights and biases
                for node_idx in range(len(current_network.weights[layer_idx])):
                    for weight_idx in range(len(current_network.weights[layer_idx][node_idx])):
                        current_network.weights[layer_idx][node_idx][weight_idx] -= learning_rate * weight_gradients[node_idx][weight_idx]
                    current_network.biases[layer_idx][node_idx] -= learning_rate * bias_gradients[0][node_idx]

                # Propagate error to previous layer
                if layer_idx > 1:
                    weights = np.array(current_network.weights[layer_idx])
                    error = np.dot(delta, weights)
                    layer_errors.append(error)

        elif optimizer == 'sgd':
            # Stochastic Gradient Descent: update per sample
            for sample_idx in range(len(X)):
                X_sample = X[sample_idx:sample_idx+1]
                y_sample = y_true[sample_idx:sample_idx+1]

                # Forward pass
                y_pred = current_network.forward(X_sample)

                # Calculate output error
                if loss_function == 'mse':
                    output_error = 2 * (y_pred - y_sample) / y_sample.size
                elif loss_function == 'categorical':
                    # Categorical cross-entropy derivative (for multi-class)
                    output_error = (y_pred - y_sample) / y_sample.shape[0]
                else:  # binary cross-entropy
                    epsilon = 1e-15
                    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
                    output_error = -(y_sample / y_pred_clipped - (1 - y_sample) / (1 - y_pred_clipped))

                # Backpropagate and update
                layer_errors = [output_error]

                for layer_idx in range(len(current_network.layers) - 1, 0, -1):
                    activation_fn = current_network.activations[layer_idx]
                    z_values = current_network.layer_z_values[layer_idx]

                    # Calculate activation derivative
                    if activation_fn == 'sigmoid':
                        activation_derivative = current_network.layer_outputs[layer_idx] * (1 - current_network.layer_outputs[layer_idx])
                    elif activation_fn == 'relu':
                        activation_derivative = (z_values > 0).astype(float)
                    else:
                        activation_derivative = np.ones_like(z_values)

                    # Calculate delta
                    delta = layer_errors[-1] * activation_derivative

                    # Calculate gradients
                    prev_activation = current_network.layer_outputs[layer_idx - 1]
                    weight_gradients = np.dot(delta.T, prev_activation)
                    bias_gradients = np.sum(delta, axis=0, keepdims=True)

                    # Update weights and biases immediately
                    for node_idx in range(len(current_network.weights[layer_idx])):
                        for weight_idx in range(len(current_network.weights[layer_idx][node_idx])):
                            current_network.weights[layer_idx][node_idx][weight_idx] -= learning_rate * weight_gradients[node_idx][weight_idx]
                        current_network.biases[layer_idx][node_idx] -= learning_rate * bias_gradients[0][node_idx]

                    # Propagate error to previous layer
                    if layer_idx > 1:
                        weights = np.array(current_network.weights[layer_idx])
                        error = np.dot(delta, weights)
                        layer_errors.append(error)

        # Calculate loss after update
        y_pred_after = current_network.forward(X)
        loss_after = current_network.calculate_loss(y_true, y_pred_after, loss_function)

        # Get new weights and biases
        new_weights = {}
        new_biases = {}
        weight_changes = {}
        bias_changes = {}

        for layer_idx in range(1, len(current_network.layers)):
            layer_key = f'layer_{layer_idx}'
            new_weights[layer_key] = current_network.weights[layer_idx]
            new_biases[layer_key] = current_network.biases[layer_idx]

            # Calculate changes
            weight_changes[layer_key] = []
            bias_changes[layer_key] = []

            for node_idx in range(len(current_network.weights[layer_idx])):
                node_weight_changes = []
                for weight_idx in range(len(current_network.weights[layer_idx][node_idx])):
                    old_w = old_weights[layer_key][node_idx][weight_idx]
                    new_w = new_weights[layer_key][node_idx][weight_idx]
                    node_weight_changes.append(new_w - old_w)
                weight_changes[layer_key].append(node_weight_changes)

                old_b = old_biases[layer_key][node_idx]
                new_b = new_biases[layer_key][node_idx]
                bias_changes[layer_key].append(new_b - old_b)

        return jsonify({
            'success': True,
            'message': f'Weights updated using {optimizer.upper()} (1 epoch completed)',
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
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on uploaded dataset"""
    global current_network

    try:
        if current_network is None:
            return jsonify({'success': False, 'error': 'No network has been built. Please build a network first.'})

        data = request.json
        dataset = data.get('dataset')
        loss_function = data.get('loss_function', 'mse')
        threshold = float(data.get('threshold', 0.5))

        if not dataset:
            return jsonify({'success': False, 'error': 'No dataset provided'})

        # Parse dataset
        df = pd.read_csv(StringIO(dataset))

        # Assuming last column is the target (y)
        if len(df.columns) < 2:
            return jsonify({'success': False, 'error': 'Dataset must have at least 2 columns (features and target)'})

        # Determine number of output neurons from network
        num_outputs = current_network.layers[-1]

        # Extract features (X) and target (y)
        # For multi-output, the last num_outputs columns are targets
        if num_outputs > 1:
            X = df.iloc[:, :-num_outputs].values
            y_true = df.iloc[:, -num_outputs:].values
        else:
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values.reshape(-1, 1)

        # Make predictions
        y_pred_classes, y_pred_probs = current_network.predict(X, threshold=threshold)

        # Calculate loss
        loss = current_network.calculate_loss(y_true, y_pred_probs, loss_function)

        # Prepare results
        results = []
        for i in range(len(X)):
            row_data = X[i].tolist()

            if num_outputs == 1:
                # Single output
                result = {
                    'index': i,
                    'features': row_data,
                    'y_true': int(y_true[i][0]),
                    'y_pred_prob': float(y_pred_probs[i][0]),
                    'y_pred_class': 'Yes' if y_pred_classes[i][0] == 1 else 'No',
                    'correct': int(y_true[i][0]) == int(y_pred_classes[i][0])
                }
            else:
                # Multi-output
                result = {
                    'index': i,
                    'features': row_data,
                    'y_true': y_true[i].tolist(),
                    'y_pred_prob': y_pred_probs[i].tolist(),
                    'y_pred_class': y_pred_classes[i].tolist(),
                    'correct': np.array_equal(y_true[i], y_pred_classes[i])
                }
            results.append(result)

        # Calculate accuracy
        accuracy = np.mean([r['correct'] for r in results])

        # Get feature and target names
        if num_outputs > 1:
            feature_names = df.columns[:-num_outputs].tolist()
            target_names = df.columns[-num_outputs:].tolist()
        else:
            feature_names = df.columns[:-1].tolist()
            target_names = [df.columns[-1]]

        return jsonify({
            'success': True,
            'results': results,
            'loss': float(loss),
            'accuracy': float(accuracy),
            'feature_names': feature_names,
            'target_names': target_names,
            'num_outputs': num_outputs
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/train', methods=['POST'])
def train():
    """Train the neural network"""
    global current_network

    try:
        if current_network is None:
            return jsonify({'success': False, 'error': 'No network has been built. Please build a network first.'})

        data = request.json
        dataset = data.get('dataset')
        epochs = int(data.get('epochs', 100))
        learning_rate = float(data.get('learning_rate', 0.01))
        optimizer = data.get('optimizer', 'gd')
        loss_function = data.get('loss_function', 'mse')
        batch_size = data.get('batch_size', None)
        if batch_size:
            batch_size = int(batch_size)

        if not dataset:
            return jsonify({'success': False, 'error': 'No dataset provided'})

        # Parse dataset
        df = pd.read_csv(StringIO(dataset))

        # Assuming last column is the target (y)
        if len(df.columns) < 2:
            return jsonify({'success': False, 'error': 'Dataset must have at least 2 columns (features and target)'})

        # Determine number of output neurons from network
        num_outputs = current_network.layers[-1]

        # Extract features (X) and target (y) based on number of outputs
        if num_outputs > 1:
            # Multi-output: last num_outputs columns are targets
            X = df.iloc[:, :-num_outputs].values
            y_true = df.iloc[:, -num_outputs:].values
        else:
            # Single output: last column is target
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values.reshape(-1, 1)

        # Store initial weights and biases
        initial_weights = [layer_weights[:] for layer_weights in current_network.weights]
        initial_biases = [layer_biases[:] for layer_biases in current_network.biases]

        # Train the network
        history = current_network.train(
            X, y_true,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_function=loss_function,
            batch_size=batch_size,
            verbose=False
        )

        # Make predictions after training
        y_pred_classes, y_pred_probs = current_network.predict(X)

        # Calculate final loss and accuracy
        final_loss = current_network.calculate_loss(y_true, y_pred_probs, loss_function)

        # Calculate accuracy (handle both single and multi-output)
        if num_outputs > 1:
            # Multi-output: calculate accuracy across all outputs
            accuracy = np.mean((y_pred_classes == y_true.astype(int)).astype(float))
        else:
            # Single output
            accuracy = np.mean((y_pred_classes.flatten() == y_true.flatten().astype(int)).astype(float))

        # Calculate evaluation metrics
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

        # Calculate precision, recall, F1 for each class
        metrics_per_class = {}
        for cls in unique_classes:
            # True Positives, False Positives, False Negatives
            tp = np.sum((y_true_flat == cls) & (y_pred_flat == cls))
            fp = np.sum((y_true_flat != cls) & (y_pred_flat == cls))
            fn = np.sum((y_true_flat == cls) & (y_pred_flat != cls))
            tn = np.sum((y_true_flat != cls) & (y_pred_flat != cls))

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = np.sum(y_true_flat == cls)

            metrics_per_class[int(cls)] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(support),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }

        # Calculate macro averages
        avg_precision = np.mean([m['precision'] for m in metrics_per_class.values()])
        avg_recall = np.mean([m['recall'] for m in metrics_per_class.values()])
        avg_f1 = np.mean([m['f1_score'] for m in metrics_per_class.values()])

        # Prepare updated weights and biases
        updated_weights = {}
        updated_biases = {}
        for layer_idx in range(1, len(current_network.layers)):
            if layer_idx < len(current_network.weights):
                updated_weights[f'layer_{layer_idx}'] = current_network.weights[layer_idx]
                updated_biases[f'layer_{layer_idx}'] = current_network.biases[layer_idx]

        # Prepare predictions for display (per sample, not flattened)
        predictions_list = []
        for i in range(len(X)):
            if num_outputs > 1:
                predictions_list.append({
                    'y_true': y_true[i].tolist(),
                    'y_pred': y_pred_probs[i].tolist(),
                    'y_pred_classes': y_pred_classes[i].tolist()
                })
            else:
                predictions_list.append({
                    'y_true': int(y_true[i][0]),
                    'y_pred': float(y_pred_probs[i][0]),
                    'y_pred_classes': int(y_pred_classes[i][0])
                })

        return jsonify({
            'success': True,
            'message': f'Training completed: {epochs} epochs',
            'history': {
                'epochs': history['epoch'],
                'loss': history['loss']
            },
            'final_loss': float(final_loss),
            'accuracy': float(accuracy),
            'num_outputs': num_outputs,
            'evaluation': {
                'confusion_matrix': confusion_matrix,
                'metrics_per_class': metrics_per_class,
                'macro_avg': {
                    'precision': float(avg_precision),
                    'recall': float(avg_recall),
                    'f1_score': float(avg_f1)
                }
            },
            'updated_weights': updated_weights,
            'updated_biases': updated_biases,
            'predictions': predictions_list
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


@app.route('/quick_start_multiclass', methods=['POST'])
def quick_start_multiclass():
    """Quick start with multi-class classification example (3-4-2 with softmax)"""
    global current_network

    try:
        # Create a 3-4-2 network for multi-class classification
        # 3 input neurons: representing 3 features (e.g., temperature, pressure, humidity)
        # 1 hidden layer with 4 neurons
        # 2 output neurons: representing 2 mutually exclusive classes (e.g., "rain" or "sunny")
        current_network = NeuralNetwork()

        # Layer 0: Input layer (3 nodes - suhu, tekanan, kelembapan)
        current_network.add_layer(3, 'linear')

        # Layer 1: Hidden layer (4 nodes)
        current_network.add_layer(4, 'sigmoid')

        # Layer 2: Output layer (2 nodes - hujan, cerah)
        # Using softmax for multi-class classification
        current_network.add_layer(2, 'softmax')

        # Set connections for layer 1 (input to hidden)
        connections_layer1 = [
            [0, 1, 2],  # Node 0 connects to all 3 input nodes
            [0, 1, 2],  # Node 1 connects to all 3 input nodes
            [0, 1, 2],  # Node 2 connects to all 3 input nodes
            [0, 1, 2]   # Node 3 connects to all 3 input nodes
        ]
        weights_layer1 = [
            [0.5, 0.3, -0.2],   # Weights for hidden node 0
            [-0.4, 0.6, 0.1],   # Weights for hidden node 1
            [0.2, -0.5, 0.4],   # Weights for hidden node 2
            [0.7, 0.2, -0.3]    # Weights for hidden node 3
        ]
        biases_layer1 = [0.1, -0.2, 0.3, 0.0]

        current_network.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

        # Set connections for layer 2 (hidden to output)
        connections_layer2 = [
            [0, 1, 2, 3],  # Output node 0 (hujan) connects to all 4 hidden nodes
            [0, 1, 2, 3]   # Output node 1 (cerah) connects to all 4 hidden nodes
        ]
        weights_layer2 = [
            [0.8, -0.3, 0.6, 0.4],   # Weights for output node 0 (hujan)
            [-0.5, 0.7, -0.2, 0.3]   # Weights for output node 1 (cerah)
        ]
        biases_layer2 = [0.1, -0.1]

        current_network.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

        summary = current_network.get_architecture_summary()

        # Get classification type and recommended loss
        classification_type = current_network.get_classification_type()
        recommended_loss = current_network.get_recommended_loss()

        # Prepare connection data for frontend
        connection_data = []
        for layer_idx in range(1, len(current_network.layers)):
            if layer_idx < len(current_network.connections):
                connection_data.append({
                    'layer_idx': layer_idx,
                    'connections': current_network.connections[layer_idx],
                    'weights': current_network.weights[layer_idx],
                    'biases': current_network.biases[layer_idx]
                })

        # Generate example dataset for multi-class classification (one-hot encoded outputs)
        # NOTE: Values are normalized to [0, 1] range to prevent sigmoid saturation
        # Original data: temp(20-32°C), pressure(1000-1018hPa), humidity(40-95%)
        # Normalized: (value - min) / (max - min)
        example_dataset = "x1,x2,x3,y1,y2\n"
        example_dataset += "0.417,0.556,0.818,1,0\n"  # Rain  (temp=25, pres=1010, hum=85)
        example_dataset += "0.833,0.833,0.091,0,1\n"  # Sunny (temp=30, pres=1015, hum=45)
        example_dataset += "0.167,0.278,0.909,1,0\n"  # Rain  (temp=22, pres=1005, hum=90)
        example_dataset += "0.667,0.667,0.182,0,1\n"  # Sunny (temp=28, pres=1012, hum=50)
        example_dataset += "0.0,0.0,1.0,1,0\n"        # Rain  (temp=20, pres=1000, hum=95)
        example_dataset += "1.0,1.0,0.0,0,1\n"        # Sunny (temp=32, pres=1018, hum=40)
        example_dataset += "0.333,0.444,0.873,1,0\n"  # Rain  (temp=24, pres=1008, hum=88)
        example_dataset += "0.75,0.778,0.145,0,1\n"   # Sunny (temp=29, pres=1014, hum=48)
        example_dataset += "0.083,0.167,0.945,1,0\n"  # Rain  (temp=21, pres=1003, hum=92)
        example_dataset += "0.917,0.889,0.036,0,1"    # Sunny (temp=31, pres=1016, hum=42)

        return jsonify({
            'success': True,
            'message': 'Multi-Class Example: 3-4-2 network with Softmax (Weather Prediction)',
            'summary': summary,
            'classification_type': classification_type,
            'recommended_loss': recommended_loss,
            'example_dataset': example_dataset,
            'layers': [
                {'num_nodes': 3, 'activation': 'linear'},
                {'num_nodes': 4, 'activation': 'sigmoid'},
                {'num_nodes': 2, 'activation': 'softmax'}
            ],
            'connections': connection_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/quick_start_binary', methods=['POST'])
def quick_start_binary():
    """Quick start with binary classification example (3-4-1 with sigmoid)"""
    global current_network

    try:
        # Create a 3-4-1 network for binary classification
        # 3 input neurons: representing 3 features (e.g., study hours, sleep hours, previous score)
        # 1 hidden layer with 4 neurons
        # 1 output neuron: representing pass/fail prediction
        current_network = NeuralNetwork()

        # Layer 0: Input layer (3 nodes - jam_belajar, jam_tidur, nilai_sebelumnya)
        current_network.add_layer(3, 'linear')

        # Layer 1: Hidden layer (4 nodes)
        current_network.add_layer(4, 'sigmoid')

        # Layer 2: Output layer (1 node - lulus/tidak_lulus)
        # Using sigmoid for binary classification
        current_network.add_layer(1, 'sigmoid')

        # Set connections for layer 1 (input to hidden)
        connections_layer1 = [
            [0, 1, 2],  # Node 0 connects to all 3 input nodes
            [0, 1, 2],  # Node 1 connects to all 3 input nodes
            [0, 1, 2],  # Node 2 connects to all 3 input nodes
            [0, 1, 2]   # Node 3 connects to all 3 input nodes
        ]
        weights_layer1 = [
            [0.6, 0.4, 0.3],    # Weights for hidden node 0
            [-0.3, 0.5, 0.2],   # Weights for hidden node 1
            [0.4, -0.2, 0.6],   # Weights for hidden node 2
            [0.3, 0.3, -0.4]    # Weights for hidden node 3
        ]
        biases_layer1 = [0.2, -0.1, 0.15, 0.0]

        current_network.set_connections(1, connections_layer1, weights_layer1, biases_layer1)

        # Set connections for layer 2 (hidden to output)
        connections_layer2 = [
            [0, 1, 2, 3]  # Output node 0 (lulus) connects to all 4 hidden nodes
        ]
        weights_layer2 = [
            [0.7, -0.4, 0.5, 0.6]   # Weights for output node 0 (lulus)
        ]
        biases_layer2 = [0.2]

        current_network.set_connections(2, connections_layer2, weights_layer2, biases_layer2)

        summary = current_network.get_architecture_summary()

        # Get classification type and recommended loss
        classification_type = current_network.get_classification_type()
        recommended_loss = current_network.get_recommended_loss()

        # Prepare connection data for frontend
        connection_data = []
        for layer_idx in range(1, len(current_network.layers)):
            if layer_idx < len(current_network.connections):
                connection_data.append({
                    'layer_idx': layer_idx,
                    'connections': current_network.connections[layer_idx],
                    'weights': current_network.weights[layer_idx],
                    'biases': current_network.biases[layer_idx]
                })

        # Generate example dataset for binary classification (single output: 0 or 1)
        # NOTE: Values are normalized to [0, 1] range to prevent sigmoid saturation
        # Original data: study_hours(1-9h), sleep_hours(3-8h), previous_score(40-90)
        # Normalized: (value - min) / (max - min)
        example_dataset = "x1,x2,x3,y1\n"
        example_dataset += "0.875,0.8,0.9,1\n"      # Pass (study=8, sleep=7, score=85)
        example_dataset += "0.125,0.2,0.1,0\n"      # Fail (study=2, sleep=4, score=45)
        example_dataset += "0.75,0.6,0.8,1\n"       # Pass (study=7, sleep=6, score=80)
        example_dataset += "0.25,0.4,0.2,0\n"       # Fail (study=3, sleep=5, score=50)
        example_dataset += "1.0,1.0,1.0,1\n"        # Pass (study=9, sleep=8, score=90)
        example_dataset += "0.0,0.0,0.0,0\n"        # Fail (study=1, sleep=3, score=40)
        example_dataset += "0.625,0.8,0.7,1\n"      # Pass (study=6, sleep=7, score=75)
        example_dataset += "0.375,0.4,0.3,0\n"      # Fail (study=4, sleep=5, score=55)
        example_dataset += "0.875,0.6,0.84,1\n"     # Pass (study=8, sleep=6, score=82)
        example_dataset += "0.125,0.2,0.16,0"       # Fail (study=2, sleep=4, score=48)

        return jsonify({
            'success': True,
            'message': 'Binary Example: 3-4-1 network with Sigmoid (Student Pass/Fail Prediction)',
            'summary': summary,
            'classification_type': classification_type,
            'recommended_loss': recommended_loss,
            'example_dataset': example_dataset,
            'layers': [
                {'num_nodes': 3, 'activation': 'linear'},
                {'num_nodes': 4, 'activation': 'sigmoid'},
                {'num_nodes': 1, 'activation': 'sigmoid'}
            ],
            'connections': connection_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("Starting ANN from Scratch Web Application...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
