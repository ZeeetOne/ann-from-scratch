// ANN from Scratch - Frontend JavaScript

// DOM Elements
const buildNetworkBtn = document.getElementById('buildNetworkBtn');
const networkSummary = document.getElementById('networkSummary');
const quickStartMultiClassBtn = document.getElementById('quickStartMultiClassBtn');
const quickStartBinaryBtn = document.getElementById('quickStartBinaryBtn');
const loadExampleBtn = document.getElementById('loadExampleBtn');
const datasetInput = document.getElementById('datasetInput');
const forwardPassBtn = document.getElementById('forwardPassBtn');
const forwardPassContainer = document.getElementById('forwardPassContainer');
const forwardPassSummary = document.getElementById('forwardPassSummary');
// const resultsContainer = document.getElementById('resultsContainer');  // Removed section
// const metricsContainer = document.getElementById('metricsContainer');  // Removed section
const trainBtn = document.getElementById('trainBtn');
const trainingResults = document.getElementById('trainingResults');
const trainingProgress = document.getElementById('trainingProgress');
const evaluationResults = document.getElementById('evaluationResults');
const evaluationResultsSection = document.getElementById('evaluationResultsSection');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const datasetValidation = document.getElementById('datasetValidation');
const datasetStats = document.getElementById('datasetStats');
const datasetRequirement = document.getElementById('datasetRequirement');
const calculateLossBtn = document.getElementById('calculateLossBtn');
const lossFunctionSelect = document.getElementById('lossFunctionSelect');
const lossContainer = document.getElementById('lossContainer');
// const backpropBtn = document.getElementById('backpropBtn');  // Removed section
// const backpropContainer = document.getElementById('backpropContainer');  // Removed section
const updateWeightsBtn = document.getElementById('updateWeightsBtn');
const updateWeightsContainer = document.getElementById('updateWeightsContainer');

// Event Listeners
buildNetworkBtn.addEventListener('click', buildNetwork);
quickStartMultiClassBtn.addEventListener('click', quickStartMultiClass);
quickStartBinaryBtn.addEventListener('click', quickStartBinary);
loadExampleBtn.addEventListener('click', loadExampleDataset);
forwardPassBtn.addEventListener('click', runForwardPass);
calculateLossBtn.addEventListener('click', calculateLoss);
// backpropBtn.addEventListener('click', runBackpropagation);  // Removed section
updateWeightsBtn.addEventListener('click', updateWeights);
trainBtn.addEventListener('click', startTraining);

// Dataset drag & drop
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);

// Validate dataset when it changes
datasetInput.addEventListener('input', validateDataset);

// Generate Layer Configuration UI
function generateLayerConfiguration() {
    const numLayers = parseInt(numLayersInput.value);

    if (numLayers < 2 || numLayers > 10) {
        alert('Please enter a number of layers between 2 and 10');
        return;
    }

    layersContainer.innerHTML = '';
    connectionsContainer.innerHTML = '';
    currentConfig.layers = [];
    currentConfig.connections = [];

    // Generate layer configuration inputs
    for (let i = 0; i < numLayers; i++) {
        const layerDiv = createLayerConfigUI(i, numLayers);
        layersContainer.appendChild(layerDiv);
    }

    // Add event listeners to update connections when layer sizes change
    document.querySelectorAll('.num-nodes-input').forEach(input => {
        input.addEventListener('change', updateConnectionsConfiguration);
    });

    updateConnectionsConfiguration();

    // Show and render the network diagram
    showNetworkDiagram();
}

// Show and render network diagram
function showNetworkDiagram() {
    const networkDiagramEl = document.getElementById('networkDiagram');
    const numNodesInputs = document.querySelectorAll('.num-nodes-input');
    const activationSelects = document.querySelectorAll('.activation-select');

    if (numNodesInputs.length === 0) return;

    // Get layer sizes and activations
    const layerSizes = Array.from(numNodesInputs).map(input => parseInt(input.value) || 0);
    const activations = Array.from(activationSelects).map(select => select.value);

    // Show diagram
    networkDiagramEl.style.display = 'block';

    // Render network
    if (networkDiagram) {
        networkDiagram.renderNetwork(layerSizes, activations);
    }
}

// Create Layer Configuration UI
function createLayerConfigUI(layerIndex, totalLayers) {
    const layerDiv = document.createElement('div');
    layerDiv.className = 'layer-config';
    layerDiv.id = `layer-${layerIndex}`;

    let layerType = 'Hidden Layer';
    if (layerIndex === 0) layerType = 'Input Layer';
    if (layerIndex === totalLayers - 1) layerType = 'Output Layer';

    layerDiv.innerHTML = `
        <h3>Layer ${layerIndex} - ${layerType}</h3>
        <div class="form-row">
            <div class="form-group">
                <label>Number of Nodes:</label>
                <input type="number" class="form-control num-nodes-input"
                       data-layer="${layerIndex}" min="1" max="100" value="${layerIndex === 0 ? 3 : layerIndex === totalLayers - 1 ? 2 : 4}">
            </div>
            <div class="form-group">
                <label>Activation Function:</label>
                <select class="form-control activation-select" data-layer="${layerIndex}">
                    <option value="linear" ${layerIndex === 0 ? 'selected' : ''}>Linear (Input)</option>
                    <option value="sigmoid" ${layerIndex > 0 ? 'selected' : ''}>Sigmoid</option>
                    <option value="relu">ReLU</option>
                    <option value="threshold">Threshold</option>
                </select>
            </div>
        </div>
    `;

    return layerDiv;
}

// Update Connections Configuration UI
function updateConnectionsConfiguration() {
    connectionsContainer.innerHTML = '';

    const numNodesInputs = document.querySelectorAll('.num-nodes-input');
    const layerSizes = Array.from(numNodesInputs).map(input => parseInt(input.value) || 0);

    // Generate connections for each layer (starting from layer 1)
    for (let layerIdx = 1; layerIdx < layerSizes.length; layerIdx++) {
        const connectionDiv = createConnectionConfigUI(layerIdx, layerSizes);
        connectionsContainer.appendChild(connectionDiv);
    }

    // Update visual diagram
    showNetworkDiagram();
}

// Create Connection Configuration UI
function createConnectionConfigUI(layerIndex, layerSizes) {
    const connectionDiv = document.createElement('div');
    connectionDiv.className = 'connection-config';
    connectionDiv.id = `connections-layer-${layerIndex}`;

    const prevLayerSize = layerSizes[layerIndex - 1];
    const currentLayerSize = layerSizes[layerIndex];

    let html = `<h3>Connections: Layer ${layerIndex - 1} → Layer ${layerIndex}</h3>`;

    for (let nodeIdx = 0; nodeIdx < currentLayerSize; nodeIdx++) {
        html += `
            <div class="node-connections">
                <h4>Node ${nodeIdx} in Layer ${layerIndex}</h4>
                <div class="form-group">
                    <label>Connect to nodes (comma-separated indices from Layer ${layerIndex - 1}, e.g., "0,1"):</label>
                    <input type="text" class="form-control connection-indices"
                           data-layer="${layerIndex}" data-node="${nodeIdx}"
                           value="${Array.from({length: prevLayerSize}, (_, i) => i).join(',')}"
                           placeholder="e.g., 0,1,2">
                </div>
                <div class="form-group">
                    <label>Weights (comma-separated, same order as connections):</label>
                    <input type="text" class="form-control connection-weights"
                           data-layer="${layerIndex}" data-node="${nodeIdx}"
                           value="${Array.from({length: prevLayerSize}, () => (Math.random() * 2 - 1).toFixed(3)).join(',')}"
                           placeholder="e.g., 0.5,-0.3,0.8">
                </div>
                <div class="form-group">
                    <label>Bias:</label>
                    <input type="number" class="form-control connection-bias"
                           data-layer="${layerIndex}" data-node="${nodeIdx}"
                           value="0" step="0.1">
                </div>
            </div>
        `;
    }

    connectionDiv.innerHTML = html;

    // Add event listeners to sync form changes to diagram
    setTimeout(() => {
        const inputs = connectionDiv.querySelectorAll('.connection-indices, .connection-weights, .connection-bias');
        inputs.forEach(input => {
            input.addEventListener('change', syncFormToDiagram);
        });
    }, 0);

    return connectionDiv;
}

// Sync form changes to diagram
function syncFormToDiagram() {
    if (!networkDiagram || networkDiagram.nodes.length === 0) return;

    const newConnections = [];

    // Parse all connections from form
    const connectionConfigs = document.querySelectorAll('.connection-config');

    connectionConfigs.forEach((configDiv, configIdx) => {
        const layerIdx = configIdx + 1;
        const nodeConnections = configDiv.querySelectorAll('.node-connections');

        nodeConnections.forEach((nodeDiv, nodeIdx) => {
            const indicesInput = nodeDiv.querySelector('.connection-indices');
            const weightsInput = nodeDiv.querySelector('.connection-weights');
            const biasInput = nodeDiv.querySelector('.connection-bias');

            if (!indicesInput || !weightsInput || !biasInput) return;

            // Parse connection indices
            const indices = indicesInput.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

            // Parse weights
            const weights = weightsInput.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));

            // Parse bias
            const bias = parseFloat(biasInput.value) || 0;

            // Create connections
            indices.forEach((fromIdx, i) => {
                const fromNode = networkDiagram.nodes.find(n => n.layer === layerIdx - 1 && n.index === fromIdx);
                const toNode = networkDiagram.nodes.find(n => n.layer === layerIdx && n.index === nodeIdx);

                if (fromNode && toNode) {
                    newConnections.push({
                        from: fromNode.id,
                        to: toNode.id,
                        weight: (weights[i] || 0).toFixed(3),
                        bias: bias,
                        element: null
                    });
                }
            });
        });
    });

    // Update diagram connections
    networkDiagram.connections = newConnections;
    networkDiagram.renderConnections();
}

// Build Network
async function buildNetwork() {
    try {
        if (!networkBuilder) {
            showError('Network builder not initialized');
            return;
        }

        // Get network configuration from the interactive builder
        const config = networkBuilder.getNetworkConfig();

        // Validate
        if (config.layers.length < 2) {
            showError('Network must have at least 2 layers (input and output)');
            return;
        }

        // Check if there are any connections
        let totalConnections = 0;
        config.connections.forEach(layerConn => {
            layerConn.connections.forEach(nodeConns => {
                totalConnections += nodeConns.length;
            });
        });

        if (totalConnections === 0) {
            showError('Network must have at least one connection. Drag from one node to another to create connections.');
            return;
        }

        // Send to backend
        buildNetworkBtn.disabled = true;
        buildNetworkBtn.innerHTML = '<span class="spinner"></span> Building...';

        const response = await fetch('/build_network', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                layers: config.layers,
                connections: config.connections
            })
        });

        const data = await response.json();

        if (data.success) {
            networkSummary.textContent = data.summary;
            networkSummary.classList.add('show');

            // Display classification info and auto-select loss
            displayClassificationInfo(data.classification_type, data.recommended_loss);

            showSuccess('Network built successfully!');

            // Update dataset requirement display
            updateDatasetRequirement();
        } else {
            showError('Error building network: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        buildNetworkBtn.disabled = false;
        buildNetworkBtn.textContent = 'Build Neural Network';
    }
}

// Quick Start with Multi-Class Example Network
async function quickStartMultiClass() {
    try {
        if (!networkBuilder) {
            showError('Network builder not initialized');
            return;
        }

        quickStartMultiClassBtn.disabled = true;
        quickStartMultiClassBtn.innerHTML = '<span class="spinner"></span> Loading...';

        const response = await fetch('/quick_start_multiclass', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            networkSummary.textContent = data.summary;
            networkSummary.classList.add('show');

            // Display classification info and auto-select loss
            displayClassificationInfo(data.classification_type, data.recommended_loss);

            showSuccess(data.message);

            // Load the example network into the interactive builder
            if (data.layers && data.connections) {
                networkBuilder.loadNetworkConfig(data.layers, data.connections);
            }

            // Update dataset requirement display
            updateDatasetRequirement();

            // Load example dataset from backend response
            if (data.example_dataset) {
                datasetInput.value = data.example_dataset;
                validateDataset();
                showSuccess('Multi-class example network and dataset loaded!');
            }
        } else {
            showError('Error: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        quickStartMultiClassBtn.disabled = false;
        quickStartMultiClassBtn.innerHTML = '📊 Multi-Class Example (3-4-2 Softmax)';
    }
}

// Quick Start with Binary Example Network
async function quickStartBinary() {
    try {
        if (!networkBuilder) {
            showError('Network builder not initialized');
            return;
        }

        quickStartBinaryBtn.disabled = true;
        quickStartBinaryBtn.innerHTML = '<span class="spinner"></span> Loading...';

        const response = await fetch('/quick_start_binary', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            networkSummary.textContent = data.summary;
            networkSummary.classList.add('show');

            // Display classification info and auto-select loss
            displayClassificationInfo(data.classification_type, data.recommended_loss);

            showSuccess(data.message);

            // Load the example network into the interactive builder
            if (data.layers && data.connections) {
                networkBuilder.loadNetworkConfig(data.layers, data.connections);
            }

            // Update dataset requirement display
            updateDatasetRequirement();

            // Load example dataset from backend response
            if (data.example_dataset) {
                datasetInput.value = data.example_dataset;
                validateDataset();
                showSuccess('Binary example network and dataset loaded!');
            }
        } else {
            showError('Error: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        quickStartBinaryBtn.disabled = false;
        quickStartBinaryBtn.innerHTML = '🎯 Binary Example (3-4-1 Sigmoid)';
    }
}

// ============================
// Dataset Management
// ============================

// Update dataset requirement based on network architecture
function updateDatasetRequirement() {
    if (!networkBuilder || !networkBuilder.layers || networkBuilder.layers.length < 2) {
        datasetRequirement.textContent = '⚠️ Build a network first to see dataset requirements';
        return;
    }

    const inputNodes = networkBuilder.layers[0].nodes;
    const outputNodes = networkBuilder.layers[networkBuilder.layers.length - 1].nodes;

    const inputCols = Array.from({length: inputNodes}, (_, i) => `x${i+1}`).join(', ');
    const outputCols = Array.from({length: outputNodes}, (_, i) => `y${i+1}`).join(', ');

    datasetRequirement.innerHTML = `
        ✅ <strong>Required format for current network:</strong>
        Inputs: <code>${inputCols}</code> |
        Outputs: <code>${outputCols}</code>
    `;
}

// Drag & Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        showValidationMessage('Please select a CSV file', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        const content = e.target.result;
        datasetInput.value = content;
        validateDataset();
        showSuccess(`File "${file.name}" loaded successfully!`);
    };
    reader.onerror = function() {
        showValidationMessage('Error reading file', 'error');
    };
    reader.readAsText(file);
}

// Validate Dataset
function validateDataset() {
    const csvData = datasetInput.value.trim();

    if (!csvData) {
        hideValidationMessage();
        hideDatasetStats();
        return { valid: false, message: 'No dataset provided' };
    }

    // Parse CSV
    const lines = csvData.split('\n').filter(line => line.trim());
    if (lines.length < 2) {
        showValidationMessage('Dataset must have at least a header and one data row', 'error');
        return { valid: false, message: 'Dataset must have at least a header and one data row' };
    }

    const headers = lines[0].split(',').map(h => h.trim());
    const dataRows = lines.slice(1);

    // Check if network is built
    if (!networkBuilder || !networkBuilder.layers || networkBuilder.layers.length < 2) {
        showValidationMessage('⚠️ Build a network first to validate dataset format', 'warning');
        showDatasetStats(headers.length, dataRows.length, headers);
        return { valid: false, message: 'Build a network first to validate dataset format' };
    }

    const inputNodes = networkBuilder.layers[0].nodes;
    const outputNodes = networkBuilder.layers[networkBuilder.layers.length - 1].nodes;

    // Expected columns
    const expectedInputs = Array.from({length: inputNodes}, (_, i) => `x${i+1}`);
    const expectedOutputs = Array.from({length: outputNodes}, (_, i) => `y${i+1}`);
    const expectedHeaders = [...expectedInputs, ...expectedOutputs];

    // Validate column count
    if (headers.length !== expectedHeaders.length) {
        const msg = `❌ Column count mismatch! Expected ${expectedHeaders.length} columns (${inputNodes} inputs + ${outputNodes} outputs), but got ${headers.length}`;
        showValidationMessage(msg, 'error');
        showDatasetStats(headers.length, dataRows.length, headers);
        return { valid: false, message: msg };
    }

    // Validate column names
    let allMatch = true;
    const mismatches = [];

    for (let i = 0; i < expectedHeaders.length; i++) {
        if (headers[i] !== expectedHeaders[i]) {
            allMatch = false;
            mismatches.push(`Column ${i+1}: expected "${expectedHeaders[i]}", got "${headers[i]}"`);
        }
    }

    if (!allMatch) {
        const msg = `❌ Column names don't match!\n${mismatches.join('\n')}`;
        showValidationMessage(msg, 'error');
        showDatasetStats(headers.length, dataRows.length, headers);
        return { valid: false, message: msg };
    }

    // All validations passed
    showValidationMessage(
        `✅ Dataset format is correct! ${inputNodes} inputs (${expectedInputs.join(', ')}) and ${outputNodes} outputs (${expectedOutputs.join(', ')})`,
        'success'
    );
    showDatasetStats(headers.length, dataRows.length, headers);
    return { valid: true, message: 'Dataset format is correct' };
}

function showValidationMessage(message, type) {
    datasetValidation.textContent = message;
    datasetValidation.className = `validation-message show ${type}`;
}

function hideValidationMessage() {
    datasetValidation.className = 'validation-message';
}

function showDatasetStats(columnCount, rowCount, headers) {
    datasetStats.innerHTML = `
        <strong>Dataset Info:</strong>
        ${columnCount} columns (${headers.join(', ')}) × ${rowCount} rows
    `;
    datasetStats.classList.add('show');
}

function hideDatasetStats() {
    datasetStats.classList.remove('show');
}

// Load Example Dataset
function loadExampleDataset() {
    // Get current network configuration
    if (!networkBuilder || !networkBuilder.layers || networkBuilder.layers.length < 2) {
        showError('Please build a network first before loading example dataset');
        return;
    }

    const inputNodes = networkBuilder.layers[0].nodes;
    const outputNodes = networkBuilder.layers[networkBuilder.layers.length - 1].nodes;

    // Generate example dataset based on current network
    const inputCols = Array.from({length: inputNodes}, (_, i) => `x${i+1}`);
    const outputCols = Array.from({length: outputNodes}, (_, i) => `y${i+1}`);
    const headers = [...inputCols, ...outputCols].join(',');

    // Generate 10 sample rows
    let rows = [headers];
    for (let i = 0; i < 10; i++) {
        const inputValues = Array.from({length: inputNodes}, () => Math.floor(Math.random() * 100));
        const outputValues = Array.from({length: outputNodes}, () => Math.random() > 0.5 ? 1 : 0);
        rows.push([...inputValues, ...outputValues].join(','));
    }

    const exampleData = rows.join('\n');
    datasetInput.value = exampleData;
    validateDataset();
    showSuccess(`Example dataset loaded! (${inputNodes} inputs, ${outputNodes} outputs, 10 samples)`);
}

// ============================
// Forward Pass
// ============================

// Global variable to store forward pass data
let forwardPassData = null;

// Run Forward Pass
async function runForwardPass() {
    try {
        const dataset = datasetInput.value.trim();

        if (!dataset) {
            showError('Please enter a dataset');
            return;
        }

        // Validate dataset first
        const validation = validateDataset();
        if (!validation.valid) {
            showError('Please fix dataset validation errors first');
            return;
        }

        forwardPassBtn.disabled = true;
        forwardPassBtn.innerHTML = '<span class="spinner"></span> Running Forward Pass...';

        const response = await fetch('/forward_pass', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset: dataset
            })
        });

        const data = await response.json();

        if (data.success) {
            // Store data globally
            forwardPassData = data;

            // Display all results in table
            displayForwardPassSummary(null, data);
            displayForwardPass(null, data);
            showSuccess(`Forward pass completed! ${data.num_samples} samples analyzed.`);
        } else {
            showError('Error: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        forwardPassBtn.disabled = false;
        forwardPassBtn.textContent = 'Run Forward Pass';
    }
}

// Display Forward Pass Summary
function displayForwardPassSummary(sample, allData) {
    const { feature_names, target_names } = allData;

    let html = `
        <h3>🧠 Forward Pass Results</h3>
        <p class="summary-description">Forward propagation completed for ${allData.num_samples} samples. Predictions (ŷ) shown in the table below.</p>
    `;

    forwardPassSummary.innerHTML = html;
    forwardPassSummary.style.display = 'block';
}

// Display Forward Pass Results as Table
function displayForwardPass(sample, allData) {
    const { feature_names, target_names, samples } = allData;

    let html = '<div class="forward-pass-table-container">';
    html += '<table class="forward-pass-table">';

    // Table header
    html += '<thead><tr>';
    html += '<th>Row</th>';

    // Input columns
    feature_names.forEach(name => {
        html += `<th>${name}</th>`;
    });

    // Expected output columns
    target_names.forEach(name => {
        html += `<th class="target-col">${name}</th>`;
    });

    // Prediction columns
    target_names.forEach((name, idx) => {
        html += `<th class="prediction-col">ŷ${idx + 1}</th>`;
    });

    html += '</tr></thead>';

    // Table body
    html += '<tbody>';
    samples.forEach((sampleData) => {
        html += '<tr>';
        html += `<td class="row-number">${sampleData.sample_index + 1}</td>`;

        // Input values
        sampleData.input.forEach(val => {
            html += `<td>${val.toFixed(2)}</td>`;
        });

        // Expected output values
        sampleData.target.forEach(val => {
            html += `<td class="target-value">${val.toFixed(2)}</td>`;
        });

        // Prediction values
        sampleData.prediction.forEach(val => {
            html += `<td class="prediction-value">${val.toFixed(4)}</td>`;
        });

        html += '</tr>';
    });
    html += '</tbody>';

    html += '</table>';
    html += '</div>';

    forwardPassContainer.innerHTML = html;
}

// Toggle section collapse
function toggleSection(sectionId) {
    const content = document.getElementById(sectionId);
    const header = content.previousElementSibling;

    if (content.style.display === 'none') {
        content.style.display = 'block';
        header.classList.remove('collapsed');
    } else {
        content.style.display = 'none';
        header.classList.add('collapsed');
    }
}

// Toggle layer collapse
function toggleLayer(layerId) {
    const content = document.getElementById(layerId);
    const header = content.previousElementSibling;

    if (content.style.display === 'none') {
        content.style.display = 'block';
        header.classList.remove('collapsed');
    } else {
        content.style.display = 'none';
        header.classList.add('collapsed');
    }
}

// ============================
// Loss Function
// ============================

// Calculate Loss
async function calculateLoss() {
    try {
        // Check if forward pass has been run
        if (!forwardPassData) {
            showError('Please run Forward Pass first before calculating loss');
            return;
        }

        const dataset = datasetInput.value.trim();
        if (!dataset) {
            showError('Please enter a dataset');
            return;
        }

        const lossFunction = lossFunctionSelect.value;

        calculateLossBtn.disabled = true;
        calculateLossBtn.innerHTML = '<span class="spinner"></span> Calculating Loss...';

        const response = await fetch('/calculate_loss', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset: dataset,
                loss_function: lossFunction
            })
        });

        const data = await response.json();

        if (data.success) {
            displayLoss(data);
            showSuccess('Loss calculated successfully!');
        } else {
            showError('Error calculating loss: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        calculateLossBtn.disabled = false;
        calculateLossBtn.textContent = 'Calculate Loss';
    }
}

// Display Loss Results
function displayLoss(data) {
    const { loss_function, total_loss, sample_losses } = data;

    // Map loss function codes to display names
    const lossNames = {
        'mse': 'Mean Squared Error (MSE)',
        'binary': 'Binary Cross-Entropy (BCE)',
        'categorical': 'Categorical Cross-Entropy (CCE)'
    };

    let html = '<div class="loss-results">';

    // Summary section
    html += '<div class="loss-summary-card">';
    html += '<h3>📊 Loss Calculation Results</h3>';
    html += `<p class="loss-function-name">Loss Function: <strong>${lossNames[loss_function] || loss_function}</strong></p>`;
    html += `<div class="total-loss-display">`;
    html += `<span class="loss-label">Total Loss:</span>`;
    html += `<span class="loss-value">${total_loss.toFixed(6)}</span>`;
    html += `</div>`;
    html += '</div>';

    // Per-sample loss table
    html += '<div class="loss-table-container">';
    html += '<h4>Loss Per Sample</h4>';
    html += '<table class="loss-table">';
    html += '<thead><tr>';
    html += '<th>Row</th>';
    html += '<th>Loss</th>';
    html += '</tr></thead>';
    html += '<tbody>';

    sample_losses.forEach((lossData) => {
        html += '<tr>';
        html += `<td class="row-number">${lossData.sample_index + 1}</td>`;
        html += `<td class="loss-value-cell">${lossData.loss.toFixed(6)}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    html += '</div>';
    html += '</div>';

    lossContainer.innerHTML = html;
}

// ============================
// Backpropagation (REMOVED - replaced by Update Weights section)
// ============================

/*
// Functions commented out as Backpropagation section was removed from UI
// The functionality is now integrated into Update Weights section

async function runBackpropagation() {
    // Function not used - Backpropagation section removed
}

function displayBackpropagation(data) {
    // Function not used - Backpropagation section removed
}
*/

// ============================
// Update Weights (Optimizer)
// ============================

// Update Weights
async function updateWeights() {
    try {
        // Check if forward pass has been run
        if (!forwardPassData) {
            showError('Please run Forward Pass first before updating weights');
            return;
        }

        const dataset = datasetInput.value.trim();
        if (!dataset) {
            showError('Please enter a dataset');
            return;
        }

        const lossFunction = lossFunctionSelect.value;
        const optimizer = document.getElementById('updateOptimizer').value;
        const learningRate = parseFloat(document.getElementById('updateLearningRate').value);

        // Validation
        if (learningRate <= 0 || learningRate > 10) {
            showError('Learning rate must be between 0 and 10');
            return;
        }

        updateWeightsBtn.disabled = true;
        updateWeightsBtn.innerHTML = '<span class="spinner"></span> Updating Weights...';

        const response = await fetch('/update_weights', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset: dataset,
                loss_function: lossFunction,
                optimizer: optimizer,
                learning_rate: learningRate
            })
        });

        const data = await response.json();

        if (data.success) {
            displayUpdateWeights(data);
            showSuccess('Weights updated successfully! (1 epoch completed)');
        } else {
            showError('Error updating weights: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        updateWeightsBtn.disabled = false;
        updateWeightsBtn.textContent = 'Update Weights (1 Epoch)';
    }
}

// Display Update Weights Results
function displayUpdateWeights(data) {
    const { optimizer, learning_rate, loss_before, loss_after, loss_reduction, old_weights, new_weights, weight_changes, bias_changes } = data;

    let html = '<div class="update-weights-results">';

    // Header with epoch info
    html += '<div class="update-header">';
    html += '<h3>✅ 1 Epoch Completed!</h3>';
    html += `<p class="epoch-description">Forward Pass → Loss Calculation → Backpropagation → Weight Update</p>`;
    html += '</div>';

    // Summary section
    html += '<div class="update-summary-card">';
    html += '<h4>📊 Training Summary</h4>';
    html += '<div class="summary-grid">';
    html += `<div class="summary-item">
        <span class="summary-label">Optimizer:</span>
        <span class="summary-value">${optimizer.toUpperCase()}</span>
    </div>`;
    html += `<div class="summary-item">
        <span class="summary-label">Learning Rate (α):</span>
        <span class="summary-value">${learning_rate}</span>
    </div>`;
    html += `<div class="summary-item">
        <span class="summary-label">Loss Before:</span>
        <span class="summary-value">${loss_before.toFixed(6)}</span>
    </div>`;
    html += `<div class="summary-item">
        <span class="summary-label">Loss After:</span>
        <span class="summary-value loss-improved">${loss_after.toFixed(6)}</span>
    </div>`;
    html += `<div class="summary-item">
        <span class="summary-label">Loss Reduction:</span>
        <span class="summary-value loss-improved">${loss_reduction.toFixed(6)} (${((loss_reduction / loss_before) * 100).toFixed(2)}%)</span>
    </div>`;
    html += '</div>';
    html += '</div>';

    // Weight changes per layer
    html += '<div class="weight-changes-section">';
    html += '<h4>🔄 Weight & Bias Changes</h4>';

    for (const [layerKey, layerWeightChanges] of Object.entries(weight_changes)) {
        const layerIdx = layerKey.split('_')[1];
        const layerBiasChanges = bias_changes[layerKey];

        html += `<div class="layer-weight-changes">`;
        html += `<h5>Layer ${layerIdx}</h5>`;
        html += '<table class="weight-changes-table">';
        html += '<thead><tr>';
        html += '<th>Node</th>';
        html += '<th>Old Weights</th>';
        html += '<th>New Weights</th>';
        html += '<th>ΔW (Change)</th>';
        html += '<th>Old Bias</th>';
        html += '<th>New Bias</th>';
        html += '<th>Δb (Change)</th>';
        html += '</tr></thead>';
        html += '<tbody>';

        layerWeightChanges.forEach((nodeChanges, nodeIdx) => {
            html += '<tr>';
            html += `<td class="node-label">Node ${nodeIdx}</td>`;

            // Old weights
            html += '<td class="weights-cell">[';
            html += old_weights[layerKey][nodeIdx].map(w => w.toFixed(4)).join(', ');
            html += ']</td>';

            // New weights
            html += '<td class="weights-cell">[';
            html += new_weights[layerKey][nodeIdx].map(w => w.toFixed(4)).join(', ');
            html += ']</td>';

            // Weight changes
            html += '<td class="changes-cell">[';
            html += nodeChanges.map(c => {
                const sign = c >= 0 ? '+' : '';
                const colorClass = c >= 0 ? 'change-positive' : 'change-negative';
                return `<span class="${colorClass}">${sign}${c.toFixed(4)}</span>`;
            }).join(', ');
            html += ']</td>';

            // Old bias
            const oldBias = old_weights[layerKey + '_biases'] ? old_weights[layerKey + '_biases'][nodeIdx] : (typeof old_weights[layerKey][nodeIdx] === 'number' ? old_weights[layerKey][nodeIdx] : 0);
            html += `<td>N/A</td>`;

            // New bias
            const newBias = new_weights[layerKey + '_biases'] ? new_weights[layerKey + '_biases'][nodeIdx] : (typeof new_weights[layerKey][nodeIdx] === 'number' ? new_weights[layerKey][nodeIdx] : 0);
            html += `<td>N/A</td>`;

            // Bias change
            const biasChange = layerBiasChanges[nodeIdx];
            const biasSign = biasChange >= 0 ? '+' : '';
            const biasColorClass = biasChange >= 0 ? 'change-positive' : 'change-negative';
            html += `<td><span class="${biasColorClass}">${biasSign}${biasChange.toFixed(4)}</span></td>`;

            html += '</tr>';
        });

        html += '</tbody></table>';
        html += '</div>';
    }

    html += '</div>';

    // Info box
    html += '<div class="info-box">';
    html += '<strong>💡 What happened?</strong><br>';
    html += `Using <strong>${optimizer.toUpperCase()}</strong> optimizer with learning rate <strong>${learning_rate}</strong>, `;
    html += 'the weights were adjusted in the direction that reduces loss.<br>';
    html += 'Positive changes (green) mean weights increased, negative changes (red) mean weights decreased.';
    html += '</div>';

    html += '</div>';

    updateWeightsContainer.innerHTML = html;
}

// NOTE: makePredictions and displayResults functions are commented out
// because "Prediction Results" section was removed from UI
// These functions are kept for potential future use

/*
// Make Predictions
async function makePredictions() {
    // Function not used - Prediction Results section removed
}

// Display Results
function displayResults(data) {
    // Function not used - Prediction Results section removed
}
*/

// Display Classification Info and Auto-Select Loss Function
function displayClassificationInfo(classificationType, recommendedLoss) {
    const classificationInfo = document.getElementById('classificationInfo');
    const classificationTypeSpan = document.getElementById('classificationType');
    const recommendedLossSpan = document.getElementById('recommendedLoss');
    const lossFunctionSelect = document.getElementById('lossFunctionSelect');

    // Classification type descriptions
    const typeDescriptions = {
        'binary': 'Binary (Single Output)',
        'multi-label': 'Multi-Label (Multiple Independent Outputs)',
        'multi-class': 'Multi-Class (Multiple Mutually Exclusive Classes)'
    };

    // Loss function names
    const lossNames = {
        'mse': 'Mean Squared Error (MSE)',
        'binary': 'Binary Cross-Entropy (BCE)',
        'categorical': 'Categorical Cross-Entropy (CCE)'
    };

    // Display classification info
    classificationTypeSpan.textContent = typeDescriptions[classificationType] || classificationType;
    recommendedLossSpan.textContent = lossNames[recommendedLoss] || recommendedLoss;
    classificationInfo.style.display = 'block';

    // Auto-select recommended loss function
    lossFunctionSelect.value = recommendedLoss;
}

// Helper Functions
function showError(message) {
    const existingError = document.querySelector('.error-message');
    if (existingError) existingError.remove();

    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;

    const section = document.querySelector('.config-section');
    section.appendChild(errorDiv);

    setTimeout(() => errorDiv.remove(), 5000);
}

function showSuccess(message) {
    const existingSuccess = document.querySelector('.success-message');
    if (existingSuccess) existingSuccess.remove();

    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.textContent = message;

    const section = document.querySelector('.config-section');
    section.appendChild(successDiv);

    setTimeout(() => successDiv.remove(), 3000);
}

// Train Neural Network
async function startTraining() {
    try {
        const dataset = datasetInput.value.trim();
        const lossFunction = lossFunctionSelect.value;  // Fixed: use lossFunctionSelect
        const optimizer = document.getElementById('optimizer').value;
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSizeInput = document.getElementById('batchSize').value;
        const batchSize = batchSizeInput ? parseInt(batchSizeInput) : null;

        if (!dataset) {
            showError('Please enter a dataset');
            return;
        }

        // Validation
        if (learningRate <= 0 || learningRate > 10) {
            showError('Learning rate must be between 0 and 10');
            return;
        }

        if (epochs < 1 || epochs > 10000) {
            showError('Epochs must be between 1 and 10000');
            return;
        }

        // Show progress and clear previous results
        trainingProgress.style.display = 'block';
        trainingResults.innerHTML = '';
        evaluationResults.innerHTML = '';
        evaluationResultsSection.style.display = 'none';  // Hide evaluation section initially

        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="spinner"></span> Training...';

        const startTime = Date.now();

        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset: dataset,
                loss_function: lossFunction,
                optimizer: optimizer,
                learning_rate: learningRate,
                epochs: epochs,
                batch_size: batchSize
            })
        });

        const data = await response.json();

        if (data.success) {
            const duration = ((Date.now() - startTime) / 1000).toFixed(2);
            displayTrainingResults(data, duration);

            // Display evaluation metrics in separate section
            if (data.evaluation) {
                displayModelEvaluation(data.evaluation);
            }

            showSuccess('Training completed successfully!');
        } else {
            showError('Error during training: ' + data.error);
            if (data.traceback) {
                console.error(data.traceback);
            }
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        trainingProgress.style.display = 'none';
        trainBtn.disabled = false;
        trainBtn.textContent = 'Start Training';
    }
}

// Display Training Results
function displayTrainingResults(data, duration) {
    let html = '<h3>Training Results</h3>';

    // Summary metrics
    html += '<div class="training-summary">';
    html += `<div class="metric-card">
        <h4>Training Time</h4>
        <div class="value">${duration}s</div>
    </div>`;
    html += `<div class="metric-card">
        <h4>Final Loss</h4>
        <div class="value">${data.final_loss.toFixed(6)}</div>
    </div>`;
    html += `<div class="metric-card">
        <h4>Accuracy</h4>
        <div class="value">${(data.accuracy * 100).toFixed(2)}%</div>
    </div>`;
    html += `<div class="metric-card">
        <h4>Epochs</h4>
        <div class="value">${data.history.epochs.length}</div>
    </div>`;
    html += '</div>';

    // Loss curve visualization
    html += '<div class="loss-curve">';
    html += '<h4>Loss Curve</h4>';
    html += renderLossCurve(data.history);
    html += '</div>';

    // Note: Evaluation metrics moved to separate section

    // Updated weights
    html += '<div class="updated-weights">';
    html += '<h4>Updated Weights & Biases</h4>';
    html += '<div class="weights-container">';

    for (const [layerName, weights] of Object.entries(data.updated_weights)) {
        const layerIdx = layerName.split('_')[1];
        const biases = data.updated_biases[layerName];

        html += `<div class="layer-weights">`;
        html += `<h5>${layerName.toUpperCase()}</h5>`;

        weights.forEach((nodeWeights, nodeIdx) => {
            html += `<div class="node-weights">`;
            html += `<strong>Node ${nodeIdx}:</strong><br>`;
            html += `Weights: [${nodeWeights.map(w => w.toFixed(4)).join(', ')}]<br>`;
            html += `Bias: ${biases[nodeIdx].toFixed(4)}`;
            html += `</div>`;
        });

        html += `</div>`;
    }

    html += '</div>';
    html += '</div>';

    // Predictions comparison
    html += '<div class="predictions-comparison">';
    html += '<h4>Predictions After Training</h4>';
    html += '<table class="results-table">';

    // Check if multi-output
    const isMultiOutput = data.num_outputs > 1;

    if (isMultiOutput) {
        // Multi-output header
        html += '<thead><tr><th>#</th><th>True Labels</th><th>Predicted Labels</th><th>Probabilities</th><th>Status</th></tr></thead>';
    } else {
        // Single output header
        html += '<thead><tr><th>#</th><th>True</th><th>Predicted</th><th>Probability</th><th>Status</th></tr></thead>';
    }

    html += '<tbody>';

    data.predictions.forEach((pred, idx) => {
        let correct;
        if (isMultiOutput) {
            // Multi-output: check if arrays are equal
            correct = JSON.stringify(pred.y_true) === JSON.stringify(pred.y_pred_classes);
        } else {
            // Single output: direct comparison
            correct = pred.y_true === pred.y_pred_classes;
        }

        const statusClass = correct ? 'status-correct' : 'status-incorrect';
        const statusText = correct ? 'Correct' : 'Incorrect';

        html += `<tr>`;
        html += `<td>${idx + 1}</td>`;

        if (isMultiOutput) {
            // Display arrays
            html += `<td>[${pred.y_true.join(', ')}]</td>`;
            html += `<td>[${pred.y_pred_classes.join(', ')}]</td>`;
            html += `<td>[${pred.y_pred.map(p => p.toFixed(4)).join(', ')}]</td>`;
        } else {
            // Display single values
            html += `<td>${pred.y_true}</td>`;
            html += `<td>${pred.y_pred_classes}</td>`;
            html += `<td>${pred.y_pred.toFixed(4)}</td>`;
        }

        html += `<td><span class="status-badge ${statusClass}">${statusText}</span></td>`;
        html += `</tr>`;
    });

    html += '</tbody></table>';
    html += '</div>';

    trainingResults.innerHTML = html;
}

// Display Model Evaluation in separate section
function displayModelEvaluation(evaluation) {
    // Show the evaluation section
    evaluationResultsSection.style.display = 'block';

    // Render evaluation metrics
    evaluationResults.innerHTML = renderEvaluationMetrics(evaluation);

    // Scroll to evaluation section
    setTimeout(() => {
        evaluationResultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

// Render Evaluation Metrics
function renderEvaluationMetrics(evaluation) {
    const { confusion_matrix, metrics_per_class, macro_avg } = evaluation;

    let html = '<div class="evaluation-section">';
    html += '<h4>📊 Model Evaluation</h4>';

    // Macro Average Metrics
    html += '<div class="macro-metrics">';
    html += '<h5>Overall Performance (Macro Average)</h5>';
    html += '<div class="macro-metrics-grid">';
    html += `<div class="macro-metric-card">
        <div class="metric-label">Precision</div>
        <div class="metric-value">${(macro_avg.precision * 100).toFixed(2)}%</div>
    </div>`;
    html += `<div class="macro-metric-card">
        <div class="metric-label">Recall</div>
        <div class="metric-value">${(macro_avg.recall * 100).toFixed(2)}%</div>
    </div>`;
    html += `<div class="macro-metric-card">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value">${(macro_avg.f1_score * 100).toFixed(2)}%</div>
    </div>`;
    html += '</div>';
    html += '</div>';

    // Confusion Matrix
    html += '<div class="confusion-matrix-section">';
    html += '<h5>Confusion Matrix</h5>';
    html += '<div class="confusion-matrix-container">';
    html += '<table class="confusion-matrix">';

    // Header
    const classes = Object.keys(confusion_matrix).sort((a, b) => parseInt(a) - parseInt(b));
    html += '<thead><tr><th class="corner-cell">Actual \\ Predicted</th>';
    classes.forEach(cls => {
        html += `<th class="pred-header">Class ${cls}</th>`;
    });
    html += '</tr></thead>';

    // Body
    html += '<tbody>';
    classes.forEach(trueClass => {
        html += '<tr>';
        html += `<th class="true-header">Class ${trueClass}</th>`;
        classes.forEach(predClass => {
            const count = confusion_matrix[trueClass][predClass];
            const cellClass = trueClass === predClass ? 'diagonal-cell' : 'off-diagonal-cell';
            html += `<td class="${cellClass}">${count}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    html += '</div>';

    // Legend
    html += '<div class="confusion-matrix-legend">';
    html += '<p><strong>How to read:</strong> Rows = Actual class, Columns = Predicted class</p>';
    html += '<p>Diagonal cells (green) = Correct predictions, Off-diagonal cells = Misclassifications</p>';
    html += '</div>';
    html += '</div>';

    // Per-Class Metrics
    html += '<div class="per-class-metrics">';
    html += '<h5>Per-Class Performance</h5>';
    html += '<table class="per-class-table">';
    html += '<thead><tr>';
    html += '<th>Class</th>';
    html += '<th>Precision</th>';
    html += '<th>Recall</th>';
    html += '<th>F1 Score</th>';
    html += '<th>Support</th>';
    html += '<th>TP</th>';
    html += '<th>FP</th>';
    html += '<th>FN</th>';
    html += '<th>TN</th>';
    html += '</tr></thead>';
    html += '<tbody>';

    classes.forEach(cls => {
        const metrics = metrics_per_class[cls];
        html += '<tr>';
        html += `<td class="class-label">Class ${cls}</td>`;
        html += `<td>${(metrics.precision * 100).toFixed(2)}%</td>`;
        html += `<td>${(metrics.recall * 100).toFixed(2)}%</td>`;
        html += `<td>${(metrics.f1_score * 100).toFixed(2)}%</td>`;
        html += `<td>${metrics.support}</td>`;
        html += `<td class="tp-cell">${metrics.true_positives}</td>`;
        html += `<td class="fp-cell">${metrics.false_positives}</td>`;
        html += `<td class="fn-cell">${metrics.false_negatives}</td>`;
        html += `<td class="tn-cell">${metrics.true_negatives}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    html += '</div>';

    // Metrics Explanation
    html += '<div class="metrics-explanation">';
    html += '<h5>📚 Metrics Explained</h5>';
    html += '<ul>';
    html += '<li><strong>Precision:</strong> Of all predicted positive cases, how many were actually positive? (TP / (TP + FP))</li>';
    html += '<li><strong>Recall:</strong> Of all actual positive cases, how many did we correctly identify? (TP / (TP + FN))</li>';
    html += '<li><strong>F1 Score:</strong> Harmonic mean of precision and recall (2 × (Precision × Recall) / (Precision + Recall))</li>';
    html += '<li><strong>Support:</strong> Number of actual occurrences of each class in the dataset</li>';
    html += '<li><strong>TP:</strong> True Positives - Correctly predicted positive</li>';
    html += '<li><strong>FP:</strong> False Positives - Incorrectly predicted positive</li>';
    html += '<li><strong>FN:</strong> False Negatives - Incorrectly predicted negative</li>';
    html += '<li><strong>TN:</strong> True Negatives - Correctly predicted negative</li>';
    html += '</ul>';
    html += '</div>';

    html += '</div>';

    return html;
}

// Render Loss Curve as ASCII-style chart
function renderLossCurve(history) {
    const epochs = history.epochs;
    const losses = history.loss;

    if (epochs.length === 0) return '<p>No loss data available</p>';

    // Sample data for visualization (show every Nth point if too many)
    const maxPoints = 50;
    const step = Math.max(1, Math.floor(epochs.length / maxPoints));
    const sampledEpochs = [];
    const sampledLosses = [];

    for (let i = 0; i < epochs.length; i += step) {
        sampledEpochs.push(epochs[i]);
        sampledLosses.push(losses[i]);
    }

    // Add last point if not included
    if (sampledEpochs[sampledEpochs.length - 1] !== epochs[epochs.length - 1]) {
        sampledEpochs.push(epochs[epochs.length - 1]);
        sampledLosses.push(losses[losses.length - 1]);
    }

    // Create SVG chart
    const width = 600;
    const height = 300;
    const padding = 40;

    const minLoss = Math.min(...sampledLosses);
    const maxLoss = Math.max(...sampledLosses);
    const lossRange = maxLoss - minLoss || 1;

    const xScale = (epoch) => padding + ((epoch - 1) / epochs[epochs.length - 1]) * (width - 2 * padding);
    const yScale = (loss) => height - padding - ((loss - minLoss) / lossRange) * (height - 2 * padding);

    // Build path
    let pathData = `M ${xScale(sampledEpochs[0])} ${yScale(sampledLosses[0])}`;
    for (let i = 1; i < sampledEpochs.length; i++) {
        pathData += ` L ${xScale(sampledEpochs[i])} ${yScale(sampledLosses[i])}`;
    }

    let svg = `<svg width="${width}" height="${height}" class="loss-chart">`;

    // Grid lines
    svg += '<g class="grid">';
    for (let i = 0; i <= 5; i++) {
        const y = padding + (i / 5) * (height - 2 * padding);
        svg += `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="#e0e0e0" stroke-width="1"/>`;
    }
    svg += '</g>';

    // Loss line
    svg += `<path d="${pathData}" fill="none" stroke="#2196F3" stroke-width="2"/>`;

    // Axes
    svg += `<line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#333" stroke-width="2"/>`;
    svg += `<line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#333" stroke-width="2"/>`;

    // Labels
    svg += `<text x="${width / 2}" y="${height - 5}" text-anchor="middle" font-size="12">Epoch</text>`;
    svg += `<text x="15" y="${height / 2}" text-anchor="middle" font-size="12" transform="rotate(-90, 15, ${height / 2})">Loss</text>`;

    // Min/Max labels
    svg += `<text x="${padding - 5}" y="${yScale(maxLoss)}" text-anchor="end" font-size="10">${maxLoss.toFixed(4)}</text>`;
    svg += `<text x="${padding - 5}" y="${yScale(minLoss)}" text-anchor="end" font-size="10">${minLoss.toFixed(4)}</text>`;

    svg += '</svg>';

    // Add text summary
    svg += `<div class="loss-summary">`;
    svg += `<p><strong>Initial Loss:</strong> ${losses[0].toFixed(6)}</p>`;
    svg += `<p><strong>Final Loss:</strong> ${losses[losses.length - 1].toFixed(6)}</p>`;
    svg += `<p><strong>Improvement:</strong> ${(losses[0] - losses[losses.length - 1]).toFixed(6)} (${((losses[0] - losses[losses.length - 1]) / losses[0] * 100).toFixed(2)}%)</p>`;
    svg += `</div>`;

    return svg;
}

// ============================
// Network Diagram Visualization
// ============================

class NetworkDiagram {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.nodes = [];
        this.connections = [];
        this.layers = [];
        this.draggedNode = null;
        this.dragStartNode = null;
        this.tempLine = null;
        this.selectedConnection = null;

        this.nodeRadius = 25;
        this.layerSpacing = 200;
        this.nodeSpacing = 80;

        this.init();
    }

    init() {
        // Clear canvas
        this.canvas.innerHTML = '';
        this.canvas.style.height = '500px';

        // Add mouse/touch event listeners
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    renderNetwork(layerSizes, activations, existingConnections = null) {
        this.clear();
        this.layers = layerSizes;

        const canvasWidth = this.canvas.clientWidth;
        const canvasHeight = Math.max(500, Math.max(...layerSizes) * this.nodeSpacing + 100);
        this.canvas.style.height = canvasHeight + 'px';

        // Calculate layer positions
        const totalWidth = (layerSizes.length - 1) * this.layerSpacing;
        const startX = (canvasWidth - totalWidth) / 2;

        // Create nodes
        this.nodes = [];
        layerSizes.forEach((numNodes, layerIdx) => {
            const layerNodes = [];
            const layerHeight = (numNodes - 1) * this.nodeSpacing;
            const startY = (canvasHeight - layerHeight) / 2;

            for (let nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
                const node = {
                    id: `L${layerIdx}N${nodeIdx}`,
                    layer: layerIdx,
                    index: nodeIdx,
                    x: startX + layerIdx * this.layerSpacing,
                    y: startY + nodeIdx * this.nodeSpacing,
                    activation: activations[layerIdx] || 'sigmoid',
                    element: null
                };

                layerNodes.push(node);
                this.nodes.push(node);
            }
        });

        // Create default connections (fully connected) or use existing
        this.connections = [];
        if (existingConnections) {
            this.connections = existingConnections;
        } else {
            // Default: fully connected network
            for (let layerIdx = 1; layerIdx < layerSizes.length; layerIdx++) {
                const prevLayerNodes = this.nodes.filter(n => n.layer === layerIdx - 1);
                const currLayerNodes = this.nodes.filter(n => n.layer === layerIdx);

                currLayerNodes.forEach(toNode => {
                    prevLayerNodes.forEach(fromNode => {
                        this.connections.push({
                            from: fromNode.id,
                            to: toNode.id,
                            weight: (Math.random() * 2 - 1).toFixed(3),
                            bias: 0,
                            element: null
                        });
                    });
                });
            }
        }

        // Render
        this.renderConnections();
        this.renderNodes();
    }

    renderNodes() {
        this.nodes.forEach(node => {
            const nodeEl = document.createElement('div');
            nodeEl.className = 'network-node';
            nodeEl.textContent = node.index;
            nodeEl.style.left = (node.x - this.nodeRadius) + 'px';
            nodeEl.style.top = (node.y - this.nodeRadius) + 'px';
            nodeEl.dataset.nodeId = node.id;

            // Add layer-specific class
            if (node.layer === 0) {
                nodeEl.classList.add('input-layer');
            } else if (node.layer === this.layers.length - 1) {
                nodeEl.classList.add('output-layer');
            } else {
                nodeEl.classList.add('hidden-layer');
            }

            // Add label
            const label = document.createElement('div');
            label.className = 'node-label';
            label.textContent = `L${node.layer}N${node.index}`;
            nodeEl.appendChild(label);

            node.element = nodeEl;
            this.canvas.appendChild(nodeEl);
        });
    }

    renderConnections() {
        // Remove old connection elements
        this.canvas.querySelectorAll('.connection-line, .connection-weight').forEach(el => el.remove());

        this.connections.forEach((conn, idx) => {
            const fromNode = this.nodes.find(n => n.id === conn.from);
            const toNode = this.nodes.find(n => n.id === conn.to);

            if (!fromNode || !toNode) return;

            // Create SVG line
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.classList.add('connection-line');
            svg.dataset.connIndex = idx;
            svg.style.pointerEvents = 'none';

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', fromNode.x);
            line.setAttribute('y1', fromNode.y);
            line.setAttribute('x2', toNode.x);
            line.setAttribute('y2', toNode.y);
            line.style.pointerEvents = 'stroke';

            // Add click handler for editing
            line.addEventListener('click', (e) => {
                e.stopPropagation();
                this.openConnectionModal(idx);
            });

            svg.appendChild(line);

            // Position SVG to cover the line area
            const minX = Math.min(fromNode.x, toNode.x);
            const minY = Math.min(fromNode.y, toNode.y);
            const width = Math.abs(toNode.x - fromNode.x);
            const height = Math.abs(toNode.y - fromNode.y);

            svg.style.left = minX + 'px';
            svg.style.top = minY + 'px';
            svg.style.width = width + 'px';
            svg.style.height = height + 'px';
            svg.setAttribute('viewBox', `${minX} ${minY} ${width} ${height}`);

            conn.element = svg;
            this.canvas.insertBefore(svg, this.canvas.firstChild);

            // Add weight label
            const weightLabel = document.createElement('div');
            weightLabel.className = 'connection-weight';
            weightLabel.textContent = `w:${conn.weight}`;
            weightLabel.style.left = ((fromNode.x + toNode.x) / 2 - 20) + 'px';
            weightLabel.style.top = ((fromNode.y + toNode.y) / 2 - 10) + 'px';
            weightLabel.dataset.connIndex = idx;
            this.canvas.appendChild(weightLabel);
        });
    }

    handleMouseDown(e) {
        const nodeEl = e.target.closest('.network-node');
        if (!nodeEl) return;

        const nodeId = nodeEl.dataset.nodeId;
        const node = this.nodes.find(n => n.id === nodeId);

        this.dragStartNode = node;
        nodeEl.classList.add('dragging');
    }

    handleMouseMove(e) {
        if (!this.dragStartNode) return;

        // Remove existing temp line
        if (this.tempLine) {
            this.tempLine.remove();
            this.tempLine = null;
        }

        // Get mouse position relative to canvas
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Create temporary line
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('temp-connection');
        svg.style.position = 'absolute';
        svg.style.left = '0';
        svg.style.top = '0';
        svg.style.width = '100%';
        svg.style.height = '100%';
        svg.style.pointerEvents = 'none';

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', this.dragStartNode.x);
        line.setAttribute('y1', this.dragStartNode.y);
        line.setAttribute('x2', mouseX);
        line.setAttribute('y2', mouseY);

        svg.appendChild(line);
        this.tempLine = svg;
        this.canvas.appendChild(svg);

        // Highlight valid target nodes
        this.nodes.forEach(n => {
            if (n.element) {
                n.element.classList.remove('drag-target');
            }
        });

        const targetNode = this.getNodeAtPosition(mouseX, mouseY);
        if (targetNode && this.isValidConnection(this.dragStartNode, targetNode)) {
            targetNode.element.classList.add('drag-target');
        }
    }

    handleMouseUp(e) {
        if (!this.dragStartNode) return;

        // Remove temp line
        if (this.tempLine) {
            this.tempLine.remove();
            this.tempLine = null;
        }

        // Remove dragging state
        this.nodes.forEach(n => {
            if (n.element) {
                n.element.classList.remove('dragging', 'drag-target');
            }
        });

        // Get target node
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const targetNode = this.getNodeAtPosition(mouseX, mouseY);

        if (targetNode && this.isValidConnection(this.dragStartNode, targetNode)) {
            // Check if connection already exists
            const existingIdx = this.connections.findIndex(
                c => c.from === this.dragStartNode.id && c.to === targetNode.id
            );

            if (existingIdx >= 0) {
                // Connection exists, open modal to edit or remove
                this.openConnectionModal(existingIdx);
            } else {
                // Create new connection
                this.connections.push({
                    from: this.dragStartNode.id,
                    to: targetNode.id,
                    weight: (Math.random() * 2 - 1).toFixed(3),
                    bias: 0,
                    element: null
                });
                this.renderConnections();
                this.syncToForm();
                showSuccess('Connection created! Click on the line to edit weight/bias.');
            }
        }

        this.dragStartNode = null;
    }

    getNodeAtPosition(x, y) {
        for (let node of this.nodes) {
            const dx = node.x - x;
            const dy = node.y - y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance <= this.nodeRadius) {
                return node;
            }
        }
        return null;
    }

    isValidConnection(fromNode, toNode) {
        // Can only connect from earlier layer to later layer
        return fromNode.layer < toNode.layer && fromNode.id !== toNode.id;
    }

    openConnectionModal(connIndex) {
        this.selectedConnection = connIndex;
        const conn = this.connections[connIndex];

        const modal = document.getElementById('connectionModal');
        const connectionInfo = document.getElementById('connectionInfo');
        const weightInput = document.getElementById('modalWeight');
        const biasInput = document.getElementById('modalBias');

        connectionInfo.textContent = `Connection: ${conn.from} → ${conn.to}`;
        weightInput.value = conn.weight;
        biasInput.value = conn.bias;

        modal.classList.add('show');
    }

    closeConnectionModal() {
        const modal = document.getElementById('connectionModal');
        modal.classList.remove('show');
        this.selectedConnection = null;
    }

    saveConnection() {
        if (this.selectedConnection === null) return;

        const weightInput = document.getElementById('modalWeight');
        const biasInput = document.getElementById('modalBias');

        this.connections[this.selectedConnection].weight = parseFloat(weightInput.value).toFixed(3);
        this.connections[this.selectedConnection].bias = parseFloat(biasInput.value);

        this.renderConnections();
        this.syncToForm();
        this.closeConnectionModal();
        showSuccess('Connection updated!');
    }

    deleteConnection() {
        if (this.selectedConnection === null) return;

        this.connections.splice(this.selectedConnection, 1);
        this.renderConnections();
        this.syncToForm();
        this.closeConnectionModal();
        showSuccess('Connection deleted!');
    }

    syncToForm() {
        // Update the connection form inputs based on diagram state
        const connectionsContainer = document.getElementById('connectionsContainer');

        // Group connections by target layer and node
        const connectionsByTarget = {};

        this.connections.forEach(conn => {
            const toNode = this.nodes.find(n => n.id === conn.to);
            const fromNode = this.nodes.find(n => n.id === conn.from);

            if (!toNode || !fromNode) return;

            const key = `${toNode.layer}-${toNode.index}`;
            if (!connectionsByTarget[key]) {
                connectionsByTarget[key] = {
                    layer: toNode.layer,
                    node: toNode.index,
                    indices: [],
                    weights: [],
                    bias: conn.bias
                };
            }

            connectionsByTarget[key].indices.push(fromNode.index);
            connectionsByTarget[key].weights.push(conn.weight);
        });

        // Update form inputs
        Object.values(connectionsByTarget).forEach(connData => {
            const indicesInput = document.querySelector(
                `.connection-indices[data-layer="${connData.layer}"][data-node="${connData.node}"]`
            );
            const weightsInput = document.querySelector(
                `.connection-weights[data-layer="${connData.layer}"][data-node="${connData.node}"]`
            );
            const biasInput = document.querySelector(
                `.connection-bias[data-layer="${connData.layer}"][data-node="${connData.node}"]`
            );

            if (indicesInput) indicesInput.value = connData.indices.join(',');
            if (weightsInput) weightsInput.value = connData.weights.join(',');
            if (biasInput) biasInput.value = connData.bias;
        });
    }

    clear() {
        this.canvas.innerHTML = '';
        this.nodes = [];
        this.connections = [];
        this.layers = [];
    }
}

// Create global diagram instance
let networkDiagram = null;

// Modal Event Handlers
const connectionModal = document.getElementById('connectionModal');
const modalClose = document.querySelector('.modal-close');
const saveConnectionBtn = document.getElementById('saveConnectionBtn');
const deleteConnectionBtn = document.getElementById('deleteConnectionBtn');
const cancelConnectionBtn = document.getElementById('cancelConnectionBtn');

if (modalClose) {
    modalClose.addEventListener('click', () => {
        if (networkDiagram) networkDiagram.closeConnectionModal();
    });
}

if (saveConnectionBtn) {
    saveConnectionBtn.addEventListener('click', () => {
        if (networkDiagram) networkDiagram.saveConnection();
    });
}

if (deleteConnectionBtn) {
    deleteConnectionBtn.addEventListener('click', () => {
        if (networkDiagram) networkDiagram.deleteConnection();
    });
}

if (cancelConnectionBtn) {
    cancelConnectionBtn.addEventListener('click', () => {
        if (networkDiagram) networkDiagram.closeConnectionModal();
    });
}

// Close modal when clicking outside
if (connectionModal) {
    connectionModal.addEventListener('click', (e) => {
        if (e.target === connectionModal) {
            if (networkDiagram) networkDiagram.closeConnectionModal();
        }
    });
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    console.log('ANN from Scratch - Ready!');

    // Initialize network diagram
    networkDiagram = new NetworkDiagram('diagramCanvas');
});
