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
document.getElementById('saveDatasetBtn').addEventListener('click', saveDatasetAndContinue);
forwardPassBtn.addEventListener('click', runForwardPass);
calculateLossBtn.addEventListener('click', calculateLoss);
// backpropBtn.addEventListener('click', runBackpropagation);  // Removed section
if (updateWeightsBtn) {
    updateWeightsBtn.addEventListener('click', updateWeights);
}
document.getElementById('run1EpochBtn').addEventListener('click', run1Epoch);
trainBtn.addEventListener('click', startTraining);

// Dataset drag & drop
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);

// Validate dataset when it changes
datasetInput.addEventListener('input', validateDataset);

// ============================
// Old UI Functions (Removed)
// ============================
// These functions were part of the old manual network configuration UI
// Now replaced by InteractiveNetworkBuilder in network-builder.js

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
            // Format the network summary in a card with better styling
            networkSummary.innerHTML = `
                <div class="card bg-base-100 shadow-xl border-2 border-success">
                    <div class="card-body">
                        <h3 class="card-title text-success flex items-center gap-2">
                            <i class="fas fa-check-circle"></i>
                            Network Built Successfully!
                        </h3>
                        <pre class="bg-base-200 p-4 rounded-lg overflow-x-auto text-sm font-mono leading-relaxed whitespace-pre-wrap">${data.summary}</pre>
                    </div>
                </div>
            `;
            networkSummary.classList.add('show');

            // Display classification info and auto-select loss
            displayClassificationInfo(data.classification_type, data.recommended_loss);

            showSuccess('Network built successfully! Redirecting to dataset page...');

            // Update dataset requirement display
            updateDatasetRequirement();

            // Update progress to step 1
            updateStep(1);

            // Redirect to dataset tab after 3 seconds
            setTimeout(() => {
                const datasetTab = document.getElementById('tab-dataset');
                if (datasetTab) {
                    datasetTab.checked = true;
                    // Scroll to dataset section
                    const datasetSection = document.getElementById('dataset');
                    if (datasetSection) {
                        datasetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }
            }, 3000);
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
            // Format the network summary in a card with better styling
            networkSummary.innerHTML = `
                <div class="card bg-base-100 shadow-xl border-2 border-success">
                    <div class="card-body">
                        <h3 class="card-title text-success flex items-center gap-2">
                            <i class="fas fa-check-circle"></i>
                            Network Built Successfully!
                        </h3>
                        <pre class="bg-base-200 p-4 rounded-lg overflow-x-auto text-sm font-mono leading-relaxed whitespace-pre-wrap">${data.summary}</pre>
                    </div>
                </div>
            `;
            networkSummary.classList.add('show');

            // Display classification info and auto-select loss
            displayClassificationInfo(data.classification_type, data.recommended_loss);

            // Load the example network into the interactive builder
            if (data.layers && data.connections) {
                networkBuilder.loadNetworkConfig(data.layers, data.connections);
            }

            // Update dataset requirement display
            updateDatasetRequirement();

            // Only network loaded, no dataset yet
            showSuccess('Multi-class network (3-4-2) built! Go to Dataset tab and click "Load Example Dataset" to get matching data. Redirecting...');
            updateStep(1);

            // Redirect to dataset tab after 3 seconds
            setTimeout(() => {
                const datasetTab = document.getElementById('tab-dataset');
                if (datasetTab) {
                    datasetTab.checked = true;
                    // Scroll to dataset section
                    const datasetSection = document.getElementById('dataset');
                    if (datasetSection) {
                        datasetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }
            }, 3000);
        } else {
            showError('Error: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        quickStartMultiClassBtn.disabled = false;
        quickStartMultiClassBtn.innerHTML = '<i class="fas fa-layer-group"></i> Multi-Class (3-4-2 Network)';
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
            // Format the network summary in a card with better styling
            networkSummary.innerHTML = `
                <div class="card bg-base-100 shadow-xl border-2 border-success">
                    <div class="card-body">
                        <h3 class="card-title text-success flex items-center gap-2">
                            <i class="fas fa-check-circle"></i>
                            Network Built Successfully!
                        </h3>
                        <pre class="bg-base-200 p-4 rounded-lg overflow-x-auto text-sm font-mono leading-relaxed whitespace-pre-wrap">${data.summary}</pre>
                    </div>
                </div>
            `;
            networkSummary.classList.add('show');

            // Display classification info and auto-select loss
            displayClassificationInfo(data.classification_type, data.recommended_loss);

            // Load the example network into the interactive builder
            if (data.layers && data.connections) {
                networkBuilder.loadNetworkConfig(data.layers, data.connections);
            }

            // Update dataset requirement display
            updateDatasetRequirement();

            // Only network loaded, no dataset yet
            showSuccess('Binary classification network (3-4-1) built! Go to Dataset tab and click "Load Example Dataset" to get matching data. Redirecting...');
            updateStep(1);

            // Redirect to dataset tab after 3 seconds
            setTimeout(() => {
                const datasetTab = document.getElementById('tab-dataset');
                if (datasetTab) {
                    datasetTab.checked = true;
                    // Scroll to dataset section
                    const datasetSection = document.getElementById('dataset');
                    if (datasetSection) {
                        datasetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }
            }, 3000);
        } else {
            showError('Error: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        quickStartBinaryBtn.disabled = false;
        quickStartBinaryBtn.innerHTML = '<i class="fas fa-circle-dot"></i> Binary (3-4-1 Network)';
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

// Load Example Dataset (Educational)
function loadExampleDataset() {
    // Get current network configuration
    if (!networkBuilder || !networkBuilder.layers || networkBuilder.layers.length < 2) {
        showError('Please build a network first before loading example dataset');
        return;
    }

    const inputNodes = networkBuilder.layers[0].nodes;
    const outputNodes = networkBuilder.layers[networkBuilder.layers.length - 1].nodes;

    // Generate column headers
    const inputCols = Array.from({length: inputNodes}, (_, i) => `x${i+1}`);
    const outputCols = Array.from({length: outputNodes}, (_, i) => `y${i+1}`);
    const headers = [...inputCols, ...outputCols].join(',');

    let rows = [headers];
    let datasetName = '';
    let description = '';

    // Select appropriate educational dataset based on architecture
    if (inputNodes === 2 && outputNodes === 1) {
        // XOR Problem (Classic Neural Network Problem)
        datasetName = 'XOR Logic Gate';
        description = 'Classic XOR problem - perfect for learning neural networks!';
        rows.push('0,0,0');
        rows.push('0,1,1');
        rows.push('1,0,1');
        rows.push('1,1,0');

    } else if (inputNodes === 2 && outputNodes === 2) {
        // Multi-output classification
        datasetName = 'Multi-Label Classification';
        description = 'Dataset with multiple independent outputs';
        rows.push('0,0,1,0');
        rows.push('0,1,1,1');
        rows.push('1,0,0,1');
        rows.push('1,1,0,0');
        rows.push('0.2,0.3,1,0');
        rows.push('0.8,0.9,0,1');
        rows.push('0.3,0.8,1,1');
        rows.push('0.7,0.2,0,0');

    } else if (inputNodes === 3 && outputNodes === 1) {
        // 3-input logic problem
        datasetName = '3-Input Logic Problem';
        description = 'Learn pattern recognition with 3 inputs';
        rows.push('0,0,0,0');
        rows.push('0,0,1,0');
        rows.push('0,1,0,0');
        rows.push('0,1,1,1');
        rows.push('1,0,0,0');
        rows.push('1,0,1,1');
        rows.push('1,1,0,1');
        rows.push('1,1,1,1');

    } else if (inputNodes === 3 && outputNodes === 2) {
        // Multi-class classification (3 inputs, 2 outputs for classes)
        datasetName = 'Multi-Class Classification';
        description = '3-input classification into 2 classes';
        // Class 0: [1,0]
        rows.push('1,2,3,1,0');
        rows.push('2,3,4,1,0');
        rows.push('1,1,2,1,0');
        rows.push('2,2,3,1,0');
        rows.push('1,3,2,1,0');
        // Class 1: [0,1]
        rows.push('8,9,10,0,1');
        rows.push('7,8,9,0,1');
        rows.push('9,10,11,0,1');
        rows.push('8,10,9,0,1');
        rows.push('7,9,10,0,1');

    } else if (inputNodes === 1 && outputNodes === 1) {
        // Simple linear relationship
        datasetName = 'Simple Linear Problem';
        description = 'Basic input-output relationship';
        rows.push('0,0');
        rows.push('1,1');
        rows.push('2,1');
        rows.push('3,1');
        rows.push('4,0');
        rows.push('5,0');
        rows.push('6,1');
        rows.push('7,1');

    } else {
        // Generic dataset for any configuration
        datasetName = 'Generic Pattern Dataset';
        description = `Educational dataset for ${inputNodes} inputs and ${outputNodes} outputs`;

        // Create a pattern-based dataset (not random)
        for (let i = 0; i < 12; i++) {
            const inputValues = [];
            const outputValues = [];

            // Create pattern-based inputs
            for (let j = 0; j < inputNodes; j++) {
                // Use patterns instead of random: 0, 0.5, 1 patterns
                inputValues.push((i % 3) * 0.5 + (j * 0.1));
            }

            // Create pattern-based outputs
            for (let j = 0; j < outputNodes; j++) {
                // Output pattern based on sum of inputs
                const sum = inputValues.reduce((a, b) => a + b, 0);
                outputValues.push(sum > (inputNodes * 0.5) ? 1 : 0);
            }

            rows.push([...inputValues.map(v => v.toFixed(1)), ...outputValues].join(','));
        }
    }

    const exampleData = rows.join('\n');
    datasetInput.value = exampleData;
    validateDataset();

    // Show detailed success message
    showSuccess(`📚 ${datasetName} loaded! ${description} (${rows.length - 1} samples)`);
}

// Save Dataset and Continue to next step
function saveDatasetAndContinue() {
    // Validate dataset first
    const validation = validateDataset();

    if (!validation.valid) {
        showError('Please fix dataset validation errors first or load example dataset');
        return;
    }

    // Dataset is valid, show success
    showSuccess('Dataset saved! Redirecting to Forward Pass...');

    // Update progress to step 2
    updateStep(2);

    // Redirect to Forward Pass tab after 3 seconds
    setTimeout(() => {
        const forwardTab = document.getElementById('tab-forward');
        if (forwardTab) {
            forwardTab.checked = true;
            // Scroll to forward pass section
            const forwardSection = document.getElementById('forward');
            if (forwardSection) {
                forwardSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    }, 3000);
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
            showSuccess(`Forward pass completed! ${data.num_samples} samples analyzed. Next: Calculate Loss`);

            // Update progress to step 3
            updateStep(3);

            // Switch to Loss tab after 3 seconds
            setTimeout(() => {
                const lossTab = document.getElementById('tab-loss');
                if (lossTab) {
                    lossTab.checked = true;
                }
            }, 3000);
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
            showSuccess('Loss calculated successfully! Next: Run 1 Epoch to update weights');

            // Update progress to step 4
            updateStep(4);

            // Switch to Epoch Summary tab after 3 seconds
            setTimeout(() => {
                const epochTab = document.getElementById('tab-epoch');
                if (epochTab) {
                    epochTab.checked = true;
                }
            }, 3000);
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

            // Update progress to step 5
            updateStep(5);
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

// ============================
// Run 1 Epoch (Forward + Loss + Update)
// ============================

async function run1Epoch() {
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

        const lossFunction = lossFunctionSelect.value;
        const optimizer = document.getElementById('optimizerSelect').value;
        const learningRate = parseFloat(document.getElementById('optimizerLearningRate').value);

        // Validation
        if (learningRate <= 0 || learningRate > 10) {
            showError('Learning rate must be between 0 and 10');
            return;
        }

        const run1EpochBtn = document.getElementById('run1EpochBtn');
        run1EpochBtn.disabled = true;
        run1EpochBtn.innerHTML = '<span class="loading loading-spinner loading-sm"></span> Running 1 Epoch...';

        showToast('Starting 1 Epoch: Forward Pass → Calculate Loss → Update Weights', 'info');

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
            // Display epoch summary
            displayEpochSummary(data);
            showSuccess('✅ 1 Epoch Completed! See full training cycle summary below.');

            // Update progress to step 5
            updateStep(5);

            // Scroll to epoch summary immediately (no tab switch, user already on Epoch Summary tab)
            const epochContainer = document.getElementById('epochSummaryContainer');
            if (epochContainer) {
                setTimeout(() => {
                    epochContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 500);
            }
        } else {
            showError('Error during 1 epoch: ' + data.error);
        }

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        const run1EpochBtn = document.getElementById('run1EpochBtn');
        run1EpochBtn.disabled = false;
        run1EpochBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Run 1 Epoch';
    }
}

// Display Epoch Summary (Complete Training Cycle)
function displayEpochSummary(data) {
    const { optimizer, learning_rate, loss_before, loss_after, loss_reduction, backprop_data, updated_weights } = data;
    const container = document.getElementById('epochSummaryContainer');

    if (!container) {
        console.error('epochSummaryContainer not found');
        return;
    }

    let html = '';

    // Header with completion badge
    html += '<div class="card bg-gradient-to-br from-purple-600 to-blue-500 text-white shadow-xl">';
    html += '<div class="card-body">';
    html += '<div class="flex items-center justify-between">';
    html += '<h3 class="text-2xl font-bold flex items-center gap-3">';
    html += '<i class="fas fa-check-circle text-green-300"></i>';
    html += '1 Complete Epoch Executed';
    html += '</h3>';
    html += '<div class="badge badge-success badge-lg">CYCLE COMPLETED</div>';
    html += '</div>';
    html += '<p class="opacity-90 mt-2">Full training cycle: Forward → Loss → Backprop → Update → Forward (new) → Loss (new)</p>';
    html += '</div>';
    html += '</div>';

    // Step 1: Initial Forward Pass & Loss
    html += '<div class="card bg-base-100 shadow-lg">';
    html += '<div class="card-body">';
    html += '<h4 class="card-title text-blue-600 flex items-center gap-2">';
    html += '<div class="badge badge-info">1</div>';
    html += '<i class="fas fa-arrow-right"></i> Initial Forward Pass & Loss';
    html += '</h4>';
    html += '<div class="stats shadow">';
    html += `<div class="stat">
        <div class="stat-title">Initial Loss (Before Training)</div>
        <div class="stat-value text-error">${loss_before.toFixed(6)}</div>
        <div class="stat-desc">Error before weight updates</div>
    </div>`;
    html += '</div>';
    html += '</div>';
    html += '</div>';

    // Step 2: Backpropagation
    html += '<div class="card bg-base-100 shadow-lg">';
    html += '<div class="card-body">';
    html += '<h4 class="card-title text-orange-600 flex items-center gap-2">';
    html += '<div class="badge badge-warning">2</div>';
    html += '<i class="fas fa-arrow-left"></i> Backpropagation';
    html += '</h4>';
    html += '<div class="alert alert-warning">';
    html += '<i class="fas fa-calculator"></i>';
    html += '<div>';
    html += '<p class="font-bold">Gradient Calculation (Batch Average)</p>';
    html += '<p class="text-sm">Computed gradients show how much each weight should change to reduce loss.</p>';
    html += '</div>';
    html += '</div>';

    // Show average gradients if available
    if (backprop_data && backprop_data.avg_gradients) {
        html += '<div class="overflow-x-auto">';
        html += '<table class="table table-zebra table-sm">';
        html += '<thead><tr><th>Layer</th><th>Avg Gradient Magnitude</th></tr></thead>';
        html += '<tbody>';
        backprop_data.avg_gradients.forEach((grad, idx) => {
            html += `<tr><td>Layer ${idx + 1}</td><td>${grad.toFixed(6)}</td></tr>`;
        });
        html += '</tbody></table>';
        html += '</div>';
    } else {
        html += '<p class="text-sm"><i class="fas fa-info-circle mr-2"></i>Gradients calculated for all weights based on loss.</p>';
    }
    html += '</div>';
    html += '</div>';

    // Step 3: Weight Update
    html += '<div class="card bg-base-100 shadow-lg">';
    html += '<div class="card-body">';
    html += '<h4 class="card-title text-purple-600 flex items-center gap-2">';
    html += '<div class="badge badge-secondary">3</div>';
    html += '<i class="fas fa-sync-alt"></i> Update Weights';
    html += '</h4>';
    html += '<div class="stats stats-vertical lg:stats-horizontal shadow w-full">';
    html += `<div class="stat">
        <div class="stat-figure text-secondary"><i class="fas fa-brain text-2xl"></i></div>
        <div class="stat-title">Optimizer</div>
        <div class="stat-value text-sm">${optimizer.toUpperCase()}</div>
    </div>`;
    html += `<div class="stat">
        <div class="stat-figure text-accent"><i class="fas fa-tachometer-alt text-2xl"></i></div>
        <div class="stat-title">Learning Rate</div>
        <div class="stat-value text-sm">${learning_rate}</div>
    </div>`;
    html += `<div class="stat">
        <div class="stat-figure text-info"><i class="fas fa-weight text-2xl"></i></div>
        <div class="stat-title">Weights Updated</div>
        <div class="stat-value text-sm">${updated_weights || 'All'}</div>
    </div>`;
    html += '</div>';
    html += '<p class="text-sm mt-2"><i class="fas fa-formula mr-2"></i><strong>Formula:</strong> new_weight = old_weight - (learning_rate × gradient)</p>';

    // Add collapsible weight details
    if (data.weight_changes && data.new_weights && data.old_weights) {
        html += '<div class="collapse collapse-arrow bg-base-200 mt-4">';
        html += '<input type="checkbox" />';
        html += '<div class="collapse-title text-base font-medium">';
        html += '<i class="fas fa-table mr-2"></i>View Detailed Weight Updates';
        html += '</div>';
        html += '<div class="collapse-content">';

        // Weight changes per layer
        html += '<div class="overflow-x-auto mt-2">';

        for (const [layerKey, layerWeightChanges] of Object.entries(data.weight_changes)) {
            const layerIdx = layerKey.split('_')[1];
            const layerBiasChanges = data.bias_changes ? data.bias_changes[layerKey] : [];

            html += `<div class="mb-6">`;
            html += `<h5 class="font-bold text-purple-600 mb-3 flex items-center gap-2">`;
            html += `<i class="fas fa-layer-group"></i> Layer ${layerIdx}`;
            html += `</h5>`;
            html += '<table class="table table-zebra table-sm">';
            html += '<thead><tr>';
            html += '<th>Node</th>';
            html += '<th>Old Weights</th>';
            html += '<th>New Weights</th>';
            html += '<th>ΔW (Change)</th>';
            if (layerBiasChanges && layerBiasChanges.length > 0) {
                html += '<th>Bias Change</th>';
            }
            html += '</tr></thead>';
            html += '<tbody>';

            layerWeightChanges.forEach((nodeChanges, nodeIdx) => {
                html += '<tr>';
                html += `<td class="font-bold">Node ${nodeIdx}</td>`;

                // Old weights
                html += '<td><span class="font-mono text-xs">[';
                html += data.old_weights[layerKey][nodeIdx].map(w => w.toFixed(4)).join(', ');
                html += ']</span></td>';

                // New weights
                html += '<td><span class="font-mono text-xs">[';
                html += data.new_weights[layerKey][nodeIdx].map(w => w.toFixed(4)).join(', ');
                html += ']</span></td>';

                // Weight changes
                html += '<td><span class="font-mono text-xs">[';
                html += nodeChanges.map(c => {
                    const sign = c >= 0 ? '+' : '';
                    const colorClass = c >= 0 ? 'text-success' : 'text-error';
                    return `<span class="${colorClass} font-semibold">${sign}${c.toFixed(4)}</span>`;
                }).join(', ');
                html += ']</span></td>';

                // Bias change (if available)
                if (layerBiasChanges && layerBiasChanges.length > nodeIdx) {
                    const biasChange = layerBiasChanges[nodeIdx];
                    const biasSign = biasChange >= 0 ? '+' : '';
                    const biasColorClass = biasChange >= 0 ? 'text-success' : 'text-error';
                    html += `<td><span class="font-mono text-xs ${biasColorClass} font-semibold">${biasSign}${biasChange.toFixed(4)}</span></td>`;
                }

                html += '</tr>';
            });

            html += '</tbody></table>';
            html += '</div>';
        }

        html += '</div>';
        html += '</div>';
        html += '</div>';
    }

    html += '</div>';
    html += '</div>';

    // Step 4: New Forward Pass & Loss
    html += '<div class="card bg-base-100 shadow-lg">';
    html += '<div class="card-body">';
    html += '<h4 class="card-title text-green-600 flex items-center gap-2">';
    html += '<div class="badge badge-success">4</div>';
    html += '<i class="fas fa-arrow-right"></i> Forward Pass (with Updated Weights)';
    html += '</h4>';
    html += '<div class="stats shadow">';
    html += `<div class="stat">
        <div class="stat-title">New Loss (After Training)</div>
        <div class="stat-value text-success">${loss_after.toFixed(6)}</div>
        <div class="stat-desc">Error after weight updates</div>
    </div>`;
    html += '</div>';
    html += '</div>';
    html += '</div>';

    // Comparison & Improvement
    html += '<div class="card bg-gradient-to-r from-green-500 to-teal-500 text-white shadow-xl">';
    html += '<div class="card-body">';
    html += '<h4 class="card-title text-2xl">';
    html += '<i class="fas fa-chart-line mr-2"></i> Training Impact';
    html += '</h4>';
    html += '<div class="grid grid-cols-1 md:grid-cols-3 gap-4">';
    html += `<div class="bg-white/20 rounded-lg p-4 backdrop-blur">
        <p class="text-sm opacity-90">Loss Before</p>
        <p class="text-3xl font-bold">${loss_before.toFixed(6)}</p>
    </div>`;
    html += `<div class="bg-white/20 rounded-lg p-4 backdrop-blur">
        <p class="text-sm opacity-90">Loss After</p>
        <p class="text-3xl font-bold">${loss_after.toFixed(6)}</p>
    </div>`;
    html += `<div class="bg-white/20 rounded-lg p-4 backdrop-blur">
        <p class="text-sm opacity-90">Improvement</p>
        <p class="text-3xl font-bold">↓ ${((loss_reduction / loss_before) * 100).toFixed(2)}%</p>
        <p class="text-xs opacity-75">${loss_reduction.toFixed(6)} reduction</p>
    </div>`;
    html += '</div>';
    html += '</div>';
    html += '</div>';

    // Next steps
    html += '<div class="alert alert-info shadow-lg">';
    html += '<i class="fas fa-lightbulb"></i>';
    html += '<div>';
    html += '<p class="font-bold">What You Learned:</p>';
    html += '<ul class="text-sm mt-2 space-y-1 list-disc list-inside">';
    html += '<li>How forward pass generates predictions</li>';
    html += '<li>How loss measures prediction error</li>';
    html += '<li>How backpropagation calculates gradients</li>';
    html += '<li>How optimizer updates weights to improve performance</li>';
    html += '</ul>';
    html += '<p class="font-bold mt-3">Next Steps:</p>';
    html += '<ul class="text-sm mt-1 space-y-1 list-disc list-inside">';
    html += '<li>Click "Run 1 Complete Epoch" again to see more improvement</li>';
    html += '<li>Or go to "Automate Training" tab to run many epochs at once</li>';
    html += '</ul>';
    html += '</div>';
    html += '</div>';

    container.innerHTML = html;
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
    // Use the toast notification system from scripts.html
    if (typeof showToast === 'function') {
        showToast(message, 'error');
    } else {
        console.error(message);
    }
}

function showSuccess(message) {
    // Use the toast notification system from scripts.html
    if (typeof showToast === 'function') {
        showToast(message, 'success');
    } else {
        console.log(message);
    }
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

            // Update progress to step 6
            updateStep(6);

            // Automatically switch to Results tab after 3 seconds
            setTimeout(() => {
                const resultsTab = document.getElementById('tab-results');
                if (resultsTab) {
                    resultsTab.checked = true;
                    updateStep(7);

                    // Update charts after tab is visible
                    setTimeout(() => {
                        if (data.history && typeof updateCharts === 'function') {
                            console.log('Updating charts with training history');
                            updateCharts(data.history);
                        }
                    }, 300);
                }
            }, 3000);
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
// NOTE: NetworkDiagram class has been removed to avoid conflict with InteractiveNetworkBuilder
// InteractiveNetworkBuilder in network-builder.js handles all network visualization

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    console.log('ANN from Scratch - Ready!');
});
