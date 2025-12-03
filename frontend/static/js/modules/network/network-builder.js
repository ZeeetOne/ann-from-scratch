/**
 * Interactive Network Builder
 * Allows users to visually construct neural networks by:
 * - Adding/removing nodes from each layer
 * - Dragging to create connections between nodes
 * - Editing connection weights and biases
 */

export class InteractiveNetworkBuilder {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.layers = [
            { type: 'input', nodes: 1, activation: 'linear' },
            { type: 'hidden', nodes: 1, activation: 'sigmoid' },
            { type: 'output', nodes: 1, activation: 'sigmoid' }
        ];
        this.hiddenLayerCount = 1;

        this.nodes = [];
        this.connections = [];

        this.nodeRadius = 25;
        this.layerSpacing = 200;
        this.nodeSpacing = 80;

        this.dragStartNode = null;
        this.tempLine = null;
        this.isDragging = false;

        // Track current editing connection
        this.currentEditingConnection = null;

        // Prevent render loops
        this.isRendering = false;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.render();
    }

    setupEventListeners() {
        // Input layer controls
        document.getElementById('inputLayerPlus').addEventListener('click', () => {
            this.changeLayerSize(0, 1);
        });
        document.getElementById('inputLayerMinus').addEventListener('click', () => {
            this.changeLayerSize(0, -1);
        });

        // Output layer controls
        document.getElementById('outputLayerPlus').addEventListener('click', () => {
            const lastIndex = this.layers.length - 1;
            this.changeLayerSize(lastIndex, 1);
        });
        document.getElementById('outputLayerMinus').addEventListener('click', () => {
            const lastIndex = this.layers.length - 1;
            this.changeLayerSize(lastIndex, -1);
        });

        // Hidden layer controls (delegated events)
        document.getElementById('hiddenLayersControls').addEventListener('click', (e) => {
            // Use closest() to handle clicks on button or icon inside button
            const plusBtn = e.target.closest('.hidden-plus');
            const minusBtn = e.target.closest('.hidden-minus');
            const removeBtn = e.target.closest('.btn-remove-layer');

            if (plusBtn) {
                const layerIndex = parseInt(plusBtn.dataset.layer);
                console.log('Plus button clicked for layer:', layerIndex);
                this.changeHiddenLayerSize(layerIndex, 1);
            } else if (minusBtn) {
                const layerIndex = parseInt(minusBtn.dataset.layer);
                console.log('Minus button clicked for layer:', layerIndex);
                this.changeHiddenLayerSize(layerIndex, -1);
            } else if (removeBtn) {
                const layerIndex = parseInt(removeBtn.dataset.layer);
                console.log('Remove button clicked for layer:', layerIndex);
                this.removeHiddenLayer(layerIndex);
            }
        });

        // Add hidden layer button
        document.getElementById('addHiddenLayerBtn').addEventListener('click', () => {
            this.addHiddenLayer();
        });

        // Reset button
        document.getElementById('resetNetworkBtn').addEventListener('click', () => {
            this.reset();
        });

        // Input and Output activation changes
        document.getElementById('inputActivation').addEventListener('change', (e) => {
            this.layers[0].activation = e.target.value;
            console.log('Input activation changed to:', e.target.value);
        });

        document.getElementById('outputActivation').addEventListener('change', (e) => {
            const lastIndex = this.layers.length - 1;
            this.layers[lastIndex].activation = e.target.value;
            console.log('Output activation changed to:', e.target.value);
        });

        // Hidden layer activation changes (event delegation)
        document.getElementById('hiddenLayersControls').addEventListener('change', (e) => {
            if (e.target.classList.contains('hidden-activation')) {
                const layerIndex = parseInt(e.target.dataset.layer);
                if (layerIndex >= 0 && layerIndex < this.layers.length) {
                    this.layers[layerIndex].activation = e.target.value;
                    console.log(`Hidden layer ${layerIndex} activation changed to: ${e.target.value}`);
                }
            }
        });

        // Canvas mouse events for drag-to-connect
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    changeLayerSize(layerIndex, delta) {
        const layer = this.layers[layerIndex];
        const newSize = layer.nodes + delta;

        if (newSize < 1 || newSize > 10) return;

        layer.nodes = newSize;

        // Update UI counter
        if (layerIndex === 0) {
            document.getElementById('inputLayerCount').textContent = newSize;
        } else if (layerIndex === this.layers.length - 1) {
            document.getElementById('outputLayerCount').textContent = newSize;
        }

        this.render();
    }

    changeHiddenLayerSize(layerIndex, delta) {
        const actualIndex = layerIndex; // layerIndex is actual index in this.layers
        const layer = this.layers[actualIndex];

        if (!layer || layer.type !== 'hidden') return;

        const newSize = layer.nodes + delta;
        if (newSize < 1 || newSize > 10) return;

        layer.nodes = newSize;

        // Update UI counter
        const countEl = document.querySelector(`.hidden-count[data-layer="${layerIndex}"]`);
        if (countEl) {
            countEl.textContent = newSize;
        }

        this.render();
    }

    addHiddenLayer() {
        if (this.hiddenLayerCount >= 5) {
            alert('Maximum 5 hidden layers allowed');
            return;
        }

        this.hiddenLayerCount++;

        // Insert before output layer
        const newLayer = {
            type: 'hidden',
            nodes: 1,
            activation: 'sigmoid'
        };

        this.layers.splice(this.layers.length - 1, 0, newLayer);

        // Update UI
        this.updateHiddenLayersUI();
        this.render();
    }

    removeHiddenLayer(layerIndex) {
        if (this.hiddenLayerCount <= 1) {
            alert('At least 1 hidden layer is required');
            return;
        }

        // Remove from layers array
        this.layers.splice(layerIndex, 1);
        this.hiddenLayerCount--;

        // Remove connections involving this layer
        this.connections = this.connections.filter(conn => {
            const fromLayer = this.getNodeLayer(conn.from);
            const toLayer = this.getNodeLayer(conn.to);
            return fromLayer !== layerIndex && toLayer !== layerIndex;
        });

        this.updateHiddenLayersUI();
        this.render();
    }

    updateHiddenLayersUI() {
        const container = document.getElementById('hiddenLayersControls');
        container.innerHTML = '';

        // Get hidden layers (skip input and output)
        this.layers.forEach((layer, index) => {
            if (layer.type !== 'hidden') return;

            const div = document.createElement('div');
            div.dataset.layerIndex = index;

            const hiddenIndex = this.layers.slice(0, index).filter(l => l.type === 'hidden').length + 1;

            // Use Tailwind/DaisyUI classes matching the new design (no background, proper alignment, square remove button)
            div.className = 'hidden-layer-item';
            div.innerHTML = `
                <div class="flex items-center justify-between mb-3 px-1">
                    <span class="text-sm font-bold text-purple-700 badge badge-outline badge-sm">Layer ${hiddenIndex}</span>
                    <button class="btn btn-xs btn-error btn-remove-layer hover:scale-110 transition-transform flex-shrink-0 h-6 w-6 min-h-6 p-0 border-0 rounded" data-layer="${index}" title="Remove this layer">
                        <i class="fas fa-trash-alt text-[0.65rem]"></i>
                    </button>
                </div>
                <div class="flex items-center justify-center gap-2 mb-3">
                    <button class="btn btn-circle btn-sm btn-primary shadow-md hover:shadow-lg hover:scale-110 transition-transform hidden-minus h-8 w-8 min-h-8 p-0 border-0" data-layer="${index}">
                        <i class="fas fa-minus"></i>
                    </button>
                    <div class="text-2xl font-bold text-purple-700 h-8 min-w-8 flex items-center justify-center leading-none">
                        <span class="hidden-count" data-layer="${index}">${layer.nodes}</span>
                    </div>
                    <button class="btn btn-circle btn-sm btn-primary shadow-md hover:shadow-lg hover:scale-110 transition-transform hidden-plus h-8 w-8 min-h-8 p-0 border-0" data-layer="${index}">
                        <i class="fas fa-plus"></i>
                    </button>
                </div>
                <select class="select select-bordered select-sm w-full hidden-activation bg-white" data-layer="${index}">
                    <option value="sigmoid" ${layer.activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                    <option value="relu" ${layer.activation === 'relu' ? 'selected' : ''}>ReLU</option>
                    <option value="threshold" ${layer.activation === 'threshold' ? 'selected' : ''}>Threshold</option>
                    <option value="softmax" ${layer.activation === 'softmax' ? 'selected' : ''}>Softmax</option>
                </select>
            `;

            container.appendChild(div);
        });

        // Add event listener for activation changes
        container.querySelectorAll('.hidden-activation').forEach(select => {
            select.addEventListener('change', (e) => {
                const layerIndex = parseInt(e.target.dataset.layer);
                this.layers[layerIndex].activation = e.target.value;
            });
        });
    }

    reset() {
        this.layers = [
            { type: 'input', nodes: 1, activation: 'linear' },
            { type: 'hidden', nodes: 1, activation: 'sigmoid' },
            { type: 'output', nodes: 1, activation: 'sigmoid' }
        ];
        this.hiddenLayerCount = 1;
        this.connections = [];

        // Reset UI counters
        document.getElementById('inputLayerCount').textContent = '1';
        document.getElementById('outputLayerCount').textContent = '1';

        this.updateHiddenLayersUI();
        this.render();
    }

    render() {
        // Prevent re-entry (avoid render loops)
        if (this.isRendering) {
            console.warn('Render already in progress, skipping');
            return;
        }

        this.isRendering = true;
        console.log('Starting render...');

        try {
            // Save existing node biases before re-render
            const savedBiases = {};
            if (this.nodes && this.nodes.length > 0) {
                this.nodes.forEach(node => {
                    savedBiases[node.id] = node.bias;
                });
                console.log('Saved biases before render:', savedBiases);
            }

            // Clear canvas
            this.canvas.innerHTML = '';

            const canvasWidth = this.canvas.clientWidth;
            const maxNodesInLayer = Math.max(...this.layers.map(l => l.nodes));
            const canvasHeight = Math.max(450, maxNodesInLayer * this.nodeSpacing + 100);
            this.canvas.style.height = canvasHeight + 'px';

            // Calculate positions
            const totalWidth = (this.layers.length - 1) * this.layerSpacing;
            const startX = (canvasWidth - totalWidth) / 2;

            // Create nodes
            this.nodes = [];
            this.layers.forEach((layer, layerIdx) => {
                const numNodes = layer.nodes;
                const layerHeight = (numNodes - 1) * this.nodeSpacing;
                const startY = (canvasHeight - layerHeight) / 2;

                for (let nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
                    const nodeId = `L${layerIdx}N${nodeIdx}`;
                    const node = {
                        id: nodeId,
                        layer: layerIdx,
                        index: nodeIdx,
                        x: startX + layerIdx * this.layerSpacing,
                        y: startY + nodeIdx * this.nodeSpacing,
                        type: layer.type,
                        activation: layer.activation,
                        bias: savedBiases[nodeId] !== undefined ? savedBiases[nodeId] : 0  // Restore saved bias or default to 0
                    };

                    this.nodes.push(node);
                    this.renderNode(node);
                }
            });

            // Render connections
            this.renderConnections();

            console.log('Render completed');
        } finally {
            this.isRendering = false;
        }
    }

    renderNode(node) {
        const nodeEl = document.createElement('div');
        nodeEl.className = `network-node ${node.type}-layer`;
        nodeEl.textContent = node.index;
        nodeEl.style.left = (node.x - this.nodeRadius) + 'px';
        nodeEl.style.top = (node.y - this.nodeRadius) + 'px';
        nodeEl.dataset.nodeId = node.id;

        // Add label
        const label = document.createElement('div');
        label.className = 'node-label';
        label.textContent = node.id;
        nodeEl.appendChild(label);

        this.canvas.appendChild(nodeEl);
    }

    renderConnections() {
        this.connections.forEach((conn, idx) => {
            const fromNode = this.nodes.find(n => n.id === conn.from);
            const toNode = this.nodes.find(n => n.id === conn.to);

            if (!fromNode || !toNode) return;

            this.renderConnection(fromNode, toNode, conn, idx);
        });
    }

    renderConnection(fromNode, toNode, conn, idx) {
        // Create SVG line - use full canvas size to avoid viewBox issues
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('connection-line');
        svg.dataset.connIndex = idx;

        // Make SVG cover entire canvas
        svg.style.position = 'absolute';
        svg.style.left = '0';
        svg.style.top = '0';
        svg.style.width = '100%';
        svg.style.height = '100%';
        svg.style.pointerEvents = 'none';

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', fromNode.x);
        line.setAttribute('y1', fromNode.y);
        line.setAttribute('x2', toNode.x);
        line.setAttribute('y2', toNode.y);
        line.style.pointerEvents = 'stroke';
        line.style.strokeWidth = '2';
        line.style.cursor = 'pointer';

        // IMPORTANT: Use event delegation with once: true to prevent multiple handlers
        // Store the handler so we can properly clean up
        const clickHandler = (e) => {
            console.log('Connection line clicked, index:', idx);
            e.stopPropagation();
            e.preventDefault();
            this.editConnection(idx);
        };

        const contextMenuHandler = (e) => {
            console.log('Connection right-clicked, index:', idx);
            e.preventDefault();
            e.stopPropagation();
            this.deleteConnection(idx);
        };

        // Add event listeners
        line.addEventListener('click', clickHandler);
        line.addEventListener('contextmenu', contextMenuHandler);

        svg.appendChild(line);
        this.canvas.insertBefore(svg, this.canvas.firstChild);

        // Add weight label
        const weightLabel = document.createElement('div');
        weightLabel.className = 'connection-weight';
        weightLabel.textContent = `w:${conn.weight.toFixed(2)}`;
        weightLabel.style.left = ((fromNode.x + toNode.x) / 2 - 25) + 'px';
        weightLabel.style.top = ((fromNode.y + toNode.y) / 2 - 10) + 'px';
        this.canvas.appendChild(weightLabel);
    }

    handleMouseDown(e) {
        const nodeEl = e.target.closest('.network-node');
        if (!nodeEl) return;

        const nodeId = nodeEl.dataset.nodeId;
        const node = this.nodes.find(n => n.id === nodeId);

        if (!node) return;

        this.dragStartNode = node;
        this.isDragging = true;
        nodeEl.classList.add('dragging-source');
    }

    handleMouseMove(e) {
        if (!this.isDragging || !this.dragStartNode) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Remove old temp line
        if (this.tempLine) {
            this.tempLine.remove();
        }

        // Draw temp line
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.classList.add('temp-line');
        svg.style.position = 'absolute';
        svg.style.left = '0';
        svg.style.top = '0';
        svg.style.width = '100%';
        svg.style.height = '100%';
        svg.style.pointerEvents = 'none';

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', this.dragStartNode.x);
        line.setAttribute('y1', this.dragStartNode.y);
        line.setAttribute('x2', x);
        line.setAttribute('y2', y);

        svg.appendChild(line);
        this.canvas.appendChild(svg);
        this.tempLine = svg;
    }

    handleMouseUp(e) {
        if (!this.isDragging || !this.dragStartNode) {
            this.isDragging = false;
            this.dragStartNode = null;
            if (this.tempLine) {
                this.tempLine.remove();
                this.tempLine = null;
            }
            return;
        }

        const nodeEl = e.target.closest('.network-node');

        // Remove dragging class
        const sourceEl = this.canvas.querySelector('.dragging-source');
        if (sourceEl) {
            sourceEl.classList.remove('dragging-source');
        }

        if (nodeEl) {
            const targetId = nodeEl.dataset.nodeId;
            const targetNode = this.nodes.find(n => n.id === targetId);

            if (targetNode && targetNode.id !== this.dragStartNode.id) {
                this.createConnection(this.dragStartNode, targetNode);
            }
        }

        // Cleanup
        this.isDragging = false;
        this.dragStartNode = null;
        if (this.tempLine) {
            this.tempLine.remove();
            this.tempLine = null;
        }
    }

    createConnection(fromNode, toNode) {
        // Check if connection already exists
        const exists = this.connections.some(conn =>
            conn.from === fromNode.id && conn.to === toNode.id
        );

        if (exists) {
            alert('Connection already exists!');
            return;
        }

        // Ensure connections only go forward (left to right)
        if (fromNode.layer >= toNode.layer) {
            alert('Connections must go from left to right (forward direction only)!');
            return;
        }

        // Add connection with random initial weight
        // Note: bias is stored in target node, but we keep a reference here for backward compatibility
        const conn = {
            from: fromNode.id,
            to: toNode.id,
            weight: (Math.random() * 2 - 1),
            bias: toNode.bias  // Get bias from target node
        };

        this.connections.push(conn);
        this.render();
    }

    editConnection(connIndex) {
        console.log('editConnection called with index:', connIndex);

        const conn = this.connections[connIndex];
        if (!conn) {
            console.error('Connection not found:', connIndex);
            return;
        }

        const fromNode = this.nodes.find(n => n.id === conn.from);
        const toNode = this.nodes.find(n => n.id === conn.to);

        if (!fromNode || !toNode) {
            console.error('Nodes not found for connection:', conn);
            return;
        }

        console.log('Opening modal for connection:', fromNode.id, '→', toNode.id);

        // Show modal
        const modal = document.getElementById('connectionModal');
        const connectionInfo = document.getElementById('connectionInfo');
        const weightInput = document.getElementById('modalWeight');
        const biasInput = document.getElementById('modalBias');

        if (!modal || !connectionInfo || !weightInput || !biasInput) {
            console.error('Modal elements not found');
            return;
        }

        console.log('=== OPENING CONNECTION MODAL ===');
        console.log('Connection:', conn.from, '→', conn.to);
        console.log('Connection weight:', conn.weight);
        console.log('Connection bias field:', conn.bias);
        console.log('Target Node:', toNode.id);
        console.log('Target Node bias property:', toNode.bias);
        console.log('All nodes:', this.nodes.map(n => `${n.id}:${n.bias}`).join(', '));

        connectionInfo.innerHTML = `<strong>Connection:</strong> ${fromNode.id} → <strong class="text-purple-600">${toNode.id}</strong> (target node)`;
        weightInput.value = conn.weight.toFixed(3);

        // Get bias from TARGET NODE (not from connection object)
        const targetBias = toNode.bias !== undefined ? toNode.bias : 0;
        biasInput.value = targetBias.toFixed(3);

        console.log('Displaying bias in modal:', targetBias);
        console.log('=== MODAL OPENED ===');

        // Update bias hint to show target node ID
        const biasHintText = document.getElementById('biasHintText');
        if (biasHintText) {
            biasHintText.textContent = `Bias of target node "${toNode.id}" (shared by all its incoming connections)`;
        }

        console.log('About to show modal...');

        // Store current connection index for save handler
        this.currentEditingConnection = connIndex;

        // Show modal (custom implementation with class toggle)
        try {
            modal.classList.remove('hidden');
            console.log('Modal shown successfully');

            // Focus on weight input
            setTimeout(() => {
                weightInput.focus();
                weightInput.select();
            }, 100);
        } catch (error) {
            console.error('Error showing modal:', error);
        }
    }

    saveConnectionEdit() {
        console.log('saveConnectionEdit called');

        if (this.currentEditingConnection === null || this.currentEditingConnection === undefined) {
            console.error('No connection being edited');
            return;
        }

        const conn = this.connections[this.currentEditingConnection];
        if (!conn) {
            console.error('Connection not found for editing');
            return;
        }

        // Find the target node
        const toNode = this.nodes.find(n => n.id === conn.to);
        if (!toNode) {
            console.error('Target node not found');
            return;
        }

        const weightInput = document.getElementById('modalWeight');
        const biasInput = document.getElementById('modalBias');
        const modal = document.getElementById('connectionModal');

        const newWeight = parseFloat(weightInput.value) || 0;
        const newBias = parseFloat(biasInput.value) || 0;

        console.log('=== SAVING CONNECTION EDIT ===');
        console.log('Connection:', conn.from, '→', conn.to);
        console.log('New Weight:', newWeight);
        console.log('New Bias:', newBias);
        console.log('Target Node BEFORE update:', toNode.id, 'bias =', toNode.bias);

        // Update connection weight
        conn.weight = newWeight;

        // Update bias in TARGET NODE (this is the correct place for bias)
        toNode.bias = newBias;

        console.log('Target Node AFTER update:', toNode.id, 'bias =', toNode.bias);

        // Sync bias to ALL connections targeting this node (for backward compatibility)
        let syncedCount = 0;
        this.connections.forEach(c => {
            if (c.to === toNode.id) {
                c.bias = newBias;
                syncedCount++;
                console.log('  Synced bias to connection:', c.from, '→', c.to);
            }
        });

        console.log('Total connections synced:', syncedCount);
        console.log('All nodes biases:', this.nodes.map(n => `${n.id}:${n.bias}`).join(', '));
        console.log('=== SAVE COMPLETE ===');

        modal.classList.add('hidden');

        // Only update the weight label, no need for full render
        const weightLabels = this.canvas.querySelectorAll('.connection-weight');
        if (weightLabels[this.currentEditingConnection]) {
            weightLabels[this.currentEditingConnection].textContent = `w:${conn.weight.toFixed(2)}`;
        }

        this.currentEditingConnection = null;
        console.log('saveConnectionEdit completed');
    }

    cancelConnectionEdit() {
        console.log('cancelConnectionEdit called');
        const modal = document.getElementById('connectionModal');
        modal.classList.add('hidden');
        this.currentEditingConnection = null;
    }

    deleteConnection(connIndex) {
        if (confirm('Delete this connection?')) {
            this.connections.splice(connIndex, 1);
            this.render();
        }
    }

    getNodeLayer(nodeId) {
        const node = this.nodes.find(n => n.id === nodeId);
        return node ? node.layer : -1;
    }

    getNetworkConfig() {
        console.log('=== GETTING NETWORK CONFIG FOR BACKEND ===');

        // Convert to format needed by backend
        const layers = this.layers.map(layer => ({
            num_nodes: layer.nodes,
            activation: layer.activation
        }));

        console.log('Layers:', layers);
        console.log('All connections:', this.connections.map(c => `${c.from}→${c.to} w=${c.weight.toFixed(3)} b=${c.bias.toFixed(3)}`));
        console.log('All nodes with bias:', this.nodes.map(n => `${n.id} bias=${n.bias.toFixed(3)}`));

        // Organize connections by target layer
        const connectionsByLayer = {};

        this.layers.forEach((layer, layerIdx) => {
            if (layerIdx === 0) return; // Skip input layer

            connectionsByLayer[layerIdx] = [];

            for (let nodeIdx = 0; nodeIdx < layer.nodes; nodeIdx++) {
                const targetId = `L${layerIdx}N${nodeIdx}`;
                const nodeConns = this.connections.filter(c => c.to === targetId);

                const connections = [];
                const weights = [];
                let bias = 0;

                nodeConns.forEach(conn => {
                    const fromNode = this.nodes.find(n => n.id === conn.from);
                    if (fromNode) {
                        connections.push(fromNode.index);
                        weights.push(conn.weight);
                        bias = conn.bias; // Take the first bias (should be same for all connections to this node)
                    }
                });

                connectionsByLayer[layerIdx].push({
                    connections,
                    weights,
                    bias
                });
            }
        });

        // Convert to backend format
        const connectionsArray = [];
        Object.keys(connectionsByLayer).forEach(layerIdx => {
            const layerConns = connectionsByLayer[layerIdx];

            const layerData = {
                layer_idx: parseInt(layerIdx),
                connections: layerConns.map(nc => nc.connections),
                weights: layerConns.map(nc => nc.weights),
                biases: layerConns.map(nc => nc.bias)
            };

            console.log(`Layer ${layerIdx}:`);
            console.log('  Connections:', layerData.connections);
            console.log('  Weights:', layerData.weights);
            console.log('  Biases:', layerData.biases);

            connectionsArray.push(layerData);
        });

        const config = {
            layers,
            connections: connectionsArray
        };

        console.log('Final config being sent to backend:', JSON.stringify(config, null, 2));
        console.log('=== END NETWORK CONFIG ===');

        return config;
    }

    loadNetworkConfig(layers, connections) {
        // Load layers
        this.layers = layers.map((layer, idx) => ({
            type: idx === 0 ? 'input' : (idx === layers.length - 1 ? 'output' : 'hidden'),
            nodes: layer.num_nodes,
            activation: layer.activation
        }));

        this.hiddenLayerCount = this.layers.filter(l => l.type === 'hidden').length;

        // Update UI
        document.getElementById('inputLayerCount').textContent = this.layers[0].nodes;
        document.getElementById('outputLayerCount').textContent = this.layers[this.layers.length - 1].nodes;

        // Update activation dropdowns
        const inputActivation = document.getElementById('inputActivation');
        const outputActivation = document.getElementById('outputActivation');
        if (inputActivation) {
            inputActivation.value = this.layers[0].activation;
        }
        if (outputActivation) {
            outputActivation.value = this.layers[this.layers.length - 1].activation;
        }

        this.updateHiddenLayersUI();

        // Load connections
        this.connections = [];
        connections.forEach(layerConn => {
            const layerIdx = layerConn.layer_idx;

            layerConn.connections.forEach((nodeConns, nodeIdx) => {
                nodeConns.forEach((fromIdx, connIdx) => {
                    const fromId = `L${layerIdx - 1}N${fromIdx}`;
                    const toId = `L${layerIdx}N${nodeIdx}`;

                    this.connections.push({
                        from: fromId,
                        to: toId,
                        weight: layerConn.weights[nodeIdx][connIdx],
                        bias: layerConn.biases[nodeIdx]
                    });
                });
            });
        });

        this.render();
    }
}

// Initialize on page load
let networkBuilder;
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing InteractiveNetworkBuilder...');
    networkBuilder = new InteractiveNetworkBuilder('networkCanvas');

    // Setup modal button handlers (only once)
    const modal = document.getElementById('connectionModal');
    const saveBtn = document.getElementById('saveConnectionBtn');
    const deleteBtn = document.getElementById('deleteConnectionBtn');
    const cancelBtn = document.getElementById('cancelConnectionBtn');
    const closeBtn = document.getElementById('closeModalBtn');

    if (saveBtn) {
        saveBtn.addEventListener('click', (e) => {
            console.log('Save button clicked');
            e.preventDefault();
            e.stopPropagation();
            if (networkBuilder) {
                networkBuilder.saveConnectionEdit();
            }
        });
    }

    if (deleteBtn) {
        deleteBtn.addEventListener('click', (e) => {
            console.log('Delete button clicked');
            e.preventDefault();
            e.stopPropagation();
            if (networkBuilder && networkBuilder.currentEditingConnection !== null) {
                networkBuilder.deleteConnection(networkBuilder.currentEditingConnection);
                modal.classList.add('hidden');
                networkBuilder.currentEditingConnection = null;
            }
        });
    }

    if (cancelBtn) {
        cancelBtn.addEventListener('click', (e) => {
            console.log('Cancel button clicked');
            e.preventDefault();
            e.stopPropagation();
            if (networkBuilder) {
                networkBuilder.cancelConnectionEdit();
            }
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', (e) => {
            console.log('Close (X) button clicked');
            e.preventDefault();
            e.stopPropagation();
            if (networkBuilder) {
                networkBuilder.cancelConnectionEdit();
            }
        });
    }

    // Handle backdrop click to close modal
    if (modal) {
        const backdrop = modal.querySelector('.custom-modal-backdrop');
        if (backdrop) {
            backdrop.addEventListener('click', (e) => {
                console.log('Backdrop clicked, closing modal');
                e.stopPropagation();
                if (networkBuilder) {
                    networkBuilder.cancelConnectionEdit();
                }
            });
        }
    }

    console.log('InteractiveNetworkBuilder initialized successfully');
});
