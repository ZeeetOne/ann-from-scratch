/**
 * Interactive Network Builder
 * Allows users to visually construct neural networks by:
 * - Adding/removing nodes from each layer
 * - Dragging to create connections between nodes
 * - Editing connection weights and biases
 */

class InteractiveNetworkBuilder {
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
            if (e.target.classList.contains('hidden-plus')) {
                const layerIndex = parseInt(e.target.dataset.layer);
                this.changeHiddenLayerSize(layerIndex, 1);
            } else if (e.target.classList.contains('hidden-minus')) {
                const layerIndex = parseInt(e.target.dataset.layer);
                this.changeHiddenLayerSize(layerIndex, -1);
            } else if (e.target.classList.contains('btn-remove-layer')) {
                const layerIndex = parseInt(e.target.dataset.layer);
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
        });

        document.getElementById('outputActivation').addEventListener('change', (e) => {
            const lastIndex = this.layers.length - 1;
            this.layers[lastIndex].activation = e.target.value;
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
            div.className = 'hidden-layer-item space-y-3 mb-3';
            div.dataset.layerIndex = index;

            const hiddenIndex = this.layers.slice(0, index).filter(l => l.type === 'hidden').length + 1;

            // Use Tailwind/DaisyUI classes for new structure
            div.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <span class="text-sm font-semibold">Layer ${hiddenIndex}:</span>
                    <button class="btn btn-circle btn-xs btn-error btn-remove-layer" data-layer="${index}">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="flex items-center justify-center gap-2 mb-2">
                    <button class="btn btn-circle btn-sm btn-outline hidden-minus" data-layer="${index}">
                        <i class="fas fa-minus"></i>
                    </button>
                    <div class="badge badge-lg badge-secondary text-xl font-bold px-4">
                        <span class="hidden-count" data-layer="${index}">${layer.nodes}</span>
                    </div>
                    <button class="btn btn-circle btn-sm btn-outline hidden-plus" data-layer="${index}">
                        <i class="fas fa-plus"></i>
                    </button>
                </div>
                <select class="select select-bordered select-sm w-full hidden-activation" data-layer="${index}">
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
                const node = {
                    id: `L${layerIdx}N${nodeIdx}`,
                    layer: layerIdx,
                    index: nodeIdx,
                    x: startX + layerIdx * this.layerSpacing,
                    y: startY + nodeIdx * this.nodeSpacing,
                    type: layer.type,
                    activation: layer.activation
                };

                this.nodes.push(node);
                this.renderNode(node);
            }
        });

        // Render connections
        this.renderConnections();
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

        // Click to edit
        line.addEventListener('click', (e) => {
            e.stopPropagation();
            this.editConnection(idx);
        });

        // Right-click to delete
        line.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.deleteConnection(idx);
        });

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
        const conn = {
            from: fromNode.id,
            to: toNode.id,
            weight: (Math.random() * 2 - 1),
            bias: 0
        };

        this.connections.push(conn);
        this.render();
    }

    editConnection(connIndex) {
        const conn = this.connections[connIndex];
        if (!conn) return;

        const fromNode = this.nodes.find(n => n.id === conn.from);
        const toNode = this.nodes.find(n => n.id === conn.to);

        // Show modal
        const modal = document.getElementById('connectionModal');
        const connectionInfo = document.getElementById('connectionInfo');
        const weightInput = document.getElementById('modalWeight');
        const biasInput = document.getElementById('modalBias');

        connectionInfo.textContent = `Connection: ${fromNode.id} → ${toNode.id}`;
        weightInput.value = conn.weight.toFixed(3);
        biasInput.value = conn.bias.toFixed(3);

        // Show modal (DaisyUI uses showModal() for <dialog> elements)
        modal.showModal();

        // Save button
        const saveBtn = document.getElementById('saveConnectionBtn');
        const deleteBtn = document.getElementById('deleteConnectionBtn');
        const cancelBtn = document.getElementById('cancelConnectionBtn');

        // Remove old listeners
        const newSaveBtn = saveBtn.cloneNode(true);
        const newDeleteBtn = deleteBtn.cloneNode(true);
        const newCancelBtn = cancelBtn.cloneNode(true);

        saveBtn.parentNode.replaceChild(newSaveBtn, saveBtn);
        deleteBtn.parentNode.replaceChild(newDeleteBtn, deleteBtn);
        cancelBtn.parentNode.replaceChild(newCancelBtn, cancelBtn);

        newSaveBtn.addEventListener('click', () => {
            conn.weight = parseFloat(weightInput.value) || 0;
            conn.bias = parseFloat(biasInput.value) || 0;
            modal.close();
            this.render();
        });

        newDeleteBtn.addEventListener('click', () => {
            this.deleteConnection(connIndex);
            modal.close();
        });

        newCancelBtn.addEventListener('click', () => {
            modal.close();
        });
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
        // Convert to format needed by backend
        const layers = this.layers.map(layer => ({
            num_nodes: layer.nodes,
            activation: layer.activation
        }));

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

            connectionsArray.push({
                layer_idx: parseInt(layerIdx),
                connections: layerConns.map(nc => nc.connections),
                weights: layerConns.map(nc => nc.weights),
                biases: layerConns.map(nc => nc.bias)
            });
        });

        return {
            layers,
            connections: connectionsArray
        };
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
    networkBuilder = new InteractiveNetworkBuilder('networkCanvas');
});
