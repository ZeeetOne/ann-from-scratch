/**
 * API Client - Centralized backend communication
 * Handles all HTTP requests to the Flask backend
 */

export class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }

    // Network endpoints
    async buildNetwork(config) {
        return this.request('/build_network', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    async getNetworkInfo() {
        return this.request('/network_info', { method: 'GET' });
    }

    async quickStartBinary() {
        return this.request('/quick_start_binary', { method: 'POST' });
    }

    async quickStartMulticlass() {
        return this.request('/quick_start_multiclass', { method: 'POST' });
    }

    // Prediction endpoints
    async predict(data) {
        return this.request('/predict', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async forwardPass(data) {
        return this.request('/forward_pass', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // Training endpoints
    async train(config) {
        return this.request('/train', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    async calculateLoss(data) {
        return this.request('/calculate_loss', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async backpropagation(data) {
        return this.request('/backpropagation', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async updateWeights(data) {
        return this.request('/update_weights', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
}

// Export singleton instance
export const api = new APIClient();
