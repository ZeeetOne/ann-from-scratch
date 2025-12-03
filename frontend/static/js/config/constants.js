/**
 * Application Constants
 * Centralized configuration values
 */

export const ACTIVATION_FUNCTIONS = ['sigmoid', 'relu', 'softmax', 'linear', 'threshold'];
export const LOSS_FUNCTIONS = ['mse', 'binary_crossentropy', 'categorical_crossentropy'];
export const OPTIMIZERS = ['sgd', 'momentum', 'gd'];

export const LIMITS = {
    MAX_LAYERS: 10,
    MAX_NODES: 100,
    MAX_EPOCHS: 10000,
    MAX_LEARNING_RATE: 1.0,
    MIN_LEARNING_RATE: 0.0001,
    MAX_BATCH_SIZE: 1000
};

export const CHART_COLORS = {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#10b981',
    error: '#ef4444',
    warning: '#f59e0b',
    info: '#3b82f6'
};

export const NODE_COLORS = {
    input: '#4caf50',
    hidden: '#2196f3',
    output: '#ff9800'
};
