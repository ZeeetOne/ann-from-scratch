"""
Activation Functions Module

Implements various activation functions for neural networks using Strategy Pattern.
Each activation function is a separate class following Open/Closed Principle.

Author: ANN from Scratch Team
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.

    Follows Strategy Pattern - allows switching activation functions at runtime.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply activation function (forward pass)

        Args:
            x: Input array

        Returns:
            Activated output
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate derivative of activation function (for backpropagation)

        Args:
            x: Input array (pre-activation values)

        Returns:
            Derivative values
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the activation function"""
        pass


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))

    Output range: (0, 1)
    Use case: Binary classification, hidden layers
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid activation

        Uses clipping to prevent overflow for large values
        """
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
        """
        sig = self.forward(x)
        return sig * (1 - sig)

    @property
    def name(self) -> str:
        return "sigmoid"


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit: f(x) = max(0, x)

    Output range: [0, ∞)
    Use case: Hidden layers (most common in deep networks)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU activation"""
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU derivative: f'(x) = 1 if x > 0, else 0
        """
        return (x > 0).astype(float)

    @property
    def name(self) -> str:
        return "relu"


class Linear(ActivationFunction):
    """
    Linear activation (identity function): f(x) = x

    Output range: (-∞, ∞)
    Use case: Input layers, regression output layers
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply linear activation (no transformation)"""
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Linear derivative: f'(x) = 1
        """
        return np.ones_like(x)

    @property
    def name(self) -> str:
        return "linear"


class Softmax(ActivationFunction):
    """
    Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))

    Output range: (0, 1) with Σ(outputs) = 1
    Use case: Multi-class classification output layer
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax activation

        Uses numerical stability trick: subtract max value
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax derivative is complex, but when combined with
        categorical cross-entropy, it simplifies to (y_pred - y_true).

        For general use, we return ones (handled in loss function).
        """
        return np.ones_like(x)

    @property
    def name(self) -> str:
        return "softmax"


class Threshold(ActivationFunction):
    """
    Threshold activation: f(x) = 1 if x > threshold, else 0

    Output range: {0, 1}
    Use case: Binary classification (not differentiable, use for inference only)
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize threshold activation

        Args:
            threshold: Threshold value (default: 0.5)
        """
        self.threshold = threshold

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply threshold activation"""
        return (x > self.threshold).astype(float)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Threshold is not differentiable.
        Use linear derivative as approximation for backprop.
        """
        return np.ones_like(x)

    @property
    def name(self) -> str:
        return "threshold"


class ActivationFactory:
    """
    Factory class for creating activation functions.

    Follows Factory Pattern - centralizes object creation.
    """

    _activations = {
        'sigmoid': Sigmoid,
        'relu': ReLU,
        'linear': Linear,
        'softmax': Softmax,
        'threshold': Threshold
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> ActivationFunction:
        """
        Create activation function by name

        Args:
            name: Name of activation function
            **kwargs: Additional arguments for activation function

        Returns:
            ActivationFunction instance

        Raises:
            ValueError: If activation function not found
        """
        name_lower = name.lower()
        if name_lower not in cls._activations:
            raise ValueError(
                f"Unknown activation function: {name}. "
                f"Available: {list(cls._activations.keys())}"
            )

        return cls._activations[name_lower](**kwargs)

    @classmethod
    def register(cls, name: str, activation_class: type):
        """
        Register a custom activation function

        Args:
            name: Name to register
            activation_class: Class implementing ActivationFunction
        """
        if not issubclass(activation_class, ActivationFunction):
            raise TypeError("activation_class must inherit from ActivationFunction")

        cls._activations[name.lower()] = activation_class

    @classmethod
    def available_activations(cls) -> list:
        """Get list of available activation functions"""
        return list(cls._activations.keys())
