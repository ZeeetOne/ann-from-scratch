"""
Loss Functions Module

Implements various loss functions for neural network training using Strategy Pattern.
Each loss function is a separate class following Open/Closed Principle.

Author: ANN from Scratch Team
"""

import numpy as np
from abc import ABC, abstractmethod


class LossFunction(ABC):
    """
    Abstract base class for loss functions.

    Follows Strategy Pattern - allows switching loss functions at runtime.
    """

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate loss value

        Args:
            y_true: True labels
            y_pred: Predicted values

        Returns:
            Loss value (scalar)
        """
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate derivative of loss function (for backpropagation)

        Args:
            y_true: True labels
            y_pred: Predicted values

        Returns:
            Gradient of loss with respect to predictions
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the loss function"""
        pass


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE): L = (1/n) * Σ(y_true - y_pred)²

    Use case: Regression, general purpose
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MSE loss

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MSE loss value
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        MSE derivative: dL/dy_pred = 2 * (y_pred - y_true) / n

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Gradient array
        """
        n = y_true.size
        return 2 * (y_pred - y_true) / n

    @property
    def name(self) -> str:
        return "mse"


class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross-Entropy: L = -mean(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))

    Use case: Binary classification with sigmoid output
    """

    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize Binary Cross-Entropy

        Args:
            epsilon: Small constant to prevent log(0)
        """
        self.epsilon = epsilon

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate binary cross-entropy loss

        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities [0, 1]

        Returns:
            Binary cross-entropy loss value
        """
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        return -np.mean(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Binary cross-entropy derivative:
        dL/dy_pred = -(y_true/y_pred - (1-y_true)/(1-y_pred))

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities

        Returns:
            Gradient array
        """
        # Clip predictions to prevent division by zero
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        return -(
            y_true / y_pred_clipped -
            (1 - y_true) / (1 - y_pred_clipped)
        )

    @property
    def name(self) -> str:
        return "binary"


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross-Entropy: L = -mean(Σ(y_true * log(y_pred)))

    Use case: Multi-class classification with softmax output
    """

    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize Categorical Cross-Entropy

        Args:
            epsilon: Small constant to prevent log(0)
        """
        self.epsilon = epsilon

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate categorical cross-entropy loss

        Args:
            y_true: True one-hot encoded labels
            y_pred: Predicted probabilities (after softmax)

        Returns:
            Categorical cross-entropy loss value
        """
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Only sum over true class (y_true is one-hot encoded)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Categorical cross-entropy derivative (with softmax):
        When combined with softmax, derivative simplifies to: (y_pred - y_true)

        This is a special mathematical property of softmax + CCE.

        Args:
            y_true: True one-hot encoded labels
            y_pred: Predicted probabilities

        Returns:
            Gradient array
        """
        # Special case: softmax + categorical cross-entropy
        # The derivative simplifies beautifully to (y_pred - y_true)
        return (y_pred - y_true) / y_true.shape[0]

    @property
    def name(self) -> str:
        return "categorical"


class LossFunctionFactory:
    """
    Factory class for creating loss functions.

    Follows Factory Pattern - centralizes object creation.
    """

    _loss_functions = {
        'mse': MeanSquaredError,
        'binary': BinaryCrossEntropy,
        'categorical': CategoricalCrossEntropy
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> LossFunction:
        """
        Create loss function by name

        Args:
            name: Name of loss function
            **kwargs: Additional arguments for loss function

        Returns:
            LossFunction instance

        Raises:
            ValueError: If loss function not found
        """
        name_lower = name.lower()
        if name_lower not in cls._loss_functions:
            raise ValueError(
                f"Unknown loss function: {name}. "
                f"Available: {list(cls._loss_functions.keys())}"
            )

        return cls._loss_functions[name_lower](**kwargs)

    @classmethod
    def register(cls, name: str, loss_class: type):
        """
        Register a custom loss function

        Args:
            name: Name to register
            loss_class: Class implementing LossFunction
        """
        if not issubclass(loss_class, LossFunction):
            raise TypeError("loss_class must inherit from LossFunction")

        cls._loss_functions[name.lower()] = loss_class

    @classmethod
    def available_losses(cls) -> list:
        """Get list of available loss functions"""
        return list(cls._loss_functions.keys())
