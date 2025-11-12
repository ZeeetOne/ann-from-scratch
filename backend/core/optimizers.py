"""
Optimizers Module

Implements various optimization algorithms for neural network training using Strategy Pattern.
Each optimizer is a separate class following Open/Closed Principle.

Author: ANN from Scratch Team
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Follows Strategy Pattern - allows switching optimizers at runtime.
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer

        Args:
            learning_rate: Learning rate for parameter updates
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        self.learning_rate = learning_rate

    @abstractmethod
    def update(
        self,
        params: Union[float, np.ndarray],
        gradients: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Update parameters based on gradients

        Args:
            params: Current parameter values
            gradients: Gradients of loss with respect to parameters

        Returns:
            Updated parameters
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the optimizer"""
        pass

    def set_learning_rate(self, learning_rate: float):
        """
        Update learning rate

        Args:
            learning_rate: New learning rate
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        self.learning_rate = learning_rate

    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.learning_rate


class GradientDescent(Optimizer):
    """
    Batch Gradient Descent (GD) optimizer

    Update rule: θ_new = θ_old - learning_rate * gradient

    Characteristics:
    - Stable convergence
    - Computes gradient over entire dataset
    - Slower per iteration but more accurate
    """

    def update(
        self,
        params: Union[float, np.ndarray],
        gradients: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Update parameters using gradient descent

        Args:
            params: Current parameter values
            gradients: Average gradient over batch

        Returns:
            Updated parameters
        """
        return params - self.learning_rate * gradients

    @property
    def name(self) -> str:
        return "gd"


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer

    Update rule: θ_new = θ_old - learning_rate * gradient

    Characteristics:
    - Faster convergence on large datasets
    - Updates parameters per mini-batch
    - Can escape local minima due to noise
    - More volatile loss curve
    """

    def update(
        self,
        params: Union[float, np.ndarray],
        gradients: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Update parameters using SGD

        Args:
            params: Current parameter values
            gradients: Gradient from mini-batch

        Returns:
            Updated parameters
        """
        return params - self.learning_rate * gradients

    @property
    def name(self) -> str:
        return "sgd"


class Momentum(Optimizer):
    """
    Momentum optimizer (extension of SGD)

    Update rule:
        v_new = momentum * v_old + learning_rate * gradient
        θ_new = θ_old - v_new

    Characteristics:
    - Accelerates convergence
    - Reduces oscillations
    - Helps escape local minima
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize Momentum optimizer

        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (typically 0.9)
        """
        super().__init__(learning_rate)

        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1)")

        self.momentum = momentum
        self.velocity = None

    def update(
        self,
        params: Union[float, np.ndarray],
        gradients: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Update parameters using momentum

        Args:
            params: Current parameter values
            gradients: Gradients

        Returns:
            Updated parameters
        """
        # Initialize velocity on first call
        if self.velocity is None:
            if isinstance(gradients, np.ndarray):
                self.velocity = np.zeros_like(gradients)
            else:
                self.velocity = 0.0

        # Update velocity
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients

        # Update parameters
        return params - self.velocity

    @property
    def name(self) -> str:
        return "momentum"

    def reset(self):
        """Reset velocity (useful when starting new training)"""
        self.velocity = None


class OptimizerFactory:
    """
    Factory class for creating optimizers.

    Follows Factory Pattern - centralizes object creation.
    """

    _optimizers = {
        'gd': GradientDescent,
        'sgd': SGD,
        'momentum': Momentum
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> Optimizer:
        """
        Create optimizer by name

        Args:
            name: Name of optimizer
            **kwargs: Additional arguments (e.g., learning_rate, momentum)

        Returns:
            Optimizer instance

        Raises:
            ValueError: If optimizer not found
        """
        name_lower = name.lower()
        if name_lower not in cls._optimizers:
            raise ValueError(
                f"Unknown optimizer: {name}. "
                f"Available: {list(cls._optimizers.keys())}"
            )

        return cls._optimizers[name_lower](**kwargs)

    @classmethod
    def register(cls, name: str, optimizer_class: type):
        """
        Register a custom optimizer

        Args:
            name: Name to register
            optimizer_class: Class implementing Optimizer
        """
        if not issubclass(optimizer_class, Optimizer):
            raise TypeError("optimizer_class must inherit from Optimizer")

        cls._optimizers[name.lower()] = optimizer_class

    @classmethod
    def available_optimizers(cls) -> list:
        """Get list of available optimizers"""
        return list(cls._optimizers.keys())
