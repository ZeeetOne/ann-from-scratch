"""
API Routes Module

Modular route definitions following Single Responsibility Principle.

Author: ANN from Scratch Team
"""

from .network_routes import network_bp
from .training_routes import training_bp
from .prediction_routes import prediction_bp
from .example_routes import example_bp

__all__ = [
    'network_bp',
    'training_bp',
    'prediction_bp',
    'example_bp'
]
