"""
Services Layer - Business Logic

This layer handles business logic and coordinates between API and Core.
Follows Facade Pattern to simplify complex interactions.

Author: ANN from Scratch Team
"""

from .network_service import NetworkService
from .training_service import TrainingService
from .data_service import DataService

__all__ = [
    'NetworkService',
    'TrainingService',
    'DataService'
]
