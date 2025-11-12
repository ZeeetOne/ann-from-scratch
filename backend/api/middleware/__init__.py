"""
Middleware Module

API middleware for error handling and request processing.

Author: ANN from Scratch Team
"""

from .error_handler import register_error_handlers

__all__ = ['register_error_handlers']
