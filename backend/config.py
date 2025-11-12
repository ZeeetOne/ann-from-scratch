"""
Configuration Module

Application configuration and constants.

Author: ANN from Scratch Team
"""

import os


class Config:
    """
    Application configuration class

    Centralizes all configuration in one place.
    """

    # Flask Configuration
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'
    HOST = os.environ.get('HOST', 'localhost')
    PORT = int(os.environ.get('PORT', 5000))

    # CORS Configuration
    CORS_ENABLED = True
    CORS_ORIGINS = ['*']

    # API Configuration
    API_VERSION = '2.0.0'
    API_PREFIX = ''  # No prefix, routes at root

    # Training Limits
    MAX_EPOCHS = 100000
    MIN_EPOCHS = 1
    MAX_LEARNING_RATE = 100.0
    MIN_LEARNING_RATE = 0.0001
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_EPOCHS = 100

    # Network Limits
    MAX_LAYERS = 100
    MIN_LAYERS = 2
    MAX_NODES_PER_LAYER = 10000
    MIN_NODES_PER_LAYER = 1

    # Dataset Limits
    MAX_DATASET_SIZE_MB = 10  # 10 MB
    MAX_SAMPLES = 100000

    # Supported Options
    SUPPORTED_ACTIVATIONS = ['sigmoid', 'relu', 'linear', 'softmax', 'threshold']
    SUPPORTED_LOSSES = ['mse', 'binary', 'categorical']
    SUPPORTED_OPTIMIZERS = ['gd', 'sgd', 'momentum']

    # Frontend Configuration
    STATIC_FOLDER = '../frontend/static'
    TEMPLATES_FOLDER = '../frontend/templates'

    @classmethod
    def get_config_dict(cls) -> dict:
        """
        Get configuration as dictionary

        Returns:
            Dict with all configuration values
        """
        return {
            'debug': cls.DEBUG,
            'host': cls.HOST,
            'port': cls.PORT,
            'api_version': cls.API_VERSION,
            'supported_activations': cls.SUPPORTED_ACTIVATIONS,
            'supported_losses': cls.SUPPORTED_LOSSES,
            'supported_optimizers': cls.SUPPORTED_OPTIMIZERS,
            'limits': {
                'max_epochs': cls.MAX_EPOCHS,
                'min_epochs': cls.MIN_EPOCHS,
                'max_learning_rate': cls.MAX_LEARNING_RATE,
                'min_learning_rate': cls.MIN_LEARNING_RATE,
                'max_layers': cls.MAX_LAYERS,
                'min_layers': cls.MIN_LAYERS
            }
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True


# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env: str = 'default') -> Config:
    """
    Get configuration by environment name

    Args:
        env: Environment name

    Returns:
        Configuration class
    """
    return config_by_name.get(env, DevelopmentConfig)
