"""
Flask Application Factory

Creates and configures Flask application using Application Factory Pattern.
Follows best practices for scalability and testability.

Author: ANN from Scratch Team
"""

import os
from flask import Flask, render_template
from .routes import network_bp, training_bp, prediction_bp, example_bp
from .middleware import register_error_handlers
from ..config import get_config


def create_app(config_name='default'):
    """
    Application factory function

    Args:
        config_name: Configuration environment name

    Returns:
        Configured Flask application
    """
    # Create Flask app
    config = get_config(config_name)

    # Determine paths relative to backend/api directory
    api_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(api_dir)
    project_root = os.path.dirname(backend_dir)

    static_folder = os.path.join(project_root, 'frontend', 'static')
    template_folder = os.path.join(project_root, 'frontend', 'templates')

    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder
    )

    # Load configuration
    app.config.from_object(config)

    # Initialize network storage
    app.current_network = None

    # Register blueprints
    app.register_blueprint(network_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(example_bp)

    # Register error handlers
    register_error_handlers(app)

    # Register main route
    @app.route('/')
    def index():
        """Serve main page"""
        return render_template('index.html')

    @app.route('/health')
    def health():
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'version': config.API_VERSION
        }

    @app.route('/config')
    def get_config_route():
        """Get configuration endpoint"""
        return config.get_config_dict()

    return app


def run_app(config_name='default'):
    """
    Run Flask application

    Args:
        config_name: Configuration environment name
    """
    app = create_app(config_name)
    config = get_config(config_name)

    print("=" * 60)
    print(" ANN from Scratch - Refactored v2.0.0")
    print("=" * 60)
    print(f" Environment: {config_name}")
    print(f" Debug Mode: {config.DEBUG}")
    print(f" Server: http://{config.HOST}:{config.PORT}")
    print("=" * 60)
    print(" Press Ctrl+C to stop")
    print("=" * 60)

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
