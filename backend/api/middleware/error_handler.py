"""
Error Handler Middleware

Centralized error handling for Flask application.
Provides consistent error responses across all endpoints.

Author: ANN from Scratch Team
"""

import traceback
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException
from ...utils.validators import ValidationError


def register_error_handlers(app: Flask):
    """
    Register error handlers with Flask app

    Args:
        app: Flask application instance
    """

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Handle validation errors"""
        return jsonify({
            'success': False,
            'error': str(error),
            'error_type': 'ValidationError'
        }), 400

    @app.errorhandler(ValueError)
    def handle_value_error(error):
        """Handle value errors"""
        return jsonify({
            'success': False,
            'error': str(error),
            'error_type': 'ValueError'
        }), 400

    @app.errorhandler(KeyError)
    def handle_key_error(error):
        """Handle key errors"""
        return jsonify({
            'success': False,
            'error': f'Missing required field: {str(error)}',
            'error_type': 'KeyError'
        }), 400

    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle HTTP exceptions"""
        return jsonify({
            'success': False,
            'error': error.description,
            'error_type': error.name
        }), error.code

    @app.errorhandler(Exception)
    def handle_generic_error(error):
        """Handle generic errors"""
        # Log the full traceback in development
        if app.debug:
            trace = traceback.format_exc()
            return jsonify({
                'success': False,
                'error': str(error),
                'error_type': type(error).__name__,
                'traceback': trace
            }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'error_type': 'InternalServerError'
            }), 500

    @app.before_request
    def before_request():
        """
        Hook before each request.
        Can be used for logging, authentication, etc.
        """
        # Currently no preprocessing needed
        pass

    @app.after_request
    def after_request(response):
        """
        Hook after each request.
        Add headers, logging, etc.
        """
        # Add CORS headers (if needed)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response
