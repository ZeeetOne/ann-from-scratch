"""
Application Entry Point

Main entry point for running the ANN from Scratch application.

Usage:
    python run.py [--config development|production|testing]

Author: ANN from Scratch Team
"""

import sys
import os

# Add backend directory to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from backend.api.app import run_app


if __name__ == '__main__':
    # Parse command line arguments
    config_name = 'development'

    if len(sys.argv) > 1:
        if sys.argv[1] == '--config' and len(sys.argv) > 2:
            config_name = sys.argv[2]
        else:
            print("Usage: python run.py [--config development|production|testing]")
            sys.exit(1)

    # Run application
    run_app(config_name)
