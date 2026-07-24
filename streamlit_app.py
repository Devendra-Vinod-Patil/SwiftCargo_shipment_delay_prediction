"""
streamlit_app.py — Root entry point for Streamlit Cloud deployment.

Streamlit Cloud looks for the app file at the repository root.
This file bootstraps the path and runs the main dashboard.
"""

import os
import sys

# Add project root to Python path so src/ and other modules resolve correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import and execute the main dashboard
import app.dashboard  # noqa: F401, E402
