# streamlit_app.py — Root entry point for Streamlit Cloud deployment.
#
# Streamlit Cloud reruns the entry script on every user interaction.
# Using exec() ensures dashboard.py is re-executed each rerun instead
# of being served from Python's module cache (which would cause a blank UI).

import os
import sys

# Add project root to Python path so src/ and other modules resolve correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Execute dashboard fresh on every Streamlit rerun (exec avoids module cache)
_dashboard_path = os.path.join(BASE_DIR, "app", "dashboard.py")
exec(open(_dashboard_path, encoding="utf-8").read())  # noqa: S102  # pyrefly: ignore [missing-import]
