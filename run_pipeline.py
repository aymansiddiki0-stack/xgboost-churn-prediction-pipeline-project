"""
Entry point for the telco customer churn prediction pipeline.
Run this script from the project root directory to execute
the complete training workflow.
Usage:
    python run_pipeline.py
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import main


if __name__ == "__main__":
    main()