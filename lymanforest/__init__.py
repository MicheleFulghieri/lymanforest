"""
lymanforest: A Python package to process Lyman-alpha forest snapshots.
"""

# Main function display for quicker access
from .pipeline import process_snapshot

# Package version definition (must match setup.py)
__version__ = "0.1.0"

# Defines what is exported with "from lymanforest import *"
__all__ = ['process_snapshot']
