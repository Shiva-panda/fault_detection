# src/faultdetection/__init__.py

__version__ = "0.0.1"
__author__ = "Shiva"

from .detector import Detector  # <-- makes Detector accessible from package

def hello():
    return 'Fault Detection Package v0.0.1'
