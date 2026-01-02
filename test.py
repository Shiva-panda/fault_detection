import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from faultdetection import Detector

detector = Detector()
print(detector.detect("some data"))

