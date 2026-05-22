import sys
import os

# Add project root to Python path so `src` can be imported as a package.
sys.path.append(os.path.dirname(__file__))

from src import Detector

detector = Detector()
print(detector.detect("some data"))

