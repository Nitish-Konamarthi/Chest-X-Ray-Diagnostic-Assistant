"""
Chest X-Ray Diagnostic Assistant
=================================

AI-powered chest X-ray analysis with CheXNet (DenseNet-121), Grad-CAM++
heatmaps, Gemini AI clinical explanations, and nearby doctor recommendations.

Quick Start::

    from chexray_diagnosis import ChestXRayAnalyzer

    analyzer = ChestXRayAnalyzer()
    result = analyzer.analyze("path/to/chest_xray.png")
    print(result)

CLI Usage::

    chest-xray-diagnosis serve          # Start the API server
    chest-xray-diagnosis analyze img.png  # Analyze a single image
    chest-xray-diagnosis setup           # Configure API keys
    chest-xray-diagnosis download-models # Pre-download models

License: MIT
"""

__version__ = "1.0.0"
__author__ = "Nitish Konamarthi"

from chexray_diagnosis.analyzer import ChestXRayAnalyzer

__all__ = [
    "ChestXRayAnalyzer",
    "__version__",
]
