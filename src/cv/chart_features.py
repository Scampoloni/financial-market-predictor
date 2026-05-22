"""Compatibility wrapper for CV feature extraction.

This module is kept so older imports do not break. The active implementation
for chart embeddings lives in ``src.cv.chart_classifier.ChartCNN``.
"""

from src.cv.chart_classifier import ChartCNN

__all__ = ["ChartCNN"]
