"""
backend/confidence/__init__.py
===============================
SatQuery AI — Confidence Package

Public API
----------
UnifiedConfidenceEngine
    Computes calibrated confidence scores for VQA, Grounding,
    Change Detection, and Fusion tasks.

TemperatureScaler
    Post-hoc calibration module. Learns a scalar temperature T on
    a held-out validation set to minimise NLL.

compute_ece
    Computes Expected Calibration Error given confidence scores and
    binary correctness labels.
"""

from .unified_confidence_engine import (
    UnifiedConfidenceEngine,
    TemperatureScaler,
    compute_ece,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
)

__all__ = [
    "UnifiedConfidenceEngine",
    "TemperatureScaler",
    "compute_ece",
    "HIGH_THRESHOLD",
    "MEDIUM_THRESHOLD",
]
