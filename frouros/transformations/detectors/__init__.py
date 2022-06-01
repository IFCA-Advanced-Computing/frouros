"""Transformations detectors init."""

from .categorical import CategoricalDetectors
from .numerical import NumericalDetectors

__all__ = [
    "CategoricalDetectors",
    "NumericalDetectors",
]
