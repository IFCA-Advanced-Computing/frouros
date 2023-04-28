"""Data drift streaming detection methods init."""

from .distance_based import MMD
from .statistical_test import IncrementalKSTest

__all__ = [
    "IncrementalKSTest",
    "MMD",
]
