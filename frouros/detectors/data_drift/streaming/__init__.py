"""Data drift streaming detection methods init."""

from .statistical_test import IncrementalKSTest

__all__ = [
    "IncrementalKSTest",
]
