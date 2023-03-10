"""Data drift streaming statistical test detection methods' init."""

from .ks import IncrementalKSTest

__all__ = [
    "IncrementalKSTest",
]
