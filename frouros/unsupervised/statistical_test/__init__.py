"""Unsupervised statistical test detection methods' init."""

from .cvm import CVMTest
from .ks import KSTest

__all__ = [
    "CVMTest",
    "KSTest",
]
