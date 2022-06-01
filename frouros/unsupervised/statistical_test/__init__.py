"""Unsupervised statistical test detection methods' init."""

from .chisquared import ChiSquaredTest
from .cvm import CVMTest
from .ks import KSTest

__all__ = [
    "ChiSquaredTest",
    "CVMTest",
    "KSTest",
]
