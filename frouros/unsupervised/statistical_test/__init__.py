"""Unsupervised statistical test detection methods' init."""

from .chisquared import ChiSquaredTest
from .cvm import CVMTest
from .ks import KSTest
from .t_test import TTest

__all__ = [
    "ChiSquaredTest",
    "CVMTest",
    "KSTest",
    "TTest",
]
