"""Unsupervised statistical test detection methods' init."""

from .chisquare import ChiSquareTest
from .cvm import CVMTest
from .ks import KSTest
from .t_test import TTest

__all__ = [
    "ChiSquareTest",
    "CVMTest",
    "KSTest",
    "TTest",
]
