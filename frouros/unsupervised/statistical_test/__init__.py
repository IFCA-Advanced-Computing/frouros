"""Unsupervised statistical test detection methods' init."""

from .chisquare import ChiSquareTest
from .cvm import CVMTest
from .ks import KSTest
from .welch_t_test import WelchTTest

__all__ = [
    "ChiSquareTest",
    "CVMTest",
    "KSTest",
    "WelchTTest",
]
