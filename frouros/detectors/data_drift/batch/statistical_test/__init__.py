"""Data drift batch statistical test detection methods' init."""

from .anderson_darling import AndersonDarlingTest
from .bws import BWSTest
from .chisquare import ChiSquareTest
from .cvm import CVMTest
from .ks import KSTest
from .kuiper_test import KuiperTest
from .mann_whitney_u import MannWhitneyUTest
from .welch_t_test import WelchTTest

__all__ = [
    "AndersonDarlingTest",
    "BWSTest",
    "ChiSquareTest",
    "CVMTest",
    "KSTest",
    "KuiperTest",
    "MannWhitneyUTest",
    "WelchTTest",
]
