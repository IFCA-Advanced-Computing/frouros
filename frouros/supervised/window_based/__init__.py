"""Supervised window based detection methods' init."""

from .adwin import ADWIN, ADWINConfig
from .kswin import KSWIN, KSWINConfig

__all__ = [
    "ADWIN",
    "ADWINConfig",
    "KSWIN",
    "KSWINConfig",
]
