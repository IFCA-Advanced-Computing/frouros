"""Supervised CUSUM based detection methods' init."""

from .cusum import CUSUM, CUSUMConfig
from .page_hinkley import PageHinkley, PageHinkleyConfig

__all__ = [
    "CUSUM",
    "CUSUMConfig",
    "PageHinkley",
    "PageHinkleyConfig",
]
