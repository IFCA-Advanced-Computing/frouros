"""Concept drift streaming detection methods init."""
# FIXME: Remove pylint disable if batch methods are added
# pylint: skip-file
from .cusum_based import (
    CUSUM,
    CUSUMConfig,
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
    PageHinkley,
    PageHinkleyConfig,
)
from .ddm_based import (
    DDM,
    DDMConfig,
    ECDDWT,
    ECDDWTConfig,
    EDDM,
    EDDMConfig,
    HDDMA,
    HDDMAConfig,
    HDDMW,
    HDDMWConfig,
    RDDM,
    RDDMConfig,
    STEPD,
    STEPDConfig,
)
from .window_based import (
    ADWIN,
    ADWINConfig,
    KSWIN,
    KSWINConfig,
)

__all__ = [
    "ADWIN",
    "ADWINConfig",
    "CUSUM",
    "CUSUMConfig",
    "DDM",
    "DDMConfig",
    "ECDDWT",
    "ECDDWTConfig",
    "EDDM",
    "EDDMConfig",
    "GeometricMovingAverage",
    "GeometricMovingAverageConfig",
    "HDDMA",
    "HDDMAConfig",
    "HDDMW",
    "HDDMWConfig",
    "KSWIN",
    "KSWINConfig",
    "PageHinkley",
    "PageHinkleyConfig",
    "RDDM",
    "RDDMConfig",
    "STEPD",
    "STEPDConfig",
]
