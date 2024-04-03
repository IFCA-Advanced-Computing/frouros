"""Concept drift streaming detection methods init."""

# FIXME: Remove pylint disable if batch methods are added
# pylint: skip-file
from .change_detection import (
    BOCD,
    CUSUM,
    BOCDConfig,
    CUSUMConfig,
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
    PageHinkley,
    PageHinkleyConfig,
)
from .statistical_process_control import (
    DDM,
    ECDDWT,
    EDDM,
    HDDMA,
    HDDMW,
    RDDM,
    DDMConfig,
    ECDDWTConfig,
    EDDMConfig,
    HDDMAConfig,
    HDDMWConfig,
    RDDMConfig,
)
from .window_based import (
    ADWIN,
    KSWIN,
    STEPD,
    ADWINConfig,
    KSWINConfig,
    STEPDConfig,
)

__all__ = [
    "ADWIN",
    "ADWINConfig",
    "BOCD",
    "BOCDConfig",
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
