"""Concept drift streaming detection methods init."""
# FIXME: Remove pylint disable if batch methods are added
# pylint: skip-file
from .change_detection import (
    BOCD,
    BOCDConfig,
    CUSUM,
    CUSUMConfig,
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
    PageHinkley,
    PageHinkleyConfig,
)
from .statistical_process_control import (
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
)
from .window_based import (
    ADWIN,
    ADWINConfig,
    KSWIN,
    KSWINConfig,
    STEPD,
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
