"""Data drift exception module."""


class DimensionError(Exception):
    """Dimension exception."""


class GetStatisticalTestError(Exception):
    """Get statistical test exception."""


class MismatchDimensionError(Exception):
    """Miss match dimension exception."""


class MissingFitError(Exception):
    """Missing fit exception."""


class InsufficientSamplesError(Exception):
    """Insufficient samples exception."""
