"""Supervised exception module."""


class ArgumentError(ValueError):
    """Argument exception."""


class NoFitMethodError(Exception):
    """Not fit method exception."""


class TrainingEstimatorError(Exception):
    """Training estimator exception."""


class UpdateDetectorError(Exception):
    """Update detector exception."""
