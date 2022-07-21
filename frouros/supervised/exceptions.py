"""Supervised exception module."""


class InvalidAverageRunLengthError(Exception):
    """Invalid average run length exception."""


class NoFitMethodError(Exception):
    """Not fit method exception."""


class TrainingEstimatorError(Exception):
    """Training estimator exception."""


class UpdateDetectorError(Exception):
    """Update detector exception."""
