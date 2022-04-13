"""Supervised base module."""

import abc
from typing import Callable, Dict, Union

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.estimator_checks import check_estimator  # type: ignore

from frouros.utils.logger import logger


class NoFitMethodError(Exception):
    """Not fit method exception."""


class TargetDelayEstimator(abc.ABC):
    """Abstract class representing a delayed target."""

    def __init__(self, estimator: BaseEstimator) -> None:
        """Init method.

        :param estimator: estimator to be used
        :type estimator: BaseEstimator
        """
        self.estimator = estimator

    @property
    def estimator(self) -> BaseEstimator:
        """Estimator property.

        :return: estimator to use
        :rtype: BaseEstimator
        """
        return self._estimator

    @estimator.setter
    def estimator(self, value: BaseEstimator) -> None:
        """Estimator setter.

        :param value: value to be set
        :type value: BaseEstimator
        """
        check_estimator(value)
        self._estimator = value
        self._fit_method = self._get_fit_method()

    def _get_fit_method(self) -> Callable:
        partial_fit = getattr(self.estimator, "partial_fit", None)
        if not callable(partial_fit):
            logger.warning(
                "%s does not have partial_fit method. "
                "Therefore, with each new sample fit method will be used to train "
                "a model from scratch, increasing considerably the computational cost.",
                self.estimator,
            )
            fit = getattr(self.estimator, "fit", None)
            if not fit:
                raise NoFitMethodError(
                    f"{self.estimator} has not partial_fit or fit method."
                )
            return fit
        return partial_fit

    @staticmethod
    def _get_number_classes(y: np.array) -> int:
        return len(np.unique(y))

    @abc.abstractmethod
    def fit(
        self, X: np.array, y: np.array, sample_weight: np.array = None  # noqa: N803
    ) -> BaseEstimator:
        """Fit abstract method."""

    @abc.abstractmethod
    def predict(self, X: np.array) -> np.ndarray:  # noqa: N803
        """Predict abstract method."""

    @abc.abstractmethod
    def update(self, y: np.array) -> Dict[str, Union[float, bool]]:
        """Update abstract method."""
