"""Semi-supervised base module."""

import abc
from typing import List, Optional, Union

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.estimator_checks import check_estimator  # type: ignore


class SemiSupervisedBaseConfig(abc.ABC):
    """Abstract class representing a supervised configuration class."""


class SemiSupervisedBaseEstimator(abc.ABC):
    """Abstract class representing a semi-supervised estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: SemiSupervisedBaseConfig,
    ) -> None:
        """Init method.

        :param estimator: estimator to be used
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: SemiSupervisedBaseConfig
        """
        self.estimator = estimator
        self.config = config
        self.sample_weight: Optional[Union[List[int], List[float]]] = None
        self.X_samples: List[np.ndarray] = []
        self.y_samples: List[np.ndarray] = []

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

    @abc.abstractmethod
    def update(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> None:
        """Update abstract method."""
