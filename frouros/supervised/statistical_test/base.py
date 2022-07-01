"""Supervised statistical test base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    Optional,
    Union,
)

import numpy as np  # type: ignore

from frouros.supervised.base import SupervisedBaseEstimator, SupervisedBaseConfig


class StatisticalTestConfig(SupervisedBaseConfig):
    """Class representing a statistical test configuration class ."""


class StatisticalTestEstimator(SupervisedBaseEstimator):
    """Statistical test algorithm class."""

    @abc.abstractmethod
    def update(
        self,
        y: np.ndarray,
        X: np.ndarray = None,  # noqa: N803
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :param X: feature data
        :type X: Optional[numpy.ndarray]
        :return predicted values
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
