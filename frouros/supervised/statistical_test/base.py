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

    # def __init__(
    #     self,
    #     estimator: BaseEstimator,
    #     config: StatisticalTestConfig,
    #     metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    # ) -> None:
    #     """Init method.
    #
    #     :param estimator: sklearn estimator
    #     :type estimator: BaseEstimator
    #     :param config: configuration parameters
    #     :type config: StatisticalTestConfig:
    #     :param metrics: performance metrics
    #     :type metrics: metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
    #     """
    #     super().__init__(
    #         estimator=estimator,
    #         config=config,
    #         metrics=metrics,
    #     )

    @abc.abstractmethod
    def update(
        self, y: np.array
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return predicted values
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
