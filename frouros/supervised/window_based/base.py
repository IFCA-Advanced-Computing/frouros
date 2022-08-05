"""Supervised window based base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    Union,
)

from sklearn.base import BaseEstimator  # type: ignore

from frouros.supervised.base import SupervisedBaseEstimator, SupervisedBaseConfig


class WindowBaseConfig(SupervisedBaseConfig):
    """Class representing a window based configuration class."""


class WindowBasedEstimator(SupervisedBaseEstimator):
    """Abstract class representing a window based estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: WindowBaseConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: WindowBaseConfig
        """
        super().__init__(estimator=estimator, config=config)
        self.drift = False

    @property
    def status(self) -> Dict[str, bool]:
        """Status property.

        :return: status dict
        :rtype: Dict[str, bool]
        """
        return {"drift": self.drift}

    def reset(self, *args, **kwargs) -> None:
        """Reset method."""

    @abc.abstractmethod
    def update(self, value: Union[int, float]) -> None:
        """Abstract update method.

        :param value: value to update detector
        :type value: Union[int, float]
        """
