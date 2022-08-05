"""Supervised CUSUM based base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    Union,
)

from sklearn.base import BaseEstimator  # type: ignore

from frouros.supervised.base import SupervisedBaseEstimator, SupervisedBaseConfig
from frouros.utils.stats import Mean


class CUSUMBaseConfig(SupervisedBaseConfig):
    """Class representing a CUSUM based configuration class."""

    def __init__(
        self,
        lambda_: float = 50.0,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param lambda_: lambda value
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.lambda_ = lambda_

    @property
    def lambda_(self) -> float:
        """Threshold property.

        :return: lambda to use
        :rtype: float
        """
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        """Threshold setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("lambda_ must be great or equal than 0.")
        self._lambda = value


class DeltaConfig:
    """Class representing a delta configuration class."""

    def __init__(
        self,
        delta: float = 0.005,
    ) -> None:
        """Init method.

        :param delta: delta value
        :type delta: float
        """
        self.delta = delta

    @property
    def delta(self) -> float:
        """Delta property.

        :return: delta to use
        :rtype: float
        """
        return self._delta

    @delta.setter
    def delta(self, value: float) -> None:
        """Delta setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("delta must be in the range [0, 1].")
        self._delta = value


class AlphaConfig:
    """Class representing an alpha configuration class."""

    def __init__(
        self,
        alpha: float = 0.9999,
    ) -> None:
        """Init method.

        :param alpha: forgetting factor value
        :type alpha: float
        """
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        """Forgetting factor property.

        :return: forgetting factor value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Forgetting factor setter.

        :param value: forgetting factor value
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("alpha must be in the range [0, 1].")
        self._alpha = value


class CUSUMBaseEstimator(SupervisedBaseEstimator):
    """CUSUM based algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: CUSUMBaseConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: CUSUMBaseConfig
        """
        super().__init__(estimator=estimator, config=config)
        self.mean_error_rate = Mean()
        self.sum_ = 0.0
        self.drift = False

    @property
    def status(self) -> Dict[str, bool]:
        """Status property.

        :return: status dict
        :rtype: Dict[str, bool]
        """
        return {"drift": self.drift}

    @property
    def mean_error_rate(self) -> Mean:
        """Mean error rate property.

        :return: mean error rate to use
        :rtype: Mean
        """
        return self._mean_error_rate

    @mean_error_rate.setter
    def mean_error_rate(self, value: Mean) -> None:
        """Mean error rate setter.

        :param value: value to be set
        :type value: Mean
        """
        self._mean_error_rate = value

    @property
    def sum_(self) -> float:
        """Sum count property.

        :return: sum count value
        :rtype: float
        """
        return self._sum

    @sum_.setter
    def sum_(self, value: float) -> None:
        """Sum count setter.

        :param value: value to be set
        :type value: float
        """
        self._sum = value

    @abc.abstractmethod
    def _update_sum(self, error_rate: float) -> None:
        pass

    def reset(self, *args, **kwargs) -> None:
        """Reset method."""
        self.num_instances = 0
        self.mean_error_rate = Mean()
        self.sum_ = 0.0

    def update(self, value: Union[int, float]) -> None:
        """Update drift detector.

        :param value: value to update detector
        :type value: Union[int, float]
        """
        self.num_instances += 1

        self.mean_error_rate.update(value=value)
        self._update_sum(error_rate=value)

        if (
            self.num_instances >= self.config.min_num_instances  # type: ignore
            and self.sum_ > self.config.lambda_  # type: ignore
        ):
            self.drift = True
        else:
            self.drift = False
