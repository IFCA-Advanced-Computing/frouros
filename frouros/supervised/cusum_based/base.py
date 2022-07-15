"""Supervised CUSUM based base module."""

import abc
from typing import (  # noqa: TYP001
    Callable,
    Dict,
    Optional,
    Union,
)

from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.utils.decorators import check_func_parameters
from frouros.supervised.base import SupervisedBaseEstimator, SupervisedBaseConfig


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
        error_scorer: Callable,
        config: CUSUMBaseConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: CUSUMBaseConfig
        """
        super().__init__(estimator=estimator, config=config)
        self.error_scorer = error_scorer  # type: ignore
        self.mean_error_rate = 0.0
        self.sum_ = 0.0

    @property
    def error_scorer(self) -> Callable:
        """Error scorer property.

        :return: error scorer function
        :rtype: Callable
        """
        return self._error_scorer

    @error_scorer.setter  # type: ignore
    @check_func_parameters
    def error_scorer(self, value: Callable) -> None:
        """Error scorer setter.

        :param value: value to be set
        :type value: Callable
        """
        self._error_scorer = value

    @property
    def mean_error_rate(self) -> float:
        """Mean error rate property.

        :return: mean error rate to use
        :rtype: float
        """
        return self._mean_error_rate

    @mean_error_rate.setter
    def mean_error_rate(self, value: float) -> None:
        """Mean error rate setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("mean_error_rate must be great or equal than 0.")
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

    def _reset(self, *args, **kwargs) -> None:
        self.num_instances = 0
        self.mean_error_rate = 0.0
        self.sum_ = 0.0

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
        check_is_fitted(self.estimator)
        _, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]

        error_rate = self.error_scorer(y_true=y, y_pred=y_pred)

        self.mean_error_rate += (error_rate - self.mean_error_rate) / self.num_instances
        self._update_sum(error_rate=error_rate)

        if (
            self.num_instances >= self.config.min_num_instances  # type: ignore
            and self.sum_ > self.config.lambda_  # type: ignore
        ):
            response = self._get_update_response(
                drift=True, sum=self.sum_, mean_error_rate=self.mean_error_rate
            )
            self._reset()
            return response

        response = self._get_update_response(
            drift=False, sum=self.sum_, mean_error_rate=self.mean_error_rate
        )
        return response
