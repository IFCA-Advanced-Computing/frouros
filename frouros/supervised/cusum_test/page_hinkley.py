"""Page Hinkley test module."""

from typing import Dict, Optional, Union  # noqa: TYP001

from sklearn.utils.validation import check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.cusum_test.base import (
    CUSUMTestEstimator,
    CUSUMTestConfig,
)


class PageHinkleyTestConfig(CUSUMTestConfig):
    """Page Hinkley test configuration class."""

    def __init__(
        self,
        delta: float = 0.005,
        forgetting_factor: float = 0.9999,
        lambda_: int = 50,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param forgetting_factor: forgetting factor value
        :type forgetting_factor: float
        :param delta: delta value
        :type delta: float
        :param lambda_: lambda value
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.delta = delta
        self.forgetting_factor = forgetting_factor
        self.lambda_ = lambda_

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
        :type value: Callable
        """
        self._delta = value

    @property
    def forgetting_factor(self) -> float:
        """Forgetting factor property.

        :return: forgetting factor value
        :rtype: float
        """
        return self._forgetting_factor

    @forgetting_factor.setter
    def forgetting_factor(self, value: float) -> None:
        """Forgetting factor setter.

        :param value: forgetting factor value
        :type value: float
        """
        self._forgetting_factor = value

    @property
    def lambda_(self) -> float:
        """Lambda property.

        :return: lambda to use
        :rtype: float
        """
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        """Lambda setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("min_error_rate must be great or equal than 0.")
        self._lambda = value


class PageHinkleyTest(CUSUMTestEstimator):
    """Page Hinkley test algorithm class."""

    def update(
        self, y: np.ndarray
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
        check_is_fitted(self.estimator)
        _, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]

        error_rate = self.error_scorer(y_true=y, y_pred=y_pred)

        self.mean_error_rate += (error_rate - self.mean_error_rate) / self.num_instances
        self.sum_ = self.config.forgetting_factor * self.sum_ + (  # type: ignore
            error_rate - self.mean_error_rate - self.config.delta  # type: ignore
        )

        if (
            self.num_instances > self.config.min_num_instances  # type: ignore
            and self.sum_ >= self.config.lambda_  # type: ignore
        ):
            response = self._get_update_response(drift=True, warning=True)
            self._reset()
            return response

        response = self._get_update_response(drift=False, warning=False)
        return response
