"""Supervised base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    Union,
)

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.estimator_checks import check_estimator  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.supervised.exceptions import (
    TrainingEstimatorError,
)


class SupervisedBaseConfig(abc.ABC):
    """Abstract class representing a supervised configuration class."""

    def __init__(
        self,
        min_num_instances: int,
    ) -> None:
        """Init method.

        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        self.min_num_instances = min_num_instances

    @property
    def min_num_instances(self) -> int:
        """Minimum number of instances property.

        :return: minimum number of instances to start looking for changes
        :rtype: int
        """
        return self._min_num_instances

    @min_num_instances.setter
    def min_num_instances(self, value: int) -> None:
        """Minimum number of instances setter.

        :param value: value to be set
        :type value: int
        """
        self._min_num_instances = value


class SupervisedBaseEstimator(abc.ABC):
    """Abstract class representing a delayed target."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: SupervisedBaseConfig,
    ) -> None:
        """Init method.

        :param estimator: estimator to be used
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: SupervisedBaseConfig
        """
        self.estimator = estimator
        self.config = config
        self.num_instances = 0

    @property
    def config(self) -> SupervisedBaseConfig:
        """Config property.

        :return: configuration parameters of the estimator
        :rtype: SupervisedBaseConfig
        """
        return self._config

    @config.setter
    def config(self, value: SupervisedBaseConfig) -> None:
        """Config setter.

        :param value: value to be set
        :type value: SupervisedBaseConfig
        :raises TypeError: Type error exception
        """
        if not isinstance(value, SupervisedBaseConfig):
            raise TypeError("value must be of type SupervisedBaseConfig.")
        self._config = value

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
        # FIXME: Workaround to use SGDClassifier with log_loss  # pylint: disable=fixme
        # https://github.com/scikit-learn/scikit-learn/issues/24025
        try:
            check_estimator(value)
        except AssertionError as e:
            from sklearn.linear_model import (  # type:ignore # pylint:disable=C0415
                SGDClassifier,
            )

            if not (isinstance(value, SGDClassifier) and value.loss == "log_loss"):
                raise e
        self._estimator = value

    @property
    def num_instances(self) -> int:
        """Number of instances counter property.

        :return: Number of instances counter value
        :rtype: int
        """
        return self._num_instances

    @num_instances.setter
    def num_instances(self, value: int) -> None:
        """Number of instances counter setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_instances must be greater or equal than 0.")
        self._num_instances = value

    def reset(self, *args, **kwargs) -> None:
        """Reset method."""

    @property
    def status(self) -> Dict[str, bool]:
        """Status property.

        :return: status dict
        :rtype: Dict[str, bool]
        """

    def fit(self, X: np.array, y: np.array, **kwargs):  # noqa: N803
        """Fit estimator.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :raises TrainingEstimatorError: Training estimator exception
        :return: fitted estimator
        :rtype: self
        """
        try:
            self.estimator.fit(X=X, y=y, **kwargs)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: fit the estimator with more samples."
            ) from e
        return self

    def partial_fit(self, X: np.array, y: np.array, **kwargs):  # noqa: N803
        """Partial fit estimator.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :raises TrainingEstimatorError: Training estimator exception
        :return: fitted estimator
        :rtype: self
        """
        # FIXME: partial_fit based on the estimator ¿metaclass?  # pylint: disable=fixme
        try:
            self.estimator.partial_fit(X=X, y=y, **kwargs)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: fit the estimator with more samples."
            ) from e
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict values.

        :param X: input data
        :type X: numpy.ndarray
        :return: predicted values
        :rtype: numpy.ndarray
        """
        check_is_fitted(self.estimator)
        X = check_array(X)  # noqa: N806
        y_pred = self.estimator.predict(X=X)
        return y_pred

    @abc.abstractmethod
    def update(self, value: Union[int, float]) -> None:
        """Abstract update method.

        :param value: value to update detector
        :type value: Union[int, float]
        """
