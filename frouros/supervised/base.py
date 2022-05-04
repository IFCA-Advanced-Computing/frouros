"""Supervised base module."""

import abc
from collections import deque
from typing import (  # noqa: TYP001
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
)

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.estimator_checks import check_estimator  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.supervised.exceptions import NoFitMethodError, TrainingEstimatorError
from frouros.utils.logger import logger


class SupervisedBaseConfig(abc.ABC):
    """Abstract class representing a supervised configuration class ."""

    def __init__(
        self,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        self.min_num_instances = min_num_instances

    @property
    def min_num_instances(self) -> int:
        """Error scorer property.

        :return: error scorer to measure error rate
        :rtype: int
        """
        return self._min_num_instances

    @min_num_instances.setter
    def min_num_instances(self, value: int) -> None:
        """Error scorer setter.

        :param value: value to be set
        :type value: Callable
        """
        self._min_num_instances = value


class TargetDelayEstimator(abc.ABC):
    """Abstract class representing a delayed target."""

    def __init__(self, estimator: BaseEstimator, config: SupervisedBaseConfig) -> None:
        """Init method.

        :param estimator: estimator to be used
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: SupervisedBaseConfig
        """
        self.estimator = estimator
        self.config = config
        self.delayed_predictions: Deque["Tuple[np.ndarray, np.ndarray]"] = deque()
        self.ground_truth: Deque["Union[str, int, float]"] = deque()
        self.num_instances = 0
        self.predictions: Deque["Union[str, int, float]"] = deque()
        self.sample_weight: Optional[Union[List[int], List[float]]] = None
        self._drift_insufficient_samples = False

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
    def delayed_predictions(self) -> Deque["Tuple[np.ndarray, np.ndarray]"]:
        """Delayed predictions' property.

        :return: delayed predictions' deque
        :rtype: Deque["Tuple[np.ndarray, np.ndarray]"]
        """
        return self._delayed_predictions

    @delayed_predictions.setter
    def delayed_predictions(
        self, value: Deque["Tuple[np.ndarray, np.ndarray]"]
    ) -> None:
        """Delayed predictions' setter.

        :param value: value to be set
        :type value: Deque["Tuple[np.ndarray, np.ndarray]"]
        """
        self._delayed_predictions = value

    @property
    def ground_truth(self) -> Deque["Union[str, int, float]"]:
        """Ground truth property.

        :return: ground truth deque
        :rtype: Deque["Union[str, int, float]"]
        """
        return self._ground_truth

    @ground_truth.setter
    def ground_truth(self, value: Deque["Union[str, int, float]"]) -> None:
        """Ground truth setter.

        :param value: value to be set
        :type value: Deque["Union[str, int, float]"]
        """
        self._ground_truth = value

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

    @property
    def num_instances(self) -> int:
        """Minimum number of instances property.

        :return: minimum number of instances to use
        :rtype: float
        """
        return self._num_instances

    @num_instances.setter
    def num_instances(self, value: int) -> None:
        """Minimum number of instances setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_instances must be greater or equal than 0.")
        self._num_instances = value

    @property
    def predictions(self) -> Deque["Union[str, int, float]"]:
        """Predictions property.

        :return: predictions deque
        :rtype: Deque["Union[str, int, float]"]
        """
        return self._predictions

    @predictions.setter
    def predictions(self, value: Deque["Union[str, int, float]"]) -> None:
        """Predictions setter.

        :param value: value to be set
        :type value: Deque["Union[str, int, float]"]
        """
        self._predictions = value

    @property
    def sample_weight(self) -> Optional[Union[List[int], List[float]]]:
        """Sample weight property.

        :return: Sample weight value
        :rtype: Optional[Union[List[int], List[float]]]
        """
        return self._sample_weight

    @sample_weight.setter
    def sample_weight(self, value: Optional[Union[List[int], List[float]]]) -> None:
        """Sample weight setter.

        :param value: value to be set
        :type value: Optional[Union[List[int], List[float]]]
        """
        self._sample_weight = value

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

    def _fit_extra(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        pass

    @staticmethod
    def _get_number_classes(y: np.array) -> int:
        return len(np.unique(y))

    @staticmethod
    def _get_update_response(
        drift: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        response = {
            "drift": drift,
        }
        response.update(**kwargs)  # type: ignore
        return response

    def _set_specific_attributes(
        self, specific_attributes: Dict[str, Tuple[Any, Any]]
    ) -> None:
        for attr_name, (_, attr_value) in specific_attributes.items():
            setattr(self, attr_name, attr_value)

    def _reset(self) -> None:
        self.num_instances = 0
        self._drift_insufficient_samples = False
        self.sample_weight = None

        map(
            lambda x: x.clear(),  # type: ignore
            [
                self.delayed_predictions,
                self.ground_truth,
                self.predictions,
            ],
        )

    def fit(
        self,
        X: np.array,  # noqa: N803
        y: np.array,
        sample_weight: Optional[Union[List[int], List[float]]] = None,
    ):
        """Fit estimator.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :param sample_weight: assigns weights to each sample
        :type sample_weight: Optional[Union[List[int], List[float]]]
        :raises TrainingEstimatorError: Training estimator exception
        :return fitted estimator
        :rtype: self
        """
        self.sample_weight = sample_weight
        try:
            self._fit_method(X=X, y=y, sample_weight=self.sample_weight)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: fit the estimator with more samples."
            ) from e
        self._fit_extra(X=X, y=y)
        return self

    def predict(self, X: np.array) -> np.ndarray:  # noqa: N803
        """Predict values.

        :param X: input data
        :type X: numpy.ndarray
        :return predicted values
        :rtype: numpy.ndarray
        """
        check_is_fitted(self.estimator)
        X = check_array(X)  # noqa: N806
        y_pred = self.estimator.predict(X=X)
        self.delayed_predictions.append((X, y_pred))
        return y_pred

    @abc.abstractmethod
    def update(self, y: np.array) -> Dict[str, Optional[Union[float, bool]]]:
        """Update abstract method."""
