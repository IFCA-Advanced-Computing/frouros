"""Supervised window based base module."""

import abc
from collections import deque
from typing import (  # noqa: TYP001
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.base import SupervisedBaseEstimator, SupervisedBaseConfig
from frouros.utils.decorators import check_func_parameters
from frouros.utils.validation import check_is_one_sample


class WindowBaseConfig(SupervisedBaseConfig):
    """Class representing a window based configuration class."""


class WindowBasedEstimator(SupervisedBaseEstimator):
    """Abstract class representing a window based estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: WindowBaseConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: WindowBaseConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(estimator=estimator, config=config, metrics=metrics)
        self.error_scorer = error_scorer  # type: ignore
        self.ground_truth: Deque["Union[str, int, float]"] = deque()
        self.predictions: Deque["Union[str, int, float]"] = deque()

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

    def _clear_target_values(self):
        self.ground_truth.clear()
        self.predictions.clear()

    def _prepare_update(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
        check_is_fitted(self.estimator)
        check_is_one_sample(array=y)
        _, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += 1

        metrics = self._metrics_func(y_true=y, y_pred=y_pred)
        return y_pred, metrics

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
        :return response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
