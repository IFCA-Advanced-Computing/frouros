"""STEPD (Statistical test of equal proportions) module."""

from typing import Callable, Dict, Optional, List, Tuple, Union  # noqa: TYP001

from scipy.stats import norm  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.base import SupervisedBaseEstimatorReFit
from frouros.supervised.statistical_test.base import (
    StatisticalTestConfig,
    StatisticalTestEstimator,
)
from frouros.utils.decorators import check_func_parameters
from frouros.utils.logger import logger


class EmptyQueueError(Exception):
    """Empty queue exception."""

    def __init__(self, *args, msg="Queue is empty.", **kwargs) -> None:
        """Init method.

        :param msg: exception message
        :type msg: str
        """
        super().__init__(msg, *args, **kwargs)


class AccuracyQueue:
    """Class representing an accuracy queue."""

    def __init__(self, max_len: int) -> None:
        """Init method.

        :param max_len: maximum capacity
        :type max_len: int
        """
        self.count = 0
        self.first = 0
        self.last = -1
        self.max_len = max_len
        self.num_true = 0
        self.queue = [None] * self.max_len

    @property
    def count(self) -> int:
        """Number of total elements property.

        :return Number of total elements
        :rtype: int
        """
        return self._count

    @count.setter
    def count(self, value: int) -> None:
        """Number of total elements setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("count must be greater or equal than 0.")
        self._count = value

    @property
    def first(self) -> int:
        """First queue index property.

        :return first queue index
        :rtype: int
        """
        return self._first

    @first.setter
    def first(self, value: int) -> None:
        """First queue index setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("first must be greater or equal than 0.")
        self._first = value

    @property
    def last(self) -> int:
        """Last queue index property.

        :return last queue index
        :rtype: int
        """
        return self._last

    @last.setter
    def last(self, value: int) -> None:
        """Last queue index setter.

        :param value: value to be set
        :type value: int
        """
        self._last = value

    @property
    def max_len(self) -> int:
        """Maximum number of allowed elements property.

        :return maximum number of allowed elements
        :rtype: int
        """
        return self._max_len

    @max_len.setter
    def max_len(self, value: int) -> None:
        """Maximum number of allowed elements setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("max_len must be greater or equal than 0.")
        self._max_len = value

    @property
    def num_false(self):
        """Number of false label property.

        :return number of false labels
        :rtype: int
        """
        return self._count - self._num_true

    @property
    def num_true(self) -> int:
        """Number of true label property.

        :return number of true labels
        :rtype: int
        """
        return self._num_true

    @num_true.setter
    def num_true(self, value: int) -> None:
        """Number of true labels setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_true value must be greater or equal than 0.")
        self._num_true = value

    @property
    def queue(self) -> List[Optional[bool]]:
        """Queue property.

        :return queue
        :rtype: List[Optional[bool]]
        """
        return self._queue

    @queue.setter
    def queue(self, value: List[Optional[bool]]) -> None:
        """Queue setter.

        :param value: value to be set
        :type value: List[Optional[bool]]
        :raises ValueError: Value error exception
        """
        if not isinstance(value, list):
            raise TypeError("queue must be of type list.")
        self._queue = value

    @property
    def size(self) -> int:
        """Number of current elements property.

        :return Number of current elements
        :rtype: int
        """
        return self.count

    def clear(self) -> None:
        """Clear queue."""
        self.count = 0
        self.first = 0
        self.last = -1
        self.num_true = 0
        self.queue = [None] * self.max_len

    def dequeue(self) -> bool:
        """Dequeue oldest element.

        :rtype: bool
        :raises EmptyQueue: Empty queue error exception
        """
        if self.is_empty():
            raise EmptyQueueError()
        element = self.queue[self.first]
        self.first = (self.first + 1) % self.max_len
        self.num_true -= 1 if element else 0
        self.count -= 1
        return element  # type: ignore

    def enqueue(self, value: Union[bool, List[bool]]) -> None:
        """Enqueue element/s.

        :param value: value to be enqueued
        :type value: Union[bool, List[bool]]
        """
        num_values = len(value)  # type: ignore
        if self.is_full():
            for _ in range(num_values):
                _ = self.dequeue()
        self.last = (self.last + num_values) % self.max_len
        self.queue[self.last] = value  # type: ignore
        self.num_true += np.count_nonzero(value)
        self.count += num_values

    def is_empty(self) -> bool:
        """Check if queue is empty.

        :return check if queue is empty
        :rtype: bool
        """
        return self.size == 0

    def is_full(self) -> bool:
        """Check if queue is full.

        :return check if queue is full
        :rtype: bool
        """
        return self.size == self.max_len

    def __len__(self) -> int:
        """Queue size.

        :return queue size
        :rtype: int
        """
        return self.size


class SPEPDConfig(StatisticalTestConfig):
    """STEPD (Statistical test of equal proportions) configuration class."""

    def __init__(
        self,
        alpha_d: float,
        alpha_w: float,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha_d: significance value for overall
        :type alpha_d: float
        :param alpha_w: significance value for last
        :type alpha_w: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.alpha_d = alpha_d
        self.alpha_w = alpha_w

    @property
    def alpha_d(self) -> float:
        """Significance level d property.

        :return: significance level d
        :rtype: float
        """
        return self._alpha_d

    @alpha_d.setter
    def alpha_d(self, value: float) -> None:
        """Significance level d setter.

        :param value: value to be set
        :type value: float
        """
        if value <= 0.0:
            raise ValueError("alpha_d must be greater than 0.0.")
        self._alpha_d = value

    @property
    def alpha_w(self) -> float:
        """Significance level w property.

        :return: significance level w
        :rtype: float
        """
        return self._alpha_w

    @alpha_w.setter
    def alpha_w(self, value: float) -> None:
        """Significance level w setter.

        :param value: value to be set
        :type value: float
        """
        if value <= 0.0:
            raise ValueError("alpha_w must be greater than 0.0.")
        if value <= self.alpha_d:
            raise ValueError("alpha_w must be greater than alpha_d.")
        self._alpha_w = value


class STEPD(SupervisedBaseEstimatorReFit, StatisticalTestEstimator):
    """STEPD (Statistical test of equal proportions) algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: StatisticalTestConfig,
        accuracy_scorer: Callable = lambda y_true, y_pred: y_true == y_pred,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: StatisticalTestConfig
        :param accuracy_scorer: accuracy scorer function
        :type accuracy_scorer: Callable
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(
            estimator=estimator,
            config=config,
            metrics=metrics,
        )
        self.accuracy_scorer = accuracy_scorer  # type: ignore
        self.correct_total = 0
        self.min_num_instances = 2 * self.config.min_num_instances
        self.window_accuracy = AccuracyQueue(max_len=self.config.min_num_instances)
        self._distribution = norm()

    @property
    def accuracy_scorer(self) -> Callable:
        """Accuracy scorer property.

        :return: accuracy scorer function
        :rtype: Callable
        """
        return self._accuracy_scorer

    @accuracy_scorer.setter  # type: ignore
    @check_func_parameters
    def accuracy_scorer(self, value: Callable) -> None:
        """Accuracy scorer setter.

        :param value: value to be set
        :type value: Callable
        """
        self._accuracy_scorer = value

    @property
    def correct_total(self) -> int:
        """Number of correct labels property.

        :return: accuracy scorer function
        :rtype: int
        """
        return self._correct_total

    @correct_total.setter
    def correct_total(self, value: int) -> None:
        """Number of correct labels setter.

        :param value: value to be set
        :type value: int
        """
        self._correct_total = value

    @property
    def num_instances_overall(self) -> int:
        """Number of overall instances property.

        :return: number of overall instances
        :rtype: int
        """
        return self.num_instances - self.num_instances_window

    @property
    def correct_overall(self) -> int:
        """Number of correct overall labels property.

        :return: number of overall labels
        :rtype: int
        """
        return self.correct_total - self.correct_window

    @property
    def correct_window(self) -> int:
        """Number of correct window labels property.

        :return: number of window labels
        :rtype: int
        """
        return self.window_accuracy.num_true

    @property
    def num_instances_window(self) -> int:
        """Number of window instances property.

        :return: number of window instances
        :rtype: int
        """
        return self.window_accuracy.size

    def _calculate_statistic(self):
        p_hat = self.correct_total / self.num_instances
        num_instances_inv = (
            1 / self.num_instances_overall + 1 / self.num_instances_window
        )
        statistic = (
            np.abs(
                self.correct_overall / self.num_instances_overall
                - self.correct_window / self.num_instances_window
            )
            - 0.5 * num_instances_inv
        ) / np.sqrt((p_hat * (1 - p_hat) * num_instances_inv))
        return statistic

    def _reset(self, *args, **kwargs) -> None:
        super()._reset()
        self.window_accuracy.clear()

    def _prepare_update(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
        check_is_fitted(self.estimator)
        X, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]

        metrics = self._metrics_func(y_true=y, y_pred=y_pred)
        return X, y_pred, metrics

    def update(
        self, y: np.ndarray
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
        X, y_pred, metrics = self._prepare_update(y=y)  # noqa: N806

        if self._drift_insufficient_samples and self._check_drift_insufficient_samples(
            X=X, y=y
        ):
            response = self._get_update_response(
                drift=True, warning=True, metrics=metrics
            )
            return response  # type: ignore

        self.ground_truth.extend(y.tolist())
        self.predictions.extend(y_pred.tolist())

        accuracy = self.accuracy_scorer(y_true=y, y_pred=y_pred)

        self.correct_total += np.sum(accuracy)
        self.window_accuracy.enqueue(value=accuracy)

        if self.num_instances >= self.min_num_instances:
            statistic = self._calculate_statistic()
            p_value = self._distribution.sf(np.abs(statistic)) * 2

            statistical_test = {"statistic": statistic, "p-value": p_value}

            if p_value < self.config.alpha_d:  # type: ignore
                # Drift case
                drift = True
                warning = True
                self._drift_case(X=X, y=y)
            elif p_value < self.config.alpha_w:  # type: ignore
                # Warning case
                drift = False
                warning = True
                # Warning
                self._warning_case(X=X, y=y)
            else:
                # In-Control
                drift, warning = False, False
                self._normal_case()
        else:
            statistical_test, drift, warning = None, False, False

        response = self._get_update_response(
            drift=drift,
            warning=warning,
            statistical_test=statistical_test,
            metrics=metrics,
        )
        return response

    def _drift_case(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        logger.warning("Changing threshold has been exceeded. Drift detected.")
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )
        X_new_context, y_new_context = self._list_to_arrays(  # noqa: N806
            list_=self._fit_method.new_context_samples
        )
        self._check_number_classes(
            X_new_context=X_new_context, y_new_context=y_new_context
        )

    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        logger.warning(
            "Warning threshold has been exceeded. "
            "New concept will be learned until drift is detected."
        )
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )

    def _normal_case(self, *args, **kwargs) -> None:
        self._fit_method.reset()
