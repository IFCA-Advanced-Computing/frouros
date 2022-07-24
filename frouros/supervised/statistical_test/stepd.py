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
from frouros.utils.data_structures import AccuracyQueue
from frouros.utils.decorators import check_func_parameters
from frouros.utils.logger import logger


class STEPDConfig(StatisticalTestConfig):
    """STEPD (Statistical test of equal proportions) configuration class."""

    def __init__(
        self,
        alpha_d: float = 0.003,
        alpha_w: float = 0.05,
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
        self.drift = False
        self.warning = False
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
        X, y_pred, metrics = self._prepare_update(y=y)  # noqa: N806

        if self._drift_insufficient_samples:
            self._insufficient_samples_case(X=X, y=y)
            if not self._check_drift_sufficient_samples:
                # Drift has been detected but there are no enough samples
                # to train a new model from scratch
                response = self._get_update_response(
                    drift=True, warning=False, statistical_test=None, metrics=metrics
                )
                return response  # type: ignore
            # There are enough samples to train a new model from scratch
            self._complete_delayed_drift()

        accuracy = self.accuracy_scorer(y_true=y, y_pred=y_pred)

        self.correct_total += np.sum(accuracy)
        self.window_accuracy.enqueue(value=accuracy)

        if self.num_instances >= self.min_num_instances:
            statistic = self._calculate_statistic()
            p_value = self._distribution.sf(np.abs(statistic)) * 2

            statistical_test = {"statistic": statistic, "p-value": p_value}

            if p_value < self.config.alpha_d:  # type: ignore
                # Drift case
                self._drift_case(X=X, y=y)
                self.drift = True
                self.warning = False
            else:
                if p_value < self.config.alpha_w:  # type: ignore
                    # Warning case
                    self._warning_case(X=X, y=y)
                    self.warning = True
                else:
                    # In-Control
                    self._normal_case(X=X, y=y)
                    self.warning = False
                self.drift = False
        else:
            self._normal_case(X=X, y=y)
            statistical_test, self.drift, self.warning = None, False, False

        response = self._get_update_response(
            drift=self.drift,
            warning=self.warning,
            statistical_test=statistical_test,
            metrics=metrics,
        )
        return response

    def _drift_case(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        if not self.drift:  # Check if drift message has already been shown
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
        if not self.warning:  # Check if warning message has already been shown
            logger.warning(
                "Warning threshold has been exceeded. "
                "New concept will be learned until drift is detected."
            )
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )

    def _normal_case(self, *args, **kwargs) -> None:
        X, y = kwargs.get("X"), kwargs.get("y")  # noqa: N806
        self._fit_method.add_fit_context_samples(X=X, y=y)
        X, y = self._list_to_arrays(  # noqa: N806
            list_=self._fit_method.fit_context_samples
        )
        self._fit_estimator(X=X, y=y)
        # Remove warning samples if performance returns to normality
        self._fit_method.post_fit_estimator()
