"""Supervised DDM based base module."""

import abc
from typing import (  # noqa: TYP001
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, is_classifier  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.base import (
    SupervisedBaseConfig,
    SupervisedBaseEstimatorReFit,
)
from frouros.supervised.exceptions import InvalidAverageRunLengthError
from frouros.utils.decorators import check_func_parameters
from frouros.utils.logger import logger


class DDMBaseConfig(SupervisedBaseConfig):
    """Class representing a DDM based configuration class."""

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param warning_level: warning level factor
        :type warning_level: float
        :param drift_level: drift level factor
        :type drift_level: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.warning_level = warning_level
        self.drift_level = drift_level

    @property
    def drift_level(self) -> float:
        """Drift level property.

        :return: drift level to use in detecting drift
        :rtype: float
        """
        return self._drift_level  # type: ignore

    @drift_level.setter
    def drift_level(self, value: float) -> None:
        """Drift level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("drift level must be greater than 0.0.")
        if value <= self.warning_level:
            raise ValueError("drift level must be greater than warning level.")
        self._drift_level = value

    @property
    def warning_level(self) -> float:
        """Warning level property.

        :return: warning level to use in detecting drift
        :rtype: float
        """
        return self._warning_level

    @warning_level.setter
    def warning_level(self, value: float) -> None:
        """Warning level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("warning level must be greater than 0.0.")
        self._warning_level = value


class DDMBasedEstimator(SupervisedBaseEstimatorReFit):
    """Abstract class representing a DDM based estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: SupervisedBaseConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: estimator to be used
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: SupervisedBaseConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(
            estimator=estimator,
            config=config,
            metrics=metrics,
        )
        self.error_scorer = error_scorer  # type: ignore
        self.drift = False
        self.warning = False

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

    def _drift_case(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        if not self.drift:  # Check if drift message has already been shown
            logger.warning("Changing threshold has been exceeded. Drift detected.")
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )
        X_new_context, y_new_context = self._list_to_arrays(  # noqa: N806
            list_=self._fit_method.new_context_samples
        )
        if not is_classifier(self.estimator):
            self._fit_estimator(X=X_new_context, y=y_new_context)
            self._reset()
        self._check_number_classes(
            X_new_context=X_new_context, y_new_context=y_new_context
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

    def _prepare_update(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
        check_is_fitted(self.estimator)
        X, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]

        metrics = self._metrics_func(y_true=y, y_pred=y_pred)
        return X, y_pred, metrics

    def _reset(self, *args, **kwargs) -> None:
        self.num_instances = 0
        self._drift_insufficient_samples = False
        self.sample_weight = None
        self.delayed_predictions.clear()
        self._fit_method.reset()
        self.drift = False
        self.warning = False

    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        if not self.warning:  # Check if warning message has already been shown
            logger.warning(
                "Warning threshold has been exceeded. "
                "New concept will be learned until drift is detected."
            )
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )

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


class DDMErrorBasedEstimator(DDMBasedEstimator):
    """Abstract class representing a DDM error based estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: DDMBaseConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: DDMBaseConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(
            estimator=estimator,
            error_scorer=error_scorer,
            config=config,
            metrics=metrics,
        )
        self.error_rate = 0
        self.min_error_rate = float("inf")
        self.min_std = float("inf")

    @property
    def error_rate(self) -> float:
        """Error rate property.

        :return: error rate to use
        :rtype: float
        """
        return self._error_rate

    @error_rate.setter
    def error_rate(self, value: float) -> None:
        """Error rate setter.

        :param value: value to be set
        :type value: float
        """
        self._error_rate = value

    @property
    def min_error_rate(self) -> float:
        """Minimum error rate property.

        :return: minimum error rate to use
        :rtype: float
        """
        return self._min_error_rate

    @min_error_rate.setter
    def min_error_rate(self, value: float) -> None:
        """Minimum error rate setter.

        :param value: value to be set
        :type value: float
        """
        if value < 0:
            raise ValueError("min_error_rate must be great or equal than 0.")
        self._min_error_rate = value

    @property
    def min_error_rate_plus_std(self) -> float:
        """Minimum error rate + std property.

        :return: minimum error rate + std to determine if a change is happening
        :rtype: float
        """
        return self.min_error_rate + self.min_std

    @property
    def min_std(self) -> float:
        """Minimum standard deviation property.

        :return: minimum standard deviation to use
        :rtype: float
        """
        return self._min_std

    @min_std.setter
    def min_std(self, value: float) -> None:
        """Minimum standard deviation setter.

        :param value: value to be set
        :type value: float
        """
        if value < 0:
            raise ValueError("min_std must be great or equal than 0.")
        self._min_std = value

    def _calculate_error_rate_plus_std(self) -> Tuple[float, float]:
        std = np.sqrt(self.error_rate * (1 - self.error_rate) / self.num_instances)
        error_rate_plus_std = self.error_rate + std
        return error_rate_plus_std, std

    def _check_min_values(self, error_rate_plus_std: float, std: float) -> None:
        if error_rate_plus_std < self.min_error_rate_plus_std:
            self.min_error_rate = self.error_rate
            self.min_std = std

    @staticmethod
    def _check_threshold(
        error_rate_plus_std: float, min_error_rate: float, min_std: float, level: float
    ) -> bool:
        return error_rate_plus_std > min_error_rate + level * min_std

    def _reset(self, *args, **kwargs) -> None:
        super()._reset()
        self.error_rate = 0
        self.min_error_rate = float("inf")
        self.min_std = float("inf")


class ECDDBaseConfig(SupervisedBaseConfig):
    """Class representing a ECDD configuration class."""

    average_run_length_map = {
        100: lambda p: 2.76
        - 6.23 * p
        + 18.12 * np.power(p, 3)
        - 312.45 * np.power(p, 5)
        + 1002.18 * np.power(p, 7),
        400: lambda p: 3.97
        - 6.56 * p
        + 48.73 * np.power(p, 3)
        - 330.13 * np.power(p, 5)
        + 848.18 * np.power(p, 7),
        1000: lambda p: 1.17
        + 7.56 * p
        - 21.24 * np.power(p, 3)
        + 112.12 * np.power(p, 5)
        - 987.23 * np.power(p, 7),
    }

    def __init__(
        self,
        lambda_: float = 0.2,
        average_run_length: int = 400,
        warning_level: float = 0.5,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param average_run_length: expected time between false positive detections
        :type average_run_length: int
        :param lambda_: weight given to recent data compared to older data
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        try:
            self.control_limit_func = self.average_run_length_map[average_run_length]
        except KeyError as e:
            raise InvalidAverageRunLengthError(
                "average_run_length must be 100, 400 or 1000."
            ) from e
        self.lambda_ = lambda_
        self.warning_level = warning_level

    @property
    def lambda_(self) -> float:
        """Weight recent data property.

        :return: weight given to recent data
        :rtype: float
        """
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        """Weight recent data setter.

        :param value: value to be set
        :type value: float
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("lambda_ must be in the range [0, 1].")
        self._lambda_ = value

    @property
    def warning_level(self) -> float:
        """Warning level property.

        :return: warning level to use in detecting drift
        :rtype: float
        """
        return self._warning_level

    @warning_level.setter
    def warning_level(self, value: float) -> None:
        """Warning level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 < value < 1.0:
            raise ValueError("warning level must be in the range (0.0, 1.0).")
        self._warning_level = value
