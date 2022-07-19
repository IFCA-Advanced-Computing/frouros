"""EDDM (Early drift detection method) module."""

import copy
from typing import Dict, Optional, List, Tuple, Union  # noqa: TYP001

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.ddm_based.base import DDMBaseConfig, DDMBasedEstimator


class EDDMConfig(DDMBaseConfig):
    """EDDM (Early drift detection method) configuration class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        alpha: float = 0.95,
        beta: float = 0.9,
        level: float = 2.0,
        min_num_misclassified_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha: warning zone value
        :type alpha: float
        :param beta: change zone value
        :type beta: float
        :param level: level factor
        :type level: float
        :param min_num_misclassified_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_misclassified_instances: int
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.level = level
        self.min_num_misclassified_instances = min_num_misclassified_instances

    @property
    def alpha(self) -> float:
        """Alpha property.

        :return: warning zone value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Alpha setter.

        :param value: value to be set
        :type value: float
        """
        self._alpha = value

    @property
    def beta(self) -> float:
        """Beta property.

        :return: change zone value
        :rtype: float
        """
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Beta setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("beta must be greater than 0.0.")
        if value >= self.alpha:
            raise ValueError("beta must be less than alpha.")
        self._beta = value

    @property
    def level(self) -> float:
        """Level property.

        :return: Level to use in detecting drift
        :rtype: float
        """
        return self._level

    @level.setter
    def level(self, value: float) -> None:
        """Level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("drift level must be greater than 0.0.")
        self._level = value

    @property
    def min_num_misclassified_instances(self) -> int:
        """Minimum number of misclassified instances property.

        :return: minimum number of misclassified instances to use
        :rtype: float
        """
        return self._min_num_misclassified_instances

    @min_num_misclassified_instances.setter
    def min_num_misclassified_instances(self, value: int) -> None:
        """Minimum number of misclassified instances setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError(
                "min_num_misclassified_instances must be greater or equal than 0."
            )
        self._min_num_misclassified_instances = value


class EDDM(DDMBasedEstimator):
    """EDDM (Early drift detection method) algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: EDDMConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: EDDMConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(
            estimator=estimator,
            error_scorer=lambda y_true, y_pred: y_true != y_pred,
            config=config,
            metrics=metrics,
        )
        self.distance_threshold = 0.0
        self.last_distance_error = 0.0
        self.max_distance_threshold = float("-inf")
        self.mean_distance_error = 0.0
        self.num_misclassified_instances = 0
        self.old_mean_distance_error = copy.copy(self.mean_distance_error)
        self.std_distance_error = 0.0
        self.variance_distance_error = 0.0

    @property
    def distance_threshold(self) -> float:
        """Distance threshold property.

        :return: distance threshold
        :rtype: float
        """
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value: float) -> None:
        """Distance threshold setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("distance_threshold must be great or equal than 0.")
        self._distance_threshold = value

    @property
    def last_distance_error(self) -> float:
        """Last distance error property.

        :return: last distance error
        :rtype: float
        """
        return self._last_distance_error

    @last_distance_error.setter
    def last_distance_error(self, value: float) -> None:
        """Last distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("last_distance_error must be great or equal than 0.")
        self._last_distance_error = value

    @property
    def max_distance_threshold(self) -> float:
        """Maximum distance threshold property.

        :return: maximum distance threshold
        :rtype: float
        """
        return self._max_distance_threshold

    @max_distance_threshold.setter
    def max_distance_threshold(self, value: float) -> None:
        """Maximum distance threshold setter.

        :param value: value to be set
        :type value: float
        """
        self._max_distance_threshold = value

    @property
    def mean_distance_error(self) -> float:
        """Mean distance error property.

        :return: mean distance error
        :rtype: float
        """
        return self._mean_distance_error

    @mean_distance_error.setter
    def mean_distance_error(self, value: float) -> None:
        """Mean distance error property.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("mean_distance_error must be great or equal than 0.")
        self._mean_distance_error = value

    @property
    def num_misclassified_instances(self) -> int:
        """Minimum number of misclassified instances property.

        :return: minimum number of misclassified instances to use
        :rtype: float
        """
        return self._num_misclassified_instances

    @num_misclassified_instances.setter
    def num_misclassified_instances(self, value: int) -> None:
        """Minimum number of misclassified instances setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError(
                "num_misclassified_instances must be greater or equal than 0."
            )
        self._num_misclassified_instances = value

    @property
    def old_mean_distance_error(self) -> float:
        """Old mean distance error property.

        :return: old mean distance error
        :rtype: float
        """
        return self._old_mean_distance_error

    @old_mean_distance_error.setter
    def old_mean_distance_error(self, value: float) -> None:
        """Old mean distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("old_mean_distance_error must be great or equal than 0.")
        self._old_mean_distance_error = value

    @property
    def std_distance_error(self) -> float:
        """Standard deviation distance error property.

        :return: standard deviation distance error
        :rtype: float
        """
        return self._std_distance_error

    @std_distance_error.setter
    def std_distance_error(self, value: float) -> None:
        """Standard deviation distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("std_distance_error must be great or equal than 0.")
        self._std_distance_error = value

    @property
    def variance_distance_error(self) -> float:
        """Variance distance error property.

        :return: variance deviation distance error
        :rtype: float
        """
        return self._variance_distance_error

    @variance_distance_error.setter
    def variance_distance_error(self, value: float) -> None:
        """Variance distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("variance must be great or equal than 0.")
        self._variance_distance_error = value

    def _normal_response(
        self, metrics: Union[BaseMetric, List[BaseMetric]]
    ) -> Dict[str, Union[bool, float]]:
        response: Dict[str, Union[bool, float]] = self._response(
            drift=False,
            warning=False,
            metrics=metrics,
        )
        return response

    def _response(
        self, drift: bool, warning: bool, metrics: Union[BaseMetric, List[BaseMetric]]
    ) -> Dict[str, Union[bool, float]]:
        response: Dict[str, Union[bool, float]] = self._get_update_response(
            drift=drift,
            warning=warning,
            metrics=metrics,
            distance_threshold=self.distance_threshold,
            max_distance_threshold=self.max_distance_threshold,
            mean_distance_error=self.mean_distance_error,
            std_distance_error=self.std_distance_error,
        )
        return response

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

        if self._drift_insufficient_samples and self._check_drift_insufficient_samples(
            X=X, y=y
        ):
            response = self._response(
                drift=True, warning=False, metrics=metrics  # type: ignore
            )
            return response  # type: ignore

        try:
            misclassified_idxs = np.argwhere(
                self.error_scorer(y_true=y, y_pred=y_pred)
            )[0]
            non_misclassified_instances = False
        except IndexError:
            non_misclassified_instances = True

        if non_misclassified_instances:
            response = self._normal_response(metrics=metrics)  # type: ignore
            return response  # type: ignore

        X = X[misclassified_idxs, :]  # noqa: N806
        y = y[misclassified_idxs]

        self.num_misclassified_instances += misclassified_idxs.size

        # Welford´s method to compute incremental mean and standard deviation
        distance = self.num_instances - self.last_distance_error
        self.old_mean_distance_error = self.mean_distance_error
        self.mean_distance_error += (
            distance - self.mean_distance_error
        ) / self.num_misclassified_instances
        self.variance_distance_error += (distance - self.mean_distance_error) * (
            distance - self.old_mean_distance_error
        )
        self.std_distance_error = (
            np.sqrt(self.variance_distance_error / self.num_misclassified_instances)
            if self.num_misclassified_instances > 0
            else 0.0
        )
        self.last_distance_error = self.num_instances

        if (
            self.num_misclassified_instances
            < self.config.min_num_misclassified_instances  # type: ignore
        ):
            response = self._normal_response(metrics=metrics)  # type: ignore
            return response  # type: ignore

        self.distance_threshold = (
            self.mean_distance_error
            + self.config.level * self.std_distance_error  # type: ignore
        )
        if self.distance_threshold > self.max_distance_threshold:
            self.max_distance_threshold = self.distance_threshold
            response = self._normal_response(metrics=metrics)  # type: ignore
            return response  # type: ignore

        p = self.distance_threshold / self.max_distance_threshold

        drift, warning = self._check_case(X=X, y=y, p=p)

        response = self._response(
            drift=drift, warning=warning, metrics=metrics  # type: ignore
        )
        return response  # type: ignore

    def _check_case(
        self, X: np.ndarray, y: np.ndarray, p: float  # noqa: N803
    ) -> Tuple[bool, bool]:
        if p < self.config.beta:  # type: ignore
            # Out-of-Control
            self._drift_case(X=X, y=y)
            self.drift = True
            self.warning = False
        else:
            if p < self.config.alpha:  # type: ignore
                # Warning
                self._warning_case(X=X, y=y)
                self.warning = True
            else:
                self._normal_case(X=X, y=y)
                self.warning = False
            self.drift = False

        return self.drift, self.warning

    def _reset(self, *args, **kwargs) -> None:
        super()._reset()
        self.distance_threshold = 0.0
        self.last_distance_error = 0.0
        self.max_distance_threshold = float("-inf")
        self.mean_distance_error = 0.0
        self.num_misclassified_instances = 0
        self.old_mean_distance_error = copy.copy(self.mean_distance_error)
        self.std_distance_error = 0.0
        self.variance_distance_error = 0.0
