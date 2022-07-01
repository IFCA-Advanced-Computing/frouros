"""RDDM (Reactive Drift detection method) module."""

from typing import Callable, Dict, Optional, List, Union  # noqa: TYP001

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.ddm_based.base import DDMBaseConfig, DDMErrorBasedEstimator
from frouros.utils.data_structures import CircularQueue


class RDDMConfig(DDMBaseConfig):
    """RDDM (Reactive Drift detection method) configuration class."""

    def __init__(
        self,
        warning_level: float = 1.773,
        drift_level: float = 2.258,
        max_concept_size: int = 40000,
        min_concept_size: int = 7000,
        max_num_instances_warning: int = 1400,
        min_num_instances: int = 129,
    ) -> None:
        """Init method.

        :param warning_level: warning level factor
        :type warning_level: float
        :param drift_level: drift level factor
        :type drift_level: float
        :param max_concept_size: maximum size of a concept
        :type max_concept_size: int
        :param min_concept_size: reduced size of a concept
        :type min_concept_size: int
        :param max_num_instances_warning: maximum number of instances at warning level
        :type max_num_instances_warning: int
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(
            drift_level=drift_level,
            warning_level=warning_level,
            min_num_instances=min_num_instances,
        )
        self.max_concept_size = max_concept_size
        self.min_concept_size = min_concept_size
        self.max_num_instances_warning = max_num_instances_warning

    @property
    def max_concept_size(self) -> int:
        """Maximum size of a concept property.

        :return: maximum size of a concept
        :rtype: int
        """
        return self._max_concept_size

    @max_concept_size.setter
    def max_concept_size(self, value: int) -> None:
        """Maximum size of a concept setter.

        :param value: value to be set
        :type value: int
        """
        self._max_concept_size = value

    @property
    def min_concept_size(self) -> int:
        """Minimum size of a concept property.

        :return: minimum size of a concept
        :rtype: int
        """
        return self._min_concept_size

    @min_concept_size.setter
    def min_concept_size(self, value: int) -> None:
        """Minimum size of a concept setter.

        :param value: value to be set
        :type value: int
        """
        self._min_concept_size = value

    @property
    def max_num_instances_warning(self) -> int:
        """Maximum number of instances at warning level property.

        :return: maximum number of instances at warning level
        :rtype: int
        """
        return self._max_num_instances_warning

    @max_num_instances_warning.setter
    def max_num_instances_warning(self, value: int) -> None:
        """Maximum number of instances at warning level setter.

        :param value: value to be set
        :type value: int
        """
        self._max_num_instances_warning = value


class RDDM(DDMErrorBasedEstimator):
    """RDDM (Reactive Drift detection method) algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: RDDMConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: RDDMConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(
            estimator=estimator,
            error_scorer=error_scorer,
            config=config,
            metrics=metrics,
        )
        self.num_warnings = 0
        self.rddm_drift = False
        self.predictions = CircularQueue(
            max_len=self.config.min_concept_size  # type: ignore
        )

    def _reset(self, *args, **kwargs) -> None:
        super()._reset()
        self.rddm_drift = False

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
            response = self._get_update_response(
                drift=True, warning=True, metrics=metrics
            )
            return response  # type: ignore

        if self.rddm_drift:
            self._rdd_drift_case()

        error_rate_sample = self.error_scorer(y_true=y, y_pred=y_pred)
        self.predictions.enqueue(value=error_rate_sample)
        self.error_rate += (error_rate_sample - self.error_rate) / self.num_instances

        if self.num_instances >= self.config.min_num_instances:
            error_rate_plus_std, std = self._calculate_error_rate_plus_std()

            self._check_min_values(error_rate_plus_std=error_rate_plus_std, std=std)

            drift_flag = self._check_threshold(
                error_rate_plus_std=error_rate_plus_std,
                min_error_rate=self.min_error_rate,
                min_std=self.min_std,
                level=self.config.drift_level,  # type: ignore
            )

            if drift_flag:
                # Out-of-Control
                self._drift_case(X=X, y=y)
                self.drift = True
                self.rddm_drift = True
                self.warning = True
                if self.num_warnings == 0:
                    self.predictions.maintain_last_element()
            else:
                warning_flag = self._check_threshold(
                    error_rate_plus_std=error_rate_plus_std,
                    min_error_rate=self.min_error_rate,
                    min_std=self.min_std,
                    level=self.config.warning_level,  # type: ignore
                )
                if warning_flag:
                    if (
                        self.num_warnings
                        >= self.config.max_num_instances_warning  # type: ignore
                    ):
                        self.rddm_drift = True
                        self.drift = True
                        self.predictions.maintain_last_element()
                    else:
                        # Warning
                        self._warning_case(X=X, y=y)
                        self.warning = True
                        self.num_warnings += 1
                else:
                    # In-Control
                    self._normal_case(X=X, y=y)
                    self.warning = False
                    self.num_warnings = 0
                if (
                    self.num_instances >= self.config.max_concept_size  # type: ignore
                    and not self.warning
                ):
                    self.rddm_drift = True
                self.drift = False
        else:
            error_rate_plus_std, self.drift, self.warning = 0.0, False, False

        response = self._get_update_response(
            drift=self.drift,
            warning=self.warning,
            error_rate_plus_std=error_rate_plus_std,
            metrics=metrics,
        )
        return response

    def _rdd_drift_case(self) -> None:
        self._reset_stats()
        pos = self.predictions.first
        for _ in range(self.predictions.count):
            self.num_instances += 1
            self.error_rate += (
                self.predictions[pos] - self.error_rate
            ) / self.num_instances
            error_rate_plus_std, std = self._calculate_error_rate_plus_std()

            if (
                self.drift
                and self.num_instances >= self.config.min_num_instances
                and error_rate_plus_std < self.min_error_rate_plus_std
            ):
                self.min_error_rate = self.error_rate
                self.min_std = std

            pos = (pos + 1) % self.config.min_concept_size  # type: ignore
        self.rddm_drift = False
        self.drift = False

    def _reset_stats(self):
        self.error_rate = 0
        self.min_error_rate = float("inf")
        self.min_std = float("inf")
        self.num_warnings = 0
        self.num_instances = 0
        self.rddm_drift = False
