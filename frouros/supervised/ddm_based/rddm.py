"""RDDM (Reactive Drift detection method) module."""

from typing import Union

from sklearn.base import BaseEstimator  # type: ignore

from frouros.supervised.ddm_based.base import DDMBaseConfig, DDMErrorBasedEstimator
from frouros.utils.data_structures import CircularQueue
from frouros.utils.stats import Mean


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
        config: RDDMConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: RDDMConfig
        """
        super().__init__(
            estimator=estimator,
            config=config,
        )
        self.num_warnings = 0
        self.rddm_drift = False
        self.predictions = CircularQueue(
            max_len=self.config.min_concept_size  # type: ignore
        )

    def reset(self, *args, **kwargs) -> None:
        """Reset method."""
        super().reset()
        self.rddm_drift = False

    def update(  # pylint: disable=too-many-branches
        self, value: Union[int, float]
    ) -> None:
        """Update drift detector.

        :param value: value to update detector
        :type value: Union[int, float]
        """
        self.num_instances += 1

        if self.rddm_drift:
            self._rdd_drift_case()

        self.predictions.enqueue(value=value)
        self.error_rate.update(value=value)

        if self.num_instances >= self.config.min_num_instances:
            error_rate_plus_std, std = self._calculate_error_rate_plus_std()

            self._update_min_values(error_rate_plus_std=error_rate_plus_std, std=std)

            drift_flag = self._check_threshold(
                error_rate_plus_std=error_rate_plus_std,
                min_error_rate=self.min_error_rate,
                min_std=self.min_std,
                level=self.config.drift_level,  # type: ignore
            )

            if drift_flag:
                # Out-of-Control
                self.rddm_drift = True
                self.drift = True
                self.warning = False
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
                        self.warning = True
                        self.num_warnings += 1
                        self.drift = False
                else:
                    # In-Control
                    self.drift = False
                    self.warning = False
                    self.num_warnings = 0
                if (
                    self.num_instances >= self.config.max_concept_size  # type: ignore
                    and not self.warning
                ):
                    self.rddm_drift = True
        else:
            self.drift, self.warning = False, False

    def _rdd_drift_case(self) -> None:
        self._reset_stats()
        pos = self.predictions.first
        for _ in range(self.predictions.count):
            self.num_instances += 1
            self.error_rate.update(value=self.predictions[pos])
            error_rate_plus_std, std = self._calculate_error_rate_plus_std()

            if (
                self.drift
                and self.num_instances >= self.config.min_num_instances
                and error_rate_plus_std < self.min_error_rate_plus_std
            ):
                self.min_error_rate = self.error_rate.mean
                self.min_std = std

            pos = (pos + 1) % self.config.min_concept_size  # type: ignore
        self.rddm_drift = False
        self.drift = False

    def _reset_stats(self):
        self.error_rate = Mean()
        self.min_error_rate = float("inf")
        self.min_std = float("inf")
        self.num_warnings = 0
        self.num_instances = 0
        self.rddm_drift = False
