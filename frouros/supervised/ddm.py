"""DDM (Drift detection method) module."""

from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Union, Tuple  # noqa: TYP001

from sklearn.base import BaseEstimator, is_classifier  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import TargetDelayEstimator
from frouros.supervised.exceptions import TrainingEstimatorError
from frouros.utils.logger import logger


class DDMConfig:
    """DDM (Drift detection method) configuration class."""

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
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_num_instances = min_num_instances

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
        """
        if value <= 0.0:
            raise ValueError("drift level must be greater than 0.0.")
        if value <= self.warning_level:
            raise ValueError("drift level must be greater than warning level.")
        self._drift_level = value

    @property
    def min_num_instances(self) -> int:
        """Minimum number of instances property.

        :return: minimum number of instances to start checking if a change is happening
        :rtype: int
        """
        return self._min_num_instances

    @min_num_instances.setter
    def min_num_instances(self, value: int) -> None:
        """Minimum number of instances setter.

        :param value: value to be set
        :type value: int
        """
        if value < 1:
            raise ValueError("min_num_instances must be great than 0.")
        self._min_num_instances = value

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
        """
        if value <= 0.0:
            raise ValueError("warning level must be great than 0.0.")
        self._warning_level = value


class DDM(TargetDelayEstimator, BaseEstimator):
    """DDM (Drift detection method) algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: DDMConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: DDMConfig
        """
        super().__init__(estimator=estimator)
        self.error_scorer = error_scorer
        self.config = config
        self.min_error_rate = float("inf")
        self.min_std = float("inf")
        self.num_instances = 0
        self.delayed_predictions: Deque["Tuple[np.ndarray, np.ndarray]"] = deque()
        self.ground_truth: Deque["Union[str, int, float]"] = deque()
        self.predictions: Deque["Union[str, int, float]"] = deque()
        self.actual_context_samples: List[
            Tuple[List[float], Union[str, int, float]]
        ] = []
        self.new_context_samples: List[Tuple[List[float], Union[str, int, float]]] = []
        self.sample_weight = None
        self._drift_insufficient_samples = False

    @property
    def error_scorer(self) -> Callable:
        """Error scorer property.

        :return: error scorer to measure error rate
        :rtype: Callable
        """
        return self._error_scorer

    @error_scorer.setter
    def error_scorer(self, value: Callable) -> None:
        """Error scorer setter.

        :param value: value to be set
        :type value: Callable
        """
        self._error_scorer = value

    @property
    def min_error_rate_plus_std(self) -> float:
        """Minimum error rate + std property.

        :return: minimum error rate + std to determine if a change is happening
        :rtype: float
        """
        return self.min_error_rate + self.min_std

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
        """
        if value < 0:
            raise ValueError("num_instances must be greater or equal than 0.")
        self._num_instances = value

    @staticmethod
    def _check_threshold(
        error_rate_plus_std: float, min_error_rate: float, min_std: float, level: float
    ) -> bool:
        return error_rate_plus_std > min_error_rate + level * min_std

    def _reset(self) -> None:
        self.min_error_rate = float("inf")
        self.min_std = float("inf")
        self.num_instances = 0
        self.actual_context_samples = self.new_context_samples
        map(
            lambda x: x.clear(),  # type: ignore
            [
                self.delayed_predictions,
                self.ground_truth,
                self.predictions,
                self.new_context_samples,
            ],
        )

    def fit(
        self, X: np.array, y: np.array, sample_weight: np.array = None  # noqa: N803
    ):
        """Fit estimator.

        :param X: feature data
        :type X: Optional[Path]
        :param y: target data
        :type y: Optional[Path]
        :param sample_weight: assigns weights to each sample
        :type sample_weight: Optional[Path]
        :return fitted estimator
        :rtype: self
        """
        try:
            self._fit_method(X=X, y=y, sample_weight=sample_weight)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: fit the estimator with more samples."
            ) from e
        self.actual_context_samples.extend([*zip(X.tolist(), y.tolist())])
        self.sample_weight = sample_weight
        return self

    def predict(self, X: np.array) -> np.ndarray:  # noqa: N803
        """Predict values.

        :param X: input data
        :type X: Optional[Path]
        :return predicted values
        :rtype: numpy.ndarray
        """
        check_is_fitted(self.estimator)
        X = check_array(X)  # noqa: N806
        y_pred = self.estimator.predict(X=X)
        self.delayed_predictions.append((X, y_pred))
        return y_pred

    @staticmethod
    def _list_to_arrays(
        list_: List[Tuple[np.array, Union[str, int, float]]]
    ) -> List[np.ndarray]:
        return [*map(np.array, zip(*list_))]

    def update(self, y: np.array) -> Dict[str, Optional[Union[float, bool]]]:
        """Update drift detector.

        :param y: input data
        :type y: Optional[Path]
        :return predicted values
        :rtype: numpy.ndarray
        """
        X, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]
        if self._drift_insufficient_samples:
            _, y = self._list_to_arrays(list_=self.new_context_samples)
            num_classes = self._get_number_classes(y=y)
            if num_classes > 1:
                logger.warning(
                    "Changing threshold has been exceeded. "
                    "Drift detected with delay because there was only one class."
                )
                error_rate_plus_std, _, _ = self._calculate_error_rate_plus_std()
                drift, warning = True, True
                self._drift_insufficient_samples = False
                self._reset()
                response = self._generate_update_response(
                    drift, error_rate_plus_std, warning
                )
                return response

        self.ground_truth.extend(y)
        self.predictions.extend(y_pred)
        if self.num_instances > self.config.min_num_instances:
            check_is_fitted(self.estimator)

            error_rate_plus_std, error_rate, std = self._calculate_error_rate_plus_std()

            if error_rate_plus_std < self.min_error_rate_plus_std:
                self.min_error_rate = error_rate
                self.min_std = std

            drift = self._check_threshold(
                error_rate_plus_std=error_rate_plus_std,
                min_error_rate=self.min_error_rate,
                min_std=self.min_std,
                level=self.config.drift_level,
            )

            if drift:
                # Out-of-Control
                warning = self._drift_case(y=y)
            else:
                warning = self._check_threshold(
                    error_rate_plus_std=error_rate_plus_std,
                    min_error_rate=self.min_error_rate,
                    min_std=self.min_std,
                    level=self.config.warning_level,
                )

                if warning:
                    # Warning
                    self._warning_case(X, y)
                else:
                    # In-Control
                    self._normal_case(y)
        else:
            error_rate_plus_std, drift, warning = None, None, None  # type: ignore

        response = self._generate_update_response(drift, error_rate_plus_std, warning)
        return response

    @staticmethod
    def _generate_update_response(
        drift: Optional[bool],
        error_rate_plus_std: Optional[float],
        warning: Optional[bool],
    ) -> Dict[str, Optional[Union[float, bool]]]:
        response = {
            "error_rate_plus_std": error_rate_plus_std,
            "drift": drift,
            "warning": warning,
        }
        return response

    def _normal_case(self, y: np.array) -> None:
        for _ in range(y.shape[0]):
            self.ground_truth.popleft()
            self.predictions.popleft()
        X, y = self._list_to_arrays(list_=self.actual_context_samples)  # noqa: N806
        self._fit_estimator(X, y)

    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        logger.warning(
            "Warning threshold has been exceeded. "
            "New concept will be learned until drift is detected."
        )
        self.new_context_samples.extend([*zip(X.tolist(), y.tolist())])

    def _drift_case(self, y: np.array) -> bool:
        logger.warning("Changing threshold has been exceeded. Drift detected.")
        if is_classifier(self.estimator):
            num_classes = self._get_number_classes(y=y)
            if num_classes > 1:
                X, y = self._list_to_arrays(  # noqa: N806
                    list_=self.new_context_samples
                )
                self._fit_estimator(X, y)
                self._reset()
            else:
                logger.warning(
                    "Classifier estimator needs at least 2 different "
                    "classes, but %s was found. Samples "
                    "will be collected until this is to be fulfilled.",
                    num_classes,
                )
                self._drift_insufficient_samples = True
        else:
            X, y = self._list_to_arrays(list_=self.new_context_samples)  # noqa: N806
            self._fit_estimator(X, y)
            self._reset()
        warning = True
        return warning

    def _fit_estimator(self, X: np.array, y: np.array) -> None:  # noqa: N803
        try:
            self._fit_method(X=X, y=y, sample_weight=self.sample_weight)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: Increase min_num_instances value."
            ) from e

    def _calculate_error_rate_plus_std(self) -> Tuple[float, float, float]:
        error_rate = self.error_scorer(
            y_true=self.ground_truth, y_pred=self.predictions
        )
        std = np.sqrt(error_rate * (1 - error_rate) / self.num_instances)
        error_rate_plus_std = error_rate + std
        return error_rate_plus_std, error_rate, std
