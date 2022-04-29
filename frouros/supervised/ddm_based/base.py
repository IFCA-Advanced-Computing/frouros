"""Supervised DDM based base module."""
import abc
from collections import deque
from typing import (  # noqa: TYP001
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
)

from sklearn.base import BaseEstimator, is_classifier  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import TargetDelayEstimator
from frouros.supervised.exceptions import TrainingEstimatorError
from frouros.utils.logger import logger


class DDMBaseConfig(abc.ABC):  # pylint: disable=too-few-public-methods
    """Abstract class representing a DDM based configuration class."""

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
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("min_num_instances must be great than 0.")
        self._min_num_instances = value


class DDMConfig(DDMBaseConfig):
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
            raise ValueError("warning level must be great than 0.0.")
        self._warning_level = value


class EDDMConfig(DDMBaseConfig):
    """EDDM (Early drift detection method) configuration class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        alpha: float = 0.95,
        beta: float = 0.9,
        level: float = 2.0,
        min_num_misclassified_instances: int = 30,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha: warning zone value
        :type alpha: float
        :param beta: change zone value
        :param level: level factor
        :type level: float
        :param min_num_misclassified_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_misclassified_instances: int
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
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


class DDMBasedEstimator(TargetDelayEstimator):
    """Abstract class representing a DDM based estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: DDMBaseConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: DDMBaseConfig
        """
        super().__init__(estimator=estimator)
        self.config = config
        self.actual_context_samples: List[
            Tuple[List[float], Union[str, int, float]]
        ] = []
        (
            self.delayed_predictions,
            self.ground_truth,
            self.new_context_samples,
            self.num_instances,
            self.predictions,
            self.sample_weight,
            self._drift_insufficient_samples,
        ), specific_attributes = self._init_attributes()
        for attr_name, (_, attr_value) in specific_attributes.items():
            setattr(self, attr_name, attr_value)

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

    def _add_new_context_samples(
        self, X: np.ndarray, y: np.ndarray  # noqa: N803
    ) -> None:
        self.new_context_samples.extend([*zip(X.tolist(), y.tolist())])

    def _check_drift_insufficient_samples(
        self, X: np.ndarray, y: np.ndarray  # noqa: N803
    ) -> bool:
        self._add_new_context_samples(X=X, y=y)
        _, y_new_context = self._list_to_arrays(list_=self.new_context_samples)
        num_classes = self._get_number_classes(y=y_new_context)
        if num_classes > 1:
            self._complete_delayed_drift()
            return True
        return False

    def _complete_delayed_drift(self) -> None:
        logger.warning(
            "Changing threshold has been exceeded. "
            "Drift detected with delay because there was only one class."
        )
        self._drift_insufficient_samples = False
        self._reset()

    def _drift_case(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        logger.warning("Changing threshold has been exceeded. Drift detected.")
        self._add_new_context_samples(X=X, y=y)
        X_new_context, y_new_context = self._list_to_arrays(  # noqa: N806
            list_=self.new_context_samples
        )
        if not is_classifier(self.estimator):
            self._fit_estimator(X=X_new_context, y=y_new_context)
            self._reset()

        num_classes = self._get_number_classes(y=y_new_context)
        if num_classes > 1:
            self._fit_estimator(X=X_new_context, y=y_new_context)
            self._reset()
        else:
            logger.warning(
                "Classifier estimator needs at least 2 different "
                "classes, but %s was found. Samples "
                "will be collected until this is to be fulfilled.",
                num_classes,
            )
            self._drift_insufficient_samples = True

    def _fit_estimator(self, X: np.array, y: np.array) -> None:  # noqa: N803
        try:
            self._fit_method(X=X, y=y, sample_weight=self.sample_weight)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: Increase min_num_instances value."
            ) from e

    @staticmethod
    def _get_update_response(
        drift: bool,
        warning: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        response = {
            "drift": drift,
            "warning": warning,
        }
        response.update(**kwargs)  # type: ignore
        return response

    def _init_attributes(
        self,
    ) -> Tuple[
        Tuple[
            Deque["Tuple[np.ndarray, np.ndarray]"],
            Deque["Union[str, int, float]"],
            List[Tuple[List[float], Union[str, int, float]]],
            int,
            Deque["Union[str, int, float]"],
            Any,
            bool,
        ],
        Dict[str, Tuple[Any, Any]],
    ]:
        base_attributes = self._init_base_attributes()
        specific_attributes = self._init_specific_attributes()
        return base_attributes, specific_attributes  # type: ignore

    @staticmethod
    def _init_base_attributes() -> Dict[
        str,
        Tuple[
            Deque["Tuple[np.ndarray, np.ndarray]"],
            Deque["Union[str, int, float]"],
            List[Tuple[List[float], Union[str, int, float]]],
            int,
            Deque["Union[str, int, float]"],
            Optional[Union[List[int], List[float]]],
            bool,
        ],
    ]:
        delayed_predictions: Deque["Tuple[np.ndarray, np.ndarray]"] = deque()
        ground_truth: Deque["Union[str, int, float]"] = deque()
        new_context_samples: List[Tuple[List[float], Union[str, int, float]]] = []
        num_instances = 0
        predictions: Deque["Union[str, int, float]"] = deque()
        sample_weight: Optional[Union[List[int], List[float]]] = None
        _drift_insufficient_samples = False

        return (
            delayed_predictions,
            ground_truth,
            new_context_samples,
            num_instances,
            predictions,
            sample_weight,
            _drift_insufficient_samples,
        )  # type: ignore

    @staticmethod
    @abc.abstractmethod
    def _init_specific_attributes() -> Dict[str, Tuple[Any, Any]]:
        pass

    @staticmethod
    def _list_to_arrays(
        list_: List[Tuple[np.array, Union[str, int, float]]]
    ) -> List[np.ndarray]:
        return [*map(np.array, zip(*list_))]

    def _normal_case(self, y: np.array) -> None:
        for _ in range(y.shape[0]):
            self.ground_truth.popleft()
            self.predictions.popleft()
        X, y = self._list_to_arrays(list_=self.actual_context_samples)  # noqa: N806
        self._fit_estimator(X=X, y=y)
        # Remove warning samples if performance returns to normality
        self.new_context_samples.clear()

    def _reset(self) -> None:
        self.actual_context_samples = self.new_context_samples
        (
            self.delayed_predictions,
            self.ground_truth,
            self.new_context_samples,
            self.num_instances,
            self.predictions,
            self.sample_weight,
            self._drift_insufficient_samples,
        ), specific_attributes = self._init_attributes()
        for attr_name, (_, attr_value) in specific_attributes.items():
            setattr(self, attr_name, attr_value)

    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        logger.warning(
            "Warning threshold has been exceeded. "
            "New concept will be learned until drift is detected."
        )
        self._add_new_context_samples(X=X, y=y)

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
        :raises TrainingEstimatorError: Training estimator exception
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

    @abc.abstractmethod
    def update(self, y: np.array) -> Dict[str, Optional[Union[float, bool]]]:
        """Update drift detector.

        :param y: input data
        :type y: Optional[Path]
        :return predicted values
        :rtype: numpy.ndarray
        """
