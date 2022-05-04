"""Supervised DDM based base module."""

import abc
import copy
from typing import (  # noqa: TYP001
    Dict,
    List,
    Optional,
    Union,
    Tuple,
)

from sklearn.base import BaseEstimator, is_classifier  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import TargetDelayEstimator, SupervisedBaseConfig
from frouros.supervised.exceptions import TrainingEstimatorError
from frouros.utils.logger import logger


class DDMBaseConfig(SupervisedBaseConfig):
    """Class representing a DDM based configuration class."""


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
        super().__init__(estimator=estimator, config=config)
        self.actual_context_samples: List[
            Tuple[List[float], Union[str, int, float]]
        ] = []
        self.new_context_samples: List[Tuple[List[float], Union[str, int, float]]] = []

    @property
    def actual_context_samples(
        self,
    ) -> List[Tuple[List[float], Union[str, int, float]]]:
        """Actual context samples property.

        :return: actual context samples
        :rtype: List[Tuple[List[float], Union[str, int, float]]]
        """
        return self._actual_context_samples

    @actual_context_samples.setter
    def actual_context_samples(
        self, value: List[Tuple[List[float], Union[str, int, float]]]
    ) -> None:
        """Actual context samples setter.

        :param value: value to be set
        :type value: List[Tuple[List[float], Union[str, int, float]]]
        """
        if not isinstance(value, List):
            raise TypeError("value must be of type List.")
        self._actual_context_samples = value

    @property
    def new_context_samples(self) -> List[Tuple[List[float], Union[str, int, float]]]:
        """New context samples property.

        :return: new context samples
        :rtype: List[Tuple[List[float], Union[str, int, float]]]
        """
        return self._new_context_samples

    @new_context_samples.setter
    def new_context_samples(
        self, value: List[Tuple[List[float], Union[str, int, float]]]
    ) -> None:
        """New context samples setter.

        :param value: value to be set
        :type value: List[Tuple[List[float], Union[str, int, float]]]
        """
        if not isinstance(value, List):
            raise TypeError("value must be of type List.")
        self._new_context_samples = value

    @staticmethod
    def _add_context_samples(
        samples_list: List[Tuple[List[float], Union[str, int, float]]],
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> None:
        samples_list.extend([*zip(X.tolist(), y.tolist())])

    def _check_drift_insufficient_samples(
        self, X: np.ndarray, y: np.ndarray  # noqa: N803
    ) -> bool:
        self._add_context_samples(samples_list=self.new_context_samples, X=X, y=y)
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
        self._add_context_samples(samples_list=self.new_context_samples, X=X, y=y)
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
                "classes, but only %s was found. Samples "
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

    def _fit_extra(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        self._add_context_samples(samples_list=self.actual_context_samples, X=X, y=y)

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
        super()._reset()
        self.actual_context_samples = copy.deepcopy(self.new_context_samples)
        self.new_context_samples.clear()

    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        logger.warning(
            "Warning threshold has been exceeded. "
            "New concept will be learned until drift is detected."
        )
        self._add_context_samples(samples_list=self.new_context_samples, X=X, y=y)

    @abc.abstractmethod
    def update(self, y: np.array) -> Dict[str, Optional[Union[float, bool]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return predicted values
        :rtype: Dict[str, Optional[Union[float, bool]]]
        """
