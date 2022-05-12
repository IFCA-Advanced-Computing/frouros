"""Supervised DDM based base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # type: ignore
from sklearn.base import is_classifier  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from frouros.supervised.base import TargetDelayEstimator, SupervisedBaseConfig
from frouros.supervised.exceptions import TrainingEstimatorError
from frouros.utils.logger import logger


class DDMBaseConfig(SupervisedBaseConfig):
    """Class representing a DDM based configuration class."""


class DDMBasedEstimator(TargetDelayEstimator):
    """Abstract class representing a DDM based estimator."""

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
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )
        _, y_new_context = self._list_to_arrays(
            list_=self._fit_method.new_context_samples
        )
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
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )
        X_new_context, y_new_context = self._list_to_arrays(  # noqa: N806
            list_=self._fit_method.new_context_samples
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
        pass

    @staticmethod
    def _list_to_arrays(
        list_: List[Tuple[np.array, Union[str, int, float]]]
    ) -> List[np.ndarray]:
        return [*map(np.array, zip(*list_))]

    def _normal_case(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        for _ in range(y.shape[0]):
            self.ground_truth.popleft()
            self.predictions.popleft()
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

    def _reset(self) -> None:
        super()._reset()
        self._fit_method.reset()

    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        logger.warning(
            "Warning threshold has been exceeded. "
            "New concept will be learned until drift is detected."
        )
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )

    @abc.abstractmethod
    def update(self, y: np.array) -> Dict[str, Optional[Union[float, bool]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return response message
        :rtype: Dict[str, Optional[Union[float, bool]]]
        """
