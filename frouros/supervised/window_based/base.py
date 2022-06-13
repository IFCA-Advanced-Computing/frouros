"""Supervised window based base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    Optional,
    Tuple,
    Union,
)

from sklearn.utils.validation import check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import TargetDelayEstimator, SupervisedBaseConfig


class WindowBaseConfig(SupervisedBaseConfig):
    """Class representing a window based configuration class."""


class WindowBasedEstimator(TargetDelayEstimator):
    """Abstract class representing a window based estimator."""

    def _clear_target_values(self):
        self.ground_truth.clear()
        self.predictions.clear()

    def _prepare_update(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
        check_is_fitted(self.estimator)
        _, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]

        metrics = self._metrics_func(y_true=y, y_pred=y_pred)
        return y_pred, metrics

    @abc.abstractmethod
    def update(
        self, y: np.array
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
