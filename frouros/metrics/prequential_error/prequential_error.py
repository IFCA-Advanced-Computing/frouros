"""Prequential error metric module."""

from typing import Callable, Optional, Union

import numpy as np  # type: ignore

from frouros.metrics.base import BaseMetric


class PrequentialError(BaseMetric):
    """Prequential error metric class."""

    def __init__(self, error_scorer: Callable, name: Optional[str] = None) -> None:
        """Init method.

        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param name: metric´s name
        :type name: Optional[str]
        """
        super().__init__(error_scorer=error_scorer, name=name)
        self.cumulative_error = 0.0
        self.num_instances = 0

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, *args, **kwargs
    ) -> Union[int, float]:
        """__call__ method that calculates the prequential error.

        :param y_true ground truth values
        :type y_true: numpy.ndarray
        :param y_pred: predicted values
        :type y_pred: numpy.ndarray
        :return: cumulative error
        :rtype: Union[int, float]
        """
        error_rate = self.error_scorer(y_true=y_true, y_pred=y_pred, **kwargs)
        self.num_instances += y_true.shape[0]
        self.cumulative_error += (
            error_rate - self.cumulative_error
        ) / self.num_instances
        return self.cumulative_error

    def reset(self) -> None:
        """Reset cumulative error."""
        self.cumulative_error = 0.0
        self.num_instances = 0
