"""Prequential error using fading factor metric module."""

from typing import Callable, Optional, Union

import numpy as np  # type: ignore

from frouros.metrics.prequential_error.base import PrequentialErrorBase


class PrequentialErrorFadingFactor(PrequentialErrorBase):
    """Prequential error using fading factor metric class."""

    def __init__(
        self,
        error_scorer: Callable,
        alpha: Union[int, float],
        name: Optional[str] = None,
    ) -> None:
        """Init method.

        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param alpha: fading factor value
        :type alpha: Union[int, float]
        :param name: metric´s name
        :type name: Optional[str]
        """
        super().__init__(error_scorer=error_scorer, name=name)
        self.alpha = alpha
        self.cumulative_instances = 0.0

    @property
    def alpha(self) -> Union[int, float]:
        """Fading factor property.

        :return: fading factor value
        :rtype: Union[int, float]
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: Union[int, float]) -> None:
        """Fading factor setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self._alpha = value

    @property
    def cumulative_instances(self) -> Union[int, float]:
        """Cumulative instances' property.

        :return: fading factor value
        :rtype: Union[int, float]
        """
        return self._cumulative_instances

    @cumulative_instances.setter
    def cumulative_instances(self, value: Union[int, float]) -> None:
        """Cumulative instances' setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self._cumulative_instances = value

    @property
    def cumulative_fading_error(self) -> Union[int, float]:
        """Cumulative fading error property.

        :return: cumulative facing error value
        :rtype: Union[int, float]
        """
        return self.cumulative_error / self.cumulative_instances

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, *args, **kwargs
    ) -> Union[int, float]:
        """__call__ method that calculates the prequential error using fading factor.

        :param y_true ground truth values
        :type y_true: numpy.ndarray
        :param y_pred: predicted values
        :type y_pred: numpy.ndarray
        :return: cumulative facing error
        :rtype: Union[int, float]
        """
        error_rate = self.error_scorer(y_true=y_true, y_pred=y_pred, **kwargs)
        self.cumulative_error = self.cumulative_error * self.alpha + error_rate
        self.cumulative_instances = self.cumulative_instances * self.alpha + 1
        return self.cumulative_fading_error

    def reset(self) -> None:
        """Reset cumulative error."""
        self.cumulative_error = 0.0
        self.cumulative_instances = 0
