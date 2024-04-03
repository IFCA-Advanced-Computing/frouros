"""Base callback batch module."""

import abc
from typing import Any

import numpy as np

from frouros.callbacks.base import BaseCallback


class BaseCallbackBatch(BaseCallback):
    """Callback batch class."""

    def on_compare_start(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X_test: np.ndarray,
    ) -> None:
        """On compare start method.

        :param X_ref: reference data
        :type X_ref: numpy.ndarray
        :param X_test: test data
        :type X_test: numpy.ndarray
        """

    def on_compare_end(
        self,
        result: Any,
        X_ref: np.ndarray,  # noqa: N803
        X_test: np.ndarray,
    ) -> None:
        """On compare end method.

        :param result: result obtained from the `compare` method
        :type result: Any
        :param X_ref: reference data
        :type X_ref: numpy.ndarray
        :param X_test: test data
        :type X_test: numpy.ndarray
        """

    # FIXME: set_detector method as a workaround to  # pylint: disable=fixme
    #  avoid circular import problem. Make it an abstract method and
    #  uncomment commented code when it is solved

    # @abc.abstractmethod
    # def set_detector(self, detector) -> None:
    #     """Set detector method."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""
