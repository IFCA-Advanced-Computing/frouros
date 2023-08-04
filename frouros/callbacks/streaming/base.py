"""Base callback streaming module."""

import abc
from typing import Union

from frouros.callbacks.base import BaseCallback


class BaseCallbackStreaming(BaseCallback):
    """Callback streaming class."""

    def on_update_start(self, value: Union[int, float]) -> None:
        """On update start method.

        :param value: value used to update the detector
        :type value: Union[int, float]
        """

    def on_update_end(self, value: Union[int, float]) -> None:
        """On update end method.

        :param value: value used to update the detector
        :type value: Union[int, float]
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
