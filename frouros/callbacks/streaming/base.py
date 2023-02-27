"""Streaming base test module."""

import abc

from typing import Union

from frouros.callbacks.base import Callback


class StreamingCallback(Callback):
    """Streaming callback class."""

    def on_update_start(self, value: Union[int, float], **kwargs) -> None:
        """On update start method."""

    def on_update_end(self, value: Union[int, float], **kwargs) -> None:
        """On update end method."""

    # FIXME: set_detector method as a workaround to  # pylint: disable=fixme
    #  avoid circular import problem. Make it an abstract method and
    #  uncomment commented code when it is solved

    # @abc.abstractmethod
    # def set_detector(self, detector) -> None:
    #     """Set detector method."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""
