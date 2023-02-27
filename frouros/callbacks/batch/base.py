"""Batch base test module."""

import abc

from frouros.callbacks.base import Callback


class BatchCallback(Callback):
    """Batch callback class."""

    def on_compare_start(self) -> None:
        """On compare start method."""

    def on_compare_end(self, **kwargs) -> None:
        """On compare end method."""

    # FIXME: set_detector method as a workaround to  # pylint: disable=fixme
    #  avoid circular import problem. Make it an abstract method and
    #  uncomment commented code when it is solved

    # @abc.abstractmethod
    # def set_detector(self, detector) -> None:
    #     """Set detector method."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""
