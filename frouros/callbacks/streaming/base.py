"""Base callback streaming module."""

import abc

from frouros.callbacks.base import BaseCallback


class BaseCallbackStreaming(BaseCallback):
    """Callback streaming class."""

    def on_update_start(self, **kwargs) -> None:
        """On update start method."""

    def on_update_end(self, **kwargs) -> None:
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
