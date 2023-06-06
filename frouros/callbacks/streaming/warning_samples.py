"""Warning samples buffer callback module."""

import copy
from typing import Any, List, Optional

from frouros.callbacks.streaming.base import BaseCallbackStreaming


class WarningSamplesBuffer(BaseCallbackStreaming):
    """Store warning samples as a buffer callback class."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Init method.

        :param name: name to be use
        :type name: Optional[str]
        """
        super().__init__(name=name)
        self.X: List[Any] = []
        self.y: List[Any] = []
        self._start_warning = False

    def on_update_start(self, **kwargs) -> None:
        """On update start method."""
        self._start_warning = not self.detector.warning  # type: ignore

    def on_update_end(self, **kwargs) -> None:
        """On update end method."""
        self.logs = {
            "X": copy.deepcopy(self.X),
            "y": copy.deepcopy(self.y),
        }

    def on_warning_detected(self, **kwargs) -> None:
        """On warning detected method."""
        if self._start_warning:
            map(lambda x: x.clear(), [self.X, self.y])
        self.X.append(kwargs["X"])
        self.y.append(kwargs["y"])

    # FIXME: set_detector method as a workaround to  # pylint: disable=fixme
    #  avoid circular import problem. Make it an abstract method and
    #  uncomment commented code when it is solved

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, DDMBased):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type DDMBased."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""
        self.X.clear()
        self.y.clear()
        self._start_warning = False
