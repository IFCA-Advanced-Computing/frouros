"""Callbacks base module."""

import abc
from typing import Optional


class Callback(abc.ABC):
    """Abstract class representing a callback."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Init method.

        :param name: name value
        :type name: Optional[str]
        """
        self.name = name  # type: ignore
        self.detector = None
        self.logs = {}  # type: ignore

    @property
    def name(self) -> str:
        """Name property.

        :return: name value
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        """Name method setter.

        :param value: value to be set
        :type value: Optional[str]
        :raises TypeError: Type error exception
        """
        if not isinstance(value, str) and value is not None:
            raise TypeError("name must be of type str or None.")
        self._name = self.__class__.__name__ if value is None else value

    def set_detector(self, detector) -> None:
        """Set detector method."""
        self.detector = detector

    # @property
    # def detector(self) -> Optional[ConceptDriftBase, DataDriftBatchBase]:
    #     return self._detector
    #
    # @detector.setter
    # def detector(self, value: Optional[ConceptDriftBase, DataDriftBatchBase]) -> None:
    #     if not isinstance(
    #             value, (ConceptDriftBase, DataDriftBatchBase)):
    #         raise TypeError(
    #             "value must be of type ConceptDriftBase or DataDriftBatchBase."
    #         )
    #     self._detector = value

    def on_fit_start(self) -> None:
        """On fit start method."""

    def on_fit_end(self) -> None:
        """On fit end method."""

    def on_drift_detected(self) -> None:
        """On drift detected method."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
