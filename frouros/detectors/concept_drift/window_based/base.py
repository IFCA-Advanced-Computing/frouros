"""Supervised window based base module."""

import abc
from typing import Union

from frouros.detectors.concept_drift.base import (
    ConceptDriftBase,
    ConceptDriftBaseConfig,
)


class WindowBaseConfig(ConceptDriftBaseConfig):
    """Class representing a window based configuration class."""


class WindowBased(ConceptDriftBase):
    """Abstract class representing a window based."""

    config_type = WindowBaseConfig

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass
