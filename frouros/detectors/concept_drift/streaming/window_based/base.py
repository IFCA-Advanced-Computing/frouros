"""Supervised window based base module."""

import abc
from typing import Union

from frouros.detectors.concept_drift.streaming.base import (
    ConceptDriftStreamingBase,
    ConceptDriftStreamingBaseConfig,
)


class WindowBaseConfig(ConceptDriftStreamingBaseConfig):
    """Class representing a window based configuration class."""


class WindowBased(ConceptDriftStreamingBase):
    """Abstract class representing a window based."""

    config_type = WindowBaseConfig

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass
