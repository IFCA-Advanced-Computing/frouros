"""Supervised window based base module."""

import abc
from typing import Union

from frouros.detectors.concept_drift.streaming.base import (
    BaseConceptDriftStreaming,
    BaseConceptDriftStreamingConfig,
)


class WindowBaseConfig(BaseConceptDriftStreamingConfig):
    """Class representing a window based configuration class."""


class WindowBased(BaseConceptDriftStreaming):
    """Abstract class representing a window based."""

    config_type = WindowBaseConfig

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass
