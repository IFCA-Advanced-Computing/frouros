"""Base concept drift window based module."""

import abc
from typing import Union

from frouros.detectors.concept_drift.streaming.base import (
    BaseConceptDriftStreaming,
    BaseConceptDriftStreamingConfig,
)


class BaseWindowConfig(BaseConceptDriftStreamingConfig):
    """Class representing a window based configuration class."""


class BaseWindow(BaseConceptDriftStreaming):
    """Abstract class representing a window based."""

    config_type = BaseWindowConfig

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass
