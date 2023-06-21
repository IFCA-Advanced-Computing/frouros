"""Base concept drift ChangeDetection based module."""

import abc
from typing import Union

from frouros.detectors.concept_drift.streaming.base import (
    BaseConceptDriftStreaming,
    BaseConceptDriftStreamingConfig,
)


class BaseChangeDetectionConfig(BaseConceptDriftStreamingConfig):
    """Class representing a ChangeDetection based configuration class."""


class BaseChangeDetection(BaseConceptDriftStreaming):
    """ChangeDetection based algorithm class."""

    config_type = BaseChangeDetectionConfig

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass
