"""Base concept drift streaming module."""

import abc
from typing import Union

from frouros.detectors.concept_drift.base import (
    BaseConceptDrift,
    BaseConceptDriftConfig,
)


class BaseConceptDriftStreamingConfig(BaseConceptDriftConfig):
    """Abstract class representing a concept drift streaming configuration class."""


class BaseConceptDriftStreaming(BaseConceptDrift):
    """Abstract class representing a concept drift streaming detector."""

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass
