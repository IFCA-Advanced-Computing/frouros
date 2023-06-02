"""Concept drift streaming base module."""

import abc

from typing import Union

from frouros.detectors.concept_drift.base import (
    ConceptDriftBase,
    BaseConceptDriftConfig,
)


class ConceptDriftStreamingBaseConfig(BaseConceptDriftConfig):
    """Abstract class representing a concept drift streaming configuration class."""


class ConceptDriftStreamingBase(ConceptDriftBase):
    """Abstract class representing a concept drift streaming detector."""

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass
