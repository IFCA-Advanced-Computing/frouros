"""Supervised window based base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    Union,
)

from frouros.concept_drift.base import ConceptDriftBase, ConceptDriftBaseConfig


class WindowBaseConfig(ConceptDriftBaseConfig):
    """Class representing a window based configuration class."""


class WindowBased(ConceptDriftBase):
    """Abstract class representing a window based."""

    def __init__(
        self,
        config: WindowBaseConfig,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: WindowBaseConfig
        """
        super().__init__(config=config)
        self.drift = False

    @property
    def status(self) -> Dict[str, bool]:
        """Status property.

        :return: status dict
        :rtype: Dict[str, bool]
        """
        return {"drift": self.drift}

    def reset(self, *args, **kwargs) -> None:
        """Reset method."""

    @abc.abstractmethod
    def update(self, value: Union[int, float]) -> None:
        """Abstract update method.

        :param value: value to update detector
        :type value: Union[int, float]
        """
