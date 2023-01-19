"""Concept drift base module."""

import abc
from typing import (  # noqa: TYP001
    Dict,
    Union,
)


class ConceptDriftBaseConfig(abc.ABC):
    """Abstract class representing a concept drift configuration class."""

    def __init__(
        self,
        min_num_instances: int,
    ) -> None:
        """Init method.

        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        self.min_num_instances = min_num_instances

    @property
    def min_num_instances(self) -> int:
        """Minimum number of instances property.

        :return: minimum number of instances to start looking for changes
        :rtype: int
        """
        return self._min_num_instances

    @min_num_instances.setter
    def min_num_instances(self, value: int) -> None:
        """Minimum number of instances setter.

        :param value: value to be set
        :type value: int
        """
        self._min_num_instances = value


class ConceptDriftBase(abc.ABC):
    """Abstract class representing a delayed target."""

    def __init__(
        self,
        config: ConceptDriftBaseConfig,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: ConceptDriftBaseConfig
        """
        self.config = config
        self.num_instances = 0

    @property
    def config(self) -> ConceptDriftBaseConfig:
        """Config property.

        :return: configuration parameters of the estimator
        :rtype: ConceptDriftBaseConfig
        """
        return self._config

    @config.setter
    def config(self, value: ConceptDriftBaseConfig) -> None:
        """Config setter.

        :param value: value to be set
        :type value: ConceptDriftBaseConfig
        :raises TypeError: Type error exception
        """
        if not isinstance(value, ConceptDriftBaseConfig):
            raise TypeError("value must be of type ConceptDriftBaseConfig.")
        self._config = value

    @property
    def num_instances(self) -> int:
        """Number of instances counter property.

        :return: Number of instances counter value
        :rtype: int
        """
        return self._num_instances

    @num_instances.setter
    def num_instances(self, value: int) -> None:
        """Number of instances counter setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_instances must be greater or equal than 0.")
        self._num_instances = value

    def reset(self, *args, **kwargs) -> None:
        """Reset method."""

    @property
    def status(self) -> Dict[str, bool]:
        """Status property.

        :return: status dict
        :rtype: Dict[str, bool]
        """

    @abc.abstractmethod
    def update(self, value: Union[int, float]) -> None:
        """Abstract update method.

        :param value: value to update detector
        :type value: Union[int, float]
        """
