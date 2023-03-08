"""Concept drift base module."""

import abc
from typing import Any, Dict, List, Optional, Union

from frouros.callbacks import Callback, History
from frouros.detectors.base import DetectorBase


class ConceptDriftBaseConfig(abc.ABC):
    """Abstract class representing a concept drift configuration class."""

    def __init__(
        self,
        min_num_instances: int = 10,
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
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("value must be greater than 0.")
        self._min_num_instances = value

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}"
            f"({', '.join(f'{k[1:]}={v}' for k, v in self.__dict__.items())})"
        )


class ConceptDriftBase(DetectorBase):
    """Abstract class representing a concept drift streaming."""

    config_type = ConceptDriftBaseConfig

    def __init__(
        self,
        config: Optional[ConceptDriftBaseConfig] = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[ConceptDriftBaseConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(callbacks=callbacks)
        self.config = config  # type: ignore
        self.additional_vars = None

        self.num_instances = 0
        self.drift = False
        for callback in self.callbacks:  # type: ignore
            callback.set_detector(detector=self)

    def _set_additional_vars_callback(self) -> None:
        for callback in self.callbacks:  # type: ignore
            if isinstance(callback, History):
                # callback.set_detector(detector=self)
                callback.add_additional_vars(
                    vars_=self.additional_vars.keys(),  # type: ignore
                )

    @property
    def additional_vars(self) -> Optional[Dict[str, Any]]:
        """Additional variables property.

        :return: additional variables
        :rtype: Optional[Dict[str, Any]]
        """
        return self._additional_vars

    @additional_vars.setter
    def additional_vars(self, value: Optional[Dict[str, Any]]) -> None:
        """Additional variables setter.

        :param value: value to be set
        :type value: Optional[Dict[str, Any]]
        """
        self._additional_vars = value if value is not None else {}

    @property
    def config(self) -> ConceptDriftBaseConfig:
        """Config property.

        :return: configuration parameters of the estimator
        :rtype: ConceptDriftBaseConfig
        """
        return self._config

    @config.setter
    def config(self, value: Optional[ConceptDriftBaseConfig]) -> None:
        """Config setter.

        :param value: value to be set
        :type value: ConceptDriftBaseConfig
        :raises TypeError: Type error exception
        """
        if value is not None:
            if not isinstance(value, self.config_type):
                raise TypeError(
                    f"value must be of type " f"{self.config_type.__class__.__name__}."
                )
            self._config = value
        else:
            self._config = self.config_type()

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

    def reset(self) -> None:
        """Reset method."""
        self.num_instances = 0
        self.drift = False
        for callback in self.callbacks:  # type: ignore
            callback.reset()

    @property
    def status(self) -> Dict[str, bool]:
        """Status property.

        :return: status dict
        :rtype: Dict[str, bool]
        """
        return {"drift": self.drift}

    def update(self, value: Union[int, float], **kwargs) -> Dict[str, Any]:
        """Update method.

        :param value: value to update detector
        :type value: Union[int, float]
        """
        for callback in self.callbacks:  # type: ignore
            callback.on_update_start(value=value, **kwargs)  # type: ignore
        self._update(value=value, **kwargs)
        for callback in self.callbacks:  # type: ignore
            callback.on_update_end(value=value, **kwargs)  # type: ignore

        callbacks_logs = self._get_callbacks_logs()
        return callbacks_logs

    def _get_callbacks_logs(self) -> Dict[str, Any]:
        logs = {
            callback.name: callback.logs for callback in self.callbacks  # type: ignore
        }
        return logs

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return f"{super().__repr__()[:-1]}, config={self.config})"
