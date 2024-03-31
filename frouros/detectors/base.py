"""Base detector module."""

import abc
from typing import Any, Optional, Union

import numpy as np

from frouros.callbacks.base import BaseCallback


class BaseDetector(abc.ABC):
    """Abstract class representing a detector."""

    def __init__(
        self,
        callbacks: Optional[Union[BaseCallback, list[BaseCallback]]] = None,
    ) -> None:
        """Init method.

        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallback, list[BaseCallback]]]
        """
        self.callbacks = callbacks  # type: ignore

    @property
    def callbacks(self) -> Optional[list[BaseCallback]]:
        """Callbacks property.

        :return: callbacks
        :rtype: Optional[list[BaseCallback]]
        """
        return self._callbacks

    @callbacks.setter
    def callbacks(
        self,
        value: Optional[Union[BaseCallback, list[BaseCallback]]],
    ) -> None:
        """Callbacks setter.

        :param value: value to be set
        :type value: Optional[Union[BaseCallback, list[Callback]]]
        :raises TypeError: Type error exception
        """
        if value is not None:
            if isinstance(value, BaseCallback):
                self._callbacks = [value]
            elif not all(isinstance(callback, BaseCallback) for callback in value):
                raise TypeError("value must be of type None or a list of BaseCallback.")
            else:
                self._callbacks = value
        else:
            self._callbacks = []

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""

    def _get_callbacks_logs(self) -> dict[str, Any]:
        logs = {
            callback.name: callback.logs
            for callback in self.callbacks  # type: ignore
        }
        return logs

    @staticmethod
    def _check_array(X: Any) -> None:  # noqa: N803
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}"
            f"(callbacks=["
            f"{', '.join([*map(str, self.callbacks)])}])"  # type: ignore
        )
