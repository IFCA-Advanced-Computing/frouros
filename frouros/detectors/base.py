"""Base detector module."""

import abc
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks.base import BaseCallback


class BaseDetector(abc.ABC):
    """Abstract class representing a detector."""

    def __init__(
        self,
        callbacks: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
    ) -> None:
        """Init method.

        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallback, List[Callback]]]
        """
        self.callbacks = callbacks  # type: ignore

    @property
    def callbacks(self) -> Optional[List[BaseCallback]]:
        """Callbacks property.

        :return: callbacks
        :rtype: Optional[List[BaseCallback]]
        """
        return self._callbacks  # type: ignore

    @callbacks.setter
    def callbacks(
        self,
        value: Optional[Union[BaseCallback, List[BaseCallback]]],
    ) -> None:
        """Callbacks setter.

        :param value: value to be set
        :type value: Optional[Union[BaseCallback, List[Callback]]]
        :raises TypeError: Type error exception
        """
        if value is not None:
            if isinstance(value, BaseCallback):
                self._callbacks = [value]
            elif not all(
                isinstance(callback, BaseCallback) for callback in value  # type: ignore
            ):
                raise TypeError("value must be of type None or a list of BaseCallback.")
            else:
                self._callbacks = value  # type: ignore
        else:
            self._callbacks = []

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""

    def _get_callbacks_logs(self) -> Dict[str, Any]:
        logs = {
            callback.name: callback.logs for callback in self.callbacks  # type: ignore
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
