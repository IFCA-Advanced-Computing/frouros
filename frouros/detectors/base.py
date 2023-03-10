"""Detector base module."""

import abc
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks import Callback


class DetectorBase(abc.ABC):
    """Abstract class representing a detector."""

    def __init__(
        self,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        self.callbacks = callbacks  # type: ignore

    @property
    def callbacks(self) -> Optional[List[Callback]]:
        """Callbacks property.

        :return: callbacks
        :rtype: Optional[List[Callback]]
        """
        return self._callbacks  # type: ignore

    @callbacks.setter
    def callbacks(self, value: Optional[Union[Callback, List[Callback]]]) -> None:
        """Callbacks setter.

        :param value: value to be set
        :type value: Optional[Union[Callback, List[Callback]]]
        :raises TypeError: Type error exception
        """
        if value is not None:
            if isinstance(value, Callback):
                self._callbacks = [value]
            elif not all(
                isinstance(callback, Callback) for callback in value  # type: ignore
            ):
                raise TypeError("value must be of type None or a list of Callback.")
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
            f"(callbacks=[{', '.join(self.callbacks)}])"  # type: ignore
        )
