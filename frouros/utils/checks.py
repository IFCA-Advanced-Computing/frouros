"""Checks module."""

from typing import Any

from frouros.callbacks.base import BaseCallback


def check_callbacks(callbacks: Any, expected_cls: BaseCallback) -> None:
    """Check callbacks.

    :param callbacks: callbacks
    :type callbacks: Any
    :param expected_cls: expected callback class
    :type expected_cls: BaseCallback
    :raises TypeError: Type error exception
    """
    if not (
        callbacks is None
        or isinstance(callbacks, expected_cls)  # type: ignore
        or (
            isinstance(callbacks, list)
            and all(
                isinstance(item, expected_cls) for item in callbacks  # type: ignore
            )
        )
    ):
        raise TypeError(
            f"callbacks must be of type None, "
            f"{expected_cls.name} or a list of {expected_cls.name}."
        )
