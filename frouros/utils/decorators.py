"""Decorators module."""

from functools import wraps
from inspect import signature
from typing import Any

from frouros.common.exceptions import ArgumentError


def check_func_parameters(func: Any) -> Any:
    """Decorator function to check function parameters.

    :param func: function to be wrapped
    :type func: Any
    :return: wrapper
    :rtype: Any
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_parameters = signature(args[1]).parameters
        if "y_true" not in func_parameters or "y_pred" not in func_parameters:
            raise ArgumentError("value function must have y_true and y_pred arguments.")
        func(*args, **kwargs)

    return wrapper
