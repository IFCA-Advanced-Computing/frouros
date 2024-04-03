"""Decorators module."""

import functools
import os
from typing import Any

import pytest


def set_os_filename(base_filename: str) -> Any:
    """Set OS filename.

    :param base_filename: Base filename
    :type base_filename: str
    :return: Decorator
    :rtype: Any
    """

    def decorator(func: Any) -> Any:
        if os.name == "nt":  # Windows
            temp_dir = os.environ.get("TEMP") or os.environ.get("TMP")
            filename = f"{temp_dir}\\{base_filename}"
        elif os.name == "posix":  # Linux or macOS
            temp_dir = "/tmp"
            filename = f"{temp_dir}/{base_filename}"
        else:
            raise Exception("Unsupported operating system.")

        @functools.wraps(func)
        @pytest.mark.filename(filename)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func(*args, **kwargs)

        return wrapper

    return decorator
