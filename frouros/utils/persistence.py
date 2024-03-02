"""Persistence module."""

import pickle

from frouros.callbacks.base import BaseCallback
from frouros.detectors.base import BaseDetector
from frouros.utils.logger import logger

DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL


def load(
    filename: str,
) -> object:
    """Load object from file.

    :param filename: Filename
    :type filename: str
    :return: Loaded object
    :rtype: object
    """
    try:
        with open(filename, "rb") as file:
            obj = pickle.load(
                file,
            )
        return obj
    except (IOError, pickle.UnpicklingError) as e:
        logger.error("Error occurred while loading object: %s", e)
        raise e


def save(
    obj: object,
    filename: str,
    pickle_protocol: int = DEFAULT_PROTOCOL,
) -> None:
    """Save object to file.

    :param obj: Object to save
    :type obj: object
    :param filename: Filename
    :type filename: str
    :param pickle_protocol: Pickle protocol, defaults to DEFAULT_PROTOCOL
    :type pickle_protocol: int, optional
    """
    try:
        if not isinstance(obj, (BaseDetector, BaseCallback)):
            raise TypeError(
                f"Object of type {type(obj)} is not serializable. "
                f"Must be an instance that inherits from BaseDetector or BaseCallback."
            )
        if pickle_protocol not in range(pickle.HIGHEST_PROTOCOL + 1):
            raise ValueError(
                f"Invalid pickle_protocol value. "
                f"Must be in range 0..{pickle.HIGHEST_PROTOCOL}."
            )
        with open(filename, "wb") as file:
            pickle.dump(
                obj,
                file,
                protocol=pickle_protocol,
            )
    except (IOError, pickle.PicklingError) as e:
        logger.error("Error occurred while saving object: %s", e)
        raise e
