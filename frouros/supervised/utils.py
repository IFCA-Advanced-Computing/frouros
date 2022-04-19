"""Supervised util functions."""

from typing import Any, Tuple

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore

from frouros.supervised.exceptions import UpdateDetectorError


def update_detector(
    estimator: BaseEstimator, y: np.ndarray, detector_name: str = "detector"
) -> Tuple[Any, ...]:
    """Update detector with new target sample/s.

    :param estimator: estimator containing the detector
    :type estimator: sklearn.base.BaseEstimator
    :param y: target samples
    :type y: np.ndarray
    :param detector_name: detector´s name
    :type detector_name: str
    :raises UpdateDetectorError: update detector error exception
    :return detector update output
    :rtype: Tuple[Any, ...]
    """
    try:
        output = estimator[detector_name].update(y=np.array([y]))
    except KeyError as e:
        raise UpdateDetectorError(e) from e
    return output
