"""Common update module."""

from typing import Any, Optional, Tuple

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore

from frouros.common.exceptions import UpdateDetectorError


def update_detector(
    estimator: BaseEstimator,
    y: np.ndarray,
    X: Optional[np.ndarray] = None,  # noqa: N803
    detector_name: str = "detector",
) -> Tuple[Any, ...]:
    """Update detector with new sample/s.

    :param estimator: estimator containing the detector
    :type estimator: sklearn.base.BaseEstimator
    :param y: target samples
    :type y: numpy.ndarray
    :param X: feature samples
    :type X: Optional[numpy.ndarray]
    :param detector_name: detector´s name
    :type detector_name: str
    :raises UpdateDetectorError: update detector error exception
    :return: detector update output
    :rtype: Tuple[Any, ...]
    """
    try:
        output = estimator[detector_name].update(X=X, y=y)
    except KeyError as e:
        raise UpdateDetectorError(e) from e
    return output
