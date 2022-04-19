"""Supervised util functions."""

from typing import Tuple

from sklearn.base import BaseEstimator  # type: ignore

from frouros.unsupervised.statistical_test.exceptions import (  # type: ignore
    GetStatisticalTestError,
)


def get_statistical_test(
    estimator: BaseEstimator, detector_name: str = "detector"
) -> Tuple[float, float]:
    """Get detector statistical test value.

    :param estimator: estimator containing the detector
    :type estimator: sklearn.base.BaseEstimator
    :param detector_name: detector´s name
    :type detector_name: str
    :raises GetStatisticalTestError: get statistical test error exception
    :return detector statistical test value
    :rtype: Tuple[Any, ...]
    """
    try:
        test = estimator[detector_name].test
    except (AttributeError, KeyError) as e:
        raise GetStatisticalTestError(e) from e
    return test
