"""Test real datasets module."""

from typing import Callable

import numpy as np  # type: ignore
import pytest  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore

from frouros.metrics import PrequentialError, PrequentialErrorFadingFactor
from frouros.metrics.prequential_error import PrequentialErrorBase


def error_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> Callable:
    """Error scorer function.

    :param y_true: ground-truth values
    :type: numpy.ndarray
    :param y_pred: predicted values
    :type: numpy.ndarray
    :return error scorer function
    :rtype Callable
    """
    return 1 - accuracy_score(y_true, y_pred)


@pytest.mark.parametrize(
    "prequential_error, expected_performance",
    [
        (PrequentialError(error_scorer=error_scorer), 0.5),
        (
            PrequentialErrorFadingFactor(error_scorer=error_scorer, alpha=0.9999),
            0.50000833,
        ),
    ],
)
def test_prequential_error(
    prequential_error: PrequentialErrorBase, expected_performance: float
) -> None:
    """Test prequential error.

    :param prequential_error: prequential error metric
    :type prequential_error: PrequentialErrorBase
    :param expected_performance: expected performance value
    :type expected_performance: float
    """
    y_true = [True, True, False, True, False, True]
    y_pred = [True, False, False, False, True, True]
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        performance = prequential_error(
            y_true=np.array([y_true_sample]), y_pred=np.array([y_pred_sample])
        )

    assert performance == pytest.approx(expected_performance)
