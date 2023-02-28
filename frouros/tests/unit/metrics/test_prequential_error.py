"""Test prequential error module."""

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.metrics import PrequentialError


def error_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error scorer function.

    :param y_true: ground-truth values
    :type: numpy.ndarray
    :param y_pred: predicted values
    :type: numpy.ndarray
    :return error value
    :rtype float
    """
    return 1 - int(y_true == y_pred)


@pytest.mark.parametrize(
    "prequential_error, expected_performance",
    [
        (PrequentialError(alpha=1.0), 0.5),
        (
            PrequentialError(
                alpha=0.9999,
            ),
            0.50000833,
        ),
    ],
)
def test_prequential_error(
    prequential_error: PrequentialError, expected_performance: float
) -> None:
    """Test prequential error.

    :param prequential_error: prequential error metric
    :type prequential_error: PrequentialError
    :param expected_performance: expected performance value
    :type expected_performance: float
    """
    y_true = [True, True, False, True, False, True]
    y_pred = [True, False, False, False, True, True]
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        error_value = error_scorer(
            y_true=np.array([y_true_sample]),
            y_pred=np.array([y_pred_sample]),
        )
        performance = prequential_error(error_value=error_value)

    assert performance == pytest.approx(expected_performance)
