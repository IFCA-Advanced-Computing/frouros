"""Test real datasets module."""

import numpy as np  # type: ignore

from frouros.metrics import PrequentialError


def test_prequential_error(prequential_error: PrequentialError) -> None:
    """Test prequential error.

    :param prequential_error: prequential error metric
    :type prequential_error: PrequentialError
    """
    y_true = [True, True, False, True, False, True]
    y_pred = [True, False, False, False, True, True]

    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        performance = prequential_error(
            y_true=np.array([y_true_sample]), y_pred=np.array([y_pred_sample])
        )

    assert performance == 0.5
