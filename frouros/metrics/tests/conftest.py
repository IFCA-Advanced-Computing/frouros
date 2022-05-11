"""Configuration file for the tests."""

import pytest  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore

from frouros.metrics.prequential_error.prequential_error import PrequentialError


@pytest.fixture(scope="module")
def prequential_error():
    """Prequential error.

    :return: prequential error
    :rtype: PrequentialError
    """
    return PrequentialError(
        error_scorer=lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred)
    )
