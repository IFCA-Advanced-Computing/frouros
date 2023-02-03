"""Test callback module."""

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.callbacks import PermutationTestOnBatchData, ResetOnBatchDataDrift
from frouros.data_drift.batch.base import DataDriftBatchBase
from frouros.data_drift.batch import (
    BhattacharyyaDistance,
    CVMTest,
    EMD,
    HellingerDistance,
    HistogramIntersection,
    JS,
    KL,
    KSTest,
    MMD,
    PSI,
    WelchTTest,
)


@pytest.mark.parametrize(
    "detector, expected_distance, expected_p_value",
    [
        (BhattacharyyaDistance, 0.55516059, 0.0),
        (EMD, 3.85346006, 0.0),
        (HellingerDistance, 0.74509099, 0.0),
        (HistogramIntersection, 0.78, 0.0),
        (JS, 0.67010107, 0.0),
        (KL, np.inf, 0.0),
        (MMD, 0.71529206, 0.0),
        (PSI, 461.20379435, 0.0),
    ],
)
def test_batch_permutation_test_data_univariate_different_distribution(
    X_ref_univariate: np.ndarray,  # noqa: N803
    X_test_univariate: np.ndarray,
    detector: DataDriftBatchBase,
    expected_distance: float,
    expected_p_value: float,
) -> None:
    """Test batch permutation test on data callback.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param detector: detector distance
    :type detector: DataDriftBatchBase
    :param expected_distance: expected distance value
    :type expected_distance: float
    :param expected_p_value: expected p-value value
    :type expected_p_value: float
    """
    np.random.seed(seed=31)

    permutation_test_name = "permutation_test"
    detector = detector(  # type: ignore
        callbacks=[
            PermutationTestOnBatchData(
                num_permutations=100,
                random_state=31,
                num_jobs=-1,
                name=permutation_test_name,
            )
        ]
    )
    _ = detector.fit(X=X_ref_univariate)
    distance, callback_logs = detector.compare(X=X_test_univariate)

    assert np.isclose(distance, expected_distance)
    assert np.isclose(callback_logs[permutation_test_name]["p_value"], expected_p_value)


@pytest.mark.parametrize(
    "detector",
    [CVMTest, KSTest, WelchTTest],
)
def test_batch_reset_on_data_drift(
    X_ref_univariate,  # noqa: N803
    X_test_univariate,
    detector: DataDriftBatchBase,
    mocker,
) -> None:
    """Test batch reset on data drift callback.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param detector: detector distance
    :type detector: DataDriftBatchBase
    """
    mocker.patch("frouros.data_drift.batch.base.DataDriftBatchBase.reset")

    detector = detector(callbacks=[ResetOnBatchDataDrift(alpha=0.01)])  # type: ignore
    _ = detector.fit(X=X_ref_univariate)
    _, _ = detector.compare(X=X_test_univariate)
    detector.reset.assert_called_once()  # type: ignore # pylint: disable=no-member
