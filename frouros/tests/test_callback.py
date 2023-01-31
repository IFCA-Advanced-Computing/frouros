"""Test callback module."""

import pytest  # type: ignore

from frouros.callbacks import ResetOnBatchDataDrift
from frouros.data_drift.batch.base import DataDriftBatchBase
from frouros.data_drift.batch import CVMTest, KSTest, WelchTTest


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
    :param mocker:
    :type mocker:
    """
    mocker.patch("frouros.data_drift.batch.base.DataDriftBatchBase.reset")

    detector = detector(callbacks=[ResetOnBatchDataDrift(alpha=0.01)])  # type: ignore
    detector.fit(X=X_ref_univariate)
    _ = detector.compare(X=X_test_univariate)
    detector.reset.assert_called_once()  # type: ignore # pylint: disable=no-member
