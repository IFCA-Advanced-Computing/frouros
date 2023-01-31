"""Test callback module."""

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.callbacks import ResetOnDataDrift
from frouros.data_drift.batch.base import DataDriftBatchBase
from frouros.data_drift.batch import KSTest


@pytest.mark.parametrize(
    "detector",
    [KSTest()],
)
def test_batch_reset_on_data_drift(
    univariate_distribution_p,
    univariate_distribution_q,
    detector: DataDriftBatchBase,
    mocker,
) -> None:
    """Test batch distance based univariate method.

    :param detector: detector distance
    :type detector: DataDriftBatchBase
    """
    mocker.patch("frouros.data_drift.batch.base.DataDriftBatchBase.reset")

    np.random.seed(seed=31)
    X_ref = np.random.normal(*univariate_distribution_p, size=500)  # noqa: N806
    X_test = np.random.normal(*univariate_distribution_q, size=500)  # noqa: N806

    detector = KSTest(callbacks=[ResetOnDataDrift(alpha=0.01)])

    detector.fit(X=X_ref)
    _ = detector.compare(X=X_test)
    detector.reset.assert_called_once()  # type: ignore # pylint: disable=no-member
