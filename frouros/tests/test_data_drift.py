"""Test data drift detectors."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore

from frouros.data_drift.batch.base import DataDriftBatchBase
from frouros.data_drift.batch.distance_based import (
    Bhattacharyya,
    EMD,
    Hellinger,
    HistogramIntersection,
    PSI,
    JS,
    KL,
    MMD,
)
from frouros.data_drift.batch.statistical_test import (
    ChiSquareTest,
    CVMTest,
    KSTest,
    WelchTTest,
)


@pytest.mark.parametrize(
    "detector, expected_statistic, expected_p_value",
    [(ChiSquareTest(), 6.13333333, 0.04657615)],
)
def test_batch_distance_based_categorical(
    categorical_dataset: Tuple[np.ndarray, np.ndarray],
    detector: ChiSquareTest,
    expected_statistic: float,
    expected_p_value: float,
) -> None:
    """Test batch categorical features method.

    :param categorical_dataset: categorical dataset
    :type categorical_dataset: Tuple[numpy.ndarray, numpy.ndarray]
    :param detector: detector test
    :type detector: ChiSquaredTest
    :param expected_statistic: expected statistic value
    :type expected_statistic: float
    :param expected_p_value: expected p-value value
    :type expected_p_value: float
    """
    X_ref, X_test = categorical_dataset  # noqa: N806

    detector.fit(X=X_ref)
    statistic, p_value = detector.compare(X=X_test)

    assert np.isclose(statistic, expected_statistic)
    assert np.isclose(p_value, expected_p_value)


@pytest.mark.parametrize(
    "detector, expected_distance",
    [
        (EMD(), 0.54726161),
        (JS(), 0.81451218),
        (KL(), np.inf),
        (HistogramIntersection(), 0.97669491),
    ],
)
def test_batch_distance_based_univariate(
    elec2_dataset,
    detector: DataDriftBatchBase,
    expected_distance: float,
) -> None:
    """Test batch distance based univariate method.

    :param elec2_dataset: Elec2 raw dataset
    :type elec2_dataset: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :param detector: detector distance
    :type detector: DataDriftBatchBase
    :param expected_distance: expected p-value value
    :type expected_distance: float
    """
    X_ref, _, X_test = elec2_dataset  # noqa: N806

    detector.fit(X=X_ref[:, 0])
    distance = detector.compare(X=X_test[:, 0])

    assert np.isclose(distance, expected_distance)


@pytest.mark.parametrize(
    "detector, expected_distance",
    [(PSI(), 468.79410784), (Hellinger(), 0.77137797), (Bhattacharyya(), 0.59502397)],
)
def test_batch_distance_bins_based_univariate_different_distribution(
    univariate_distribution_p: Tuple[float, float],
    univariate_distribution_q: Tuple[float, float],
    detector: DataDriftBatchBase,
    expected_distance: float,
    num_samples: int = 500,
) -> None:
    """Test distance based univariate different distribution method.

    :param univariate_distribution_p: mean and standard deviation of distribution p
    :type univariate_distribution_p: Tuple[float, float]
    :param univariate_distribution_q: mean and standard deviation of distribution q
    :type univariate_distribution_q: Tuple[float, float]
    :param detector: detector distance
    :type detector: DataDriftBatchBase
    :param expected_distance: expected p-value value
    :type expected_distance: float
    """
    np.random.seed(seed=31)
    X_ref = np.random.normal(*univariate_distribution_p, size=num_samples)  # noqa: N806
    X_test = np.random.normal(  # noqa: N806
        *univariate_distribution_q, size=num_samples
    )

    detector.fit(X=X_ref)
    distance = detector.compare(X=X_test)

    assert np.isclose(distance, expected_distance)


@pytest.mark.parametrize(
    "detector, expected_distance",
    [(PSI(), 0.01840072), (Hellinger(), 0.04792538), (Bhattacharyya(), 0.00229684)],
)
def test_batch_distance_bins_based_univariate_same_distribution(
    univariate_distribution_p: Tuple[float, float],
    detector: DataDriftBatchBase,
    expected_distance: float,
    num_samples: int = 500,
) -> None:
    """Test distance based univariate same distribution method.

    :param univariate_distribution_p: mean and standard deviation of distribution p
    :type univariate_distribution_p: Tuple[float, float]
    :param detector: detector distance
    :type detector: DataDriftBatchBase
    :param expected_distance: expected p-value value
    :type expected_distance: float
    """
    np.random.seed(seed=31)
    X_ref = np.random.normal(*univariate_distribution_p, size=num_samples)  # noqa: N806
    X_test = np.random.normal(  # noqa: N806
        *univariate_distribution_p, size=num_samples
    )

    detector.fit(X=X_ref)
    distance = detector.compare(X=X_test)

    assert np.isclose(distance, expected_distance)


@pytest.mark.parametrize(
    "detector, expected_statistic, expected_p_value",
    [
        (CVMTest(), 3776.09848103, 5.38105056e-07),
        (KSTest(), 0.99576271, 0.0),
        (WelchTTest(), -287.92032554, 0.0),
    ],
)
def test_batch_statistical_univariate(
    elec2_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray],
    detector: DataDriftBatchBase,
    expected_statistic: float,
    expected_p_value: float,
) -> None:
    """Test statistical univariate method.

    :param elec2_dataset: Elec2 raw dataset
    :type elec2_dataset: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :param detector: detector test
    :type detector: DataDriftBatchBase
    :param expected_statistic: expected statistic value
    :type expected_statistic: float
    :param expected_p_value: expected p-value value
    :type expected_p_value: float
    """
    X_ref, _, X_test = elec2_dataset  # noqa: N806

    detector.fit(X=X_ref[:, 0])
    statistic, p_value = detector.compare(X=X_test[:, 0])

    assert np.isclose(statistic, expected_statistic)
    assert np.isclose(p_value, expected_p_value)


@pytest.mark.parametrize("detector", [MMD()])
def test_batch_distance_based_multivariate_different_distribution(
    multivariate_distribution_p: Tuple[np.ndarray, np.ndarray],
    multivariate_distribution_q: Tuple[np.ndarray, np.ndarray],
    detector: DataDriftBatchBase,
    num_samples: int = 500,
) -> None:
    """Test distance based multivariate different distribution method.

    :param multivariate_distribution_p: mean and covariance matrix of distribution p
    :type multivariate_distribution_p: Tuple[numpy.ndarray, numpy.ndarray]
    :param multivariate_distribution_q: mean and covariance matrix of distribution q
    :type multivariate_distribution_q: Tuple[numpy.ndarray, numpy.ndarray]
    :param detector: detector test
    :type detector: DataDriftBatchBase
    :param num_samples: number of random samples
    :type num_samples: int
    """
    np.random.seed(seed=31)
    X_ref = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )
    X_test = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_q, size=num_samples
    )

    detector.fit(X=X_ref)
    statistic = detector.compare(X=X_test)

    assert np.isclose(statistic, 0.09446612)


@pytest.mark.parametrize("detector", [MMD()])
def test_batch_distance_based_multivariate_same_distribution(
    multivariate_distribution_p: Tuple[np.ndarray, np.ndarray],
    detector: DataDriftBatchBase,
    num_samples: int = 500,
) -> None:
    """Test distance based multivariate same distribution method.

    :param multivariate_distribution_p: mean and covariance matrix of distribution p
    :type multivariate_distribution_p: Tuple[numpy.ndarray, numpy.ndarray]
    :param detector: detector test
    :type detector: DataDriftBatchBase
    :param num_samples: number of random samples
    :type num_samples: int
    """
    np.random.seed(seed=31)
    X_ref = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )
    X_test = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )

    detector.fit(X=X_ref)
    statistic = detector.compare(X=X_test)

    assert np.isclose(statistic, 0.00256109)
