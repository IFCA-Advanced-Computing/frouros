"""Test data drift detectors."""

from typing import Tuple, Union

import pytest  # type: ignore
import numpy as np  # type: ignore

from frouros.detectors.data_drift.batch.base import DataDriftBatchBase
from frouros.detectors.data_drift.batch import (
    BhattacharyyaDistance,
    EMD,
    HellingerDistance,
    HINormalizedComplement,
    PSI,
    JS,
    KL,
    MMD,
)
from frouros.detectors.data_drift.batch import (
    ChiSquareTest,
    CVMTest,
    KSTest,
    WelchTTest,
)
from frouros.detectors.data_drift.streaming import IncrementalKSTest


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

    _ = detector.fit(X=X_ref)
    (statistic, p_value), _ = detector.compare(X=X_test)

    assert np.isclose(statistic, expected_statistic)
    assert np.isclose(p_value, expected_p_value)


@pytest.mark.parametrize(
    "detector, expected_distance",
    [
        (EMD(), 3.85346006),
        (JS(), 0.67010107),
        (KL(), np.inf),
        (HINormalizedComplement(), 0.78),
    ],
)
def test_batch_distance_based_univariate(
    X_ref_univariate: np.ndarray,  # noqa: N803
    X_test_univariate: np.ndarray,  # noqa: N803
    detector: DataDriftBatchBase,
    expected_distance: float,
) -> None:
    """Test batch distance based univariate method.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param detector: detector distance
    :type detector: DataDriftBatchBase
    :param expected_distance: expected p-value value
    :type expected_distance: float
    """
    _ = detector.fit(X=X_ref_univariate)
    distance, _ = detector.compare(X=X_test_univariate)

    assert np.isclose(distance, expected_distance)


@pytest.mark.parametrize(
    "detector, expected_distance",
    [
        (PSI(), 461.20379435),
        (HellingerDistance(), 0.74509099),
        (BhattacharyyaDistance(), 0.55516059),
    ],
)
def test_batch_distance_bins_based_univariate_different_distribution(
    X_ref_univariate: np.ndarray,  # noqa: N803
    X_test_univariate: np.ndarray,  # noqa: N803
    detector: DataDriftBatchBase,
    expected_distance: float,
) -> None:
    """Test batch distance bins based univariate different distribution method.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param detector: detector distance
    :type detector: DataDriftBatchBase
    :param expected_distance: expected p-value value
    :type expected_distance: float
    """
    _ = detector.fit(X=X_ref_univariate)
    distance, _ = detector.compare(X=X_test_univariate)

    assert np.isclose(distance, expected_distance)


@pytest.mark.parametrize(
    "detector, expected_distance",
    [
        (PSI(), 0.01840072),
        (HellingerDistance(), 0.04792538),
        (BhattacharyyaDistance(), 0.00229684),
    ],
)
def test_batch_distance_bins_based_univariate_same_distribution(
    univariate_distribution_p: Tuple[float, float],
    detector: DataDriftBatchBase,
    expected_distance: float,
    num_samples: int = 500,
) -> None:
    """Test batch distance based univariate same distribution method.

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

    _ = detector.fit(X=X_ref)
    distance, _ = detector.compare(X=X_test)

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

    _ = detector.fit(X=X_ref[:, 0])
    (statistic, p_value), _ = detector.compare(X=X_test[:, 0])

    assert np.isclose(statistic, expected_statistic)
    assert np.isclose(p_value, expected_p_value)


@pytest.mark.parametrize("detector, expected_distance", [(MMD(), 0.12183835)])
def test_batch_distance_based_multivariate_different_distribution(
    X_ref_multivariate: np.ndarray,  # noqa: N803
    X_test_multivariate: np.ndarray,  # noqa: N803
    detector: DataDriftBatchBase,
    expected_distance: float,
) -> None:
    """Test batch distance based multivariate different distribution method.

    :param X_ref_multivariate: reference multivariate data
    :type X_ref_multivariate: numpy.ndarray
    :param X_test_multivariate: test multivariate data
    :type X_test_multivariate: numpy.ndarray
    :param detector: detector test
    :type detector: DataDriftBatchBase
    :param expected_distance: expected distance value
    :type expected_distance: float
    """
    _ = detector.fit(X=X_ref_multivariate)
    statistic, _ = detector.compare(X=X_test_multivariate)

    assert np.isclose(statistic, expected_distance)


@pytest.mark.parametrize("detector, expected_distance", [(MMD(), 0.03590599)])
def test_batch_distance_based_multivariate_same_distribution(
    multivariate_distribution_p: Tuple[np.ndarray, np.ndarray],
    detector: DataDriftBatchBase,
    expected_distance: float,
    num_samples: int = 100,
) -> None:
    """Test batch distance based multivariate same distribution method.

    :param multivariate_distribution_p: mean and covariance matrix of distribution p
    :type multivariate_distribution_p: Tuple[numpy.ndarray, numpy.ndarray]
    :param detector: detector test
    :type detector: DataDriftBatchBase
    :param num_samples: number of random samples
    :type num_samples: int
    :param expected_distance: expected distance value
    :type expected_distance: float
    """
    np.random.seed(seed=31)
    X_ref = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )
    X_test = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )

    _ = detector.fit(X=X_ref)
    statistic, _ = detector.compare(X=X_test)

    assert np.isclose(statistic, expected_distance)


@pytest.mark.parametrize(
    "detector, expected_distance",
    [(MMD(chunk_size=10), 0.12183835), (MMD(chunk_size=None), 0.12183835)],
)
def test_batch_distance_based_chunk_size_valid(
    X_ref_multivariate: np.ndarray,  # noqa: N803
    X_test_multivariate: np.ndarray,  # noqa: N803
    detector: DataDriftBatchBase,
    expected_distance: float,
) -> None:
    """Test batch distance based chunk size valid method.

    :param X_ref_multivariate: reference multivariate data
    :type X_ref_multivariate: numpy.ndarray
    :param X_test_multivariate: test multivariate data
    :type X_test_multivariate: numpy.ndarray
    :param detector: detector test
    :type detector: DataDriftBatchBase
    :param expected_distance: expected distance value
    :type expected_distance: float
    """
    _ = detector.fit(X=X_ref_multivariate)
    statistic, _ = detector.compare(X=X_test_multivariate)

    assert np.isclose(statistic, expected_distance)


@pytest.mark.parametrize(
    "chunk_size, expected_exception",
    [
        (1.5, TypeError),
        ("10", TypeError),
        (-1, ValueError),
    ],
)
def test_batch_distance_based_chunk_size_invalid(
    chunk_size: Union[int, float, str],
    expected_exception: Union[TypeError, ValueError],
) -> None:
    """Test batch distance based chunk size invalid method.

    :param chunk_size: chunk size
    :type chunk_size: Union[int, float, str]
    :param expected_exception: expected exception
    :type expected_exception: Union[TypeError, ValueError]
    """
    with pytest.raises(expected_exception):
        _ = MMD(chunk_size=chunk_size)  # type: ignore


@pytest.mark.parametrize(
    "detector, expected_statistic, expected_p_value",
    [
        (IncrementalKSTest(), 0.27, 0.46046910),
    ],
)
def test_streaming_statistical_univariate_same_distribution(
    univariate_distribution_p: Tuple[float, float],
    detector: DataDriftBatchBase,
    expected_statistic: float,
    expected_p_value: float,
) -> None:
    """Test streaming statistical test univariate same distribution method.

    :param univariate_distribution_p: mean and standard deviation of distribution p
    :type univariate_distribution_p: Tuple[float, float]
    :param detector: detector statistical test
    :type detector: DataDriftStreamingBase
    :param expected_statistic: expected statistic value
    :type expected_statistic: float
    :param expected_p_value: expected p-value
    :type expected_p_value: float
    """
    np.random.seed(seed=31)
    X_ref = np.random.normal(*univariate_distribution_p, size=100)  # noqa: N806
    X_test = np.random.normal(*univariate_distribution_p, size=100)  # noqa: N806

    _ = detector.fit(X=X_ref)

    for value in X_test:
        test = detector.update(value=value)  # type: ignore

    # Check last statistic and p-value
    assert np.isclose(test.statistic, expected_statistic)
    assert np.isclose(test.p_value, expected_p_value)


@pytest.mark.parametrize(
    "detector, expected_statistic, expected_p_value",
    [
        (IncrementalKSTest(), 1.0, 0.0),
    ],
)
def test_streaming_statistical_univariate_different_distribution(
    univariate_distribution_p: Tuple[float, float],
    univariate_distribution_q: Tuple[float, float],
    detector: DataDriftBatchBase,
    expected_statistic: float,
    expected_p_value: float,
) -> None:
    """Test streaming statistical test univariate different distribution method.

    :param univariate_distribution_p: mean and standard deviation of distribution p
    :type univariate_distribution_p: Tuple[float, float]
    :param univariate_distribution_q: mean and standard deviation of distribution q
    :type univariate_distribution_q: Tuple[float, float]
    :param detector: detector statistical test
    :type detector: DataDriftStreamingBase
    :param expected_statistic: expected statistic value
    :type expected_statistic: float
    :param expected_p_value: expected p-value
    :type expected_p_value: float
    """
    np.random.seed(seed=31)
    X_ref = np.random.normal(*univariate_distribution_p, size=100)  # noqa: N806
    X_test = np.random.normal(*univariate_distribution_q, size=100)  # noqa: N806

    _ = detector.fit(X=X_ref)

    for value in X_test:
        test = detector.update(value=value)  # type: ignore

    # Check last statistic and p-value
    assert np.isclose(test.statistic, expected_statistic)
    assert np.isclose(test.p_value, expected_p_value)
