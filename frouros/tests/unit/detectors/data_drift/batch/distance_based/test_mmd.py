"""Test MMD."""

from functools import partial
from typing import Optional, Tuple

import numpy as np
import pytest

from frouros.detectors.data_drift import MMD  # type: ignore
from frouros.utils.kernels import rbf_kernel


@pytest.mark.parametrize(
    "distribution_p, distribution_q, expected_distance",
    [
        ((0, 1, 100), (0, 1, 100), 0.00052755),  # (mean, std, size)
        ((0, 1, 100), (0, 1, 10), -0.03200193),
        ((0, 1, 10), (0, 1, 100), 0.07154671),
        ((2, 1, 100), (0, 1, 100), 0.43377622),
        ((2, 1, 100), (0, 1, 10), 0.23051378),
        ((2, 1, 10), (0, 1, 100), 0.62530767),
    ],
)
def test_mmd_batch_univariate(
    distribution_p: Tuple[float, float, int],
    distribution_q: Tuple[float, float, int],
    expected_distance: float,
) -> None:
    """Test MMD batch with univariate data.

    :param distribution_p: mean, std and size of samples from distribution p
    :type distribution_p: Tuple[float, float, int]
    :param distribution_q: mean, std and size of samples from distribution q
    :type distribution_q: Tuple[float, float, int]
    :param expected_distance: expected distance value
    :type expected_distance: float
    """
    np.random.seed(seed=31)
    X_ref = np.random.normal(*distribution_p)  # noqa: N806
    X_test = np.random.normal(*distribution_q)  # noqa: N806

    detector = MMD(
        kernel=partial(rbf_kernel, sigma=0.5),
    )
    _ = detector.fit(X=X_ref)

    result = detector.compare(X=X_test)[0]

    assert np.isclose(result.distance, expected_distance)


@pytest.mark.parametrize(
    "distribution_p, distribution_q, chunk_size",
    [
        ((0, 1, 100), (0, 1, 100), None),  # (mean, std, size)
        ((0, 1, 100), (0, 1, 100), 2),
        ((0, 1, 100), (0, 1, 100), 10),
        ((0, 1, 100), (0, 1, 10), None),
        ((0, 1, 100), (0, 1, 10), 2),
        ((0, 1, 100), (0, 1, 10), 10),
        ((0, 1, 10), (0, 1, 100), None),
        ((0, 1, 10), (0, 1, 100), 2),
        ((0, 1, 10), (0, 1, 100), 10),
    ],
)
def test_mmd_batch_precomputed_expected_k_xx(
    distribution_p: Tuple[float, float, int],
    distribution_q: Tuple[float, float, int],
    chunk_size: Optional[int],
) -> None:
    """Test MMD batch with precomputed expected k_xx.

    :param distribution_p: mean, std and size of samples from distribution p
    :type distribution_p: Tuple[float, float, int]
    :param distribution_q: mean, std and size of samples from distribution q
    :type distribution_q: Tuple[float, float, int]
    :param chunk_size: chunk size
    :type chunk_size: Optional[int]
    """
    np.random.seed(seed=31)
    X_ref = np.random.normal(*distribution_p)  # noqa: N806
    X_test = np.random.normal(*distribution_q)  # noqa: N806

    kernel = partial(rbf_kernel, sigma=0.5)

    detector = MMD(
        kernel=kernel,
        chunk_size=chunk_size,
    )
    _ = detector.fit(X=X_ref)

    # Computes mmd using precomputed expected k_xx
    precomputed_distance = detector.compare(X=X_test)[0].distance

    # Computes mmd from scratch
    scratch_distance = MMD._mmd(  # pylint: disable=protected-access
        X=X_ref,
        Y=X_test,
        kernel=kernel,
        chunk_size=chunk_size,
    )

    assert np.isclose(precomputed_distance, scratch_distance)
