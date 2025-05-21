"""Test MMD."""

from functools import partial
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
)

import numpy as np
import pytest

from frouros.detectors.data_drift import MMD  # type: ignore
from frouros.utils.kernels import rbf_kernel

RANDOM_SEED = 31
DEFAULT_SIGMA = 0.5


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
    np.random.seed(seed=RANDOM_SEED)
    X_ref = np.random.normal(*distribution_p)
    X_test = np.random.normal(*distribution_q)

    detector = MMD(
        kernel=partial(
            rbf_kernel,
            sigma=DEFAULT_SIGMA,
        ),
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
    np.random.seed(seed=RANDOM_SEED)
    X_ref = np.random.normal(*distribution_p)
    X_test = np.random.normal(*distribution_q)

    kernel = partial(
        rbf_kernel,
        sigma=DEFAULT_SIGMA,
    )

    detector = MMD(
        kernel=kernel,
        chunk_size=chunk_size,
    )
    _ = detector.fit(X=X_ref)

    # Computes mmd using precomputed expected k_xx
    precomputed_distance = detector.compare(X=X_test)[0].distance

    # Computes mmd from scratch
    scratch_distance = MMD._mmd(
        X=X_ref,
        Y=X_test,
        kernel=kernel,
        chunk_size=chunk_size,
    )

    assert np.isclose(precomputed_distance, scratch_distance)


@pytest.mark.parametrize(
    "distribution_p, distribution_q, chunk_size",
    [
        ((0, 1, size), (2, 1, size), chunk_size)
        for size in [10, 100]
        for chunk_size in list(range(1, 11))
    ],
)
def test_mmd_chunk_size_equivalence(
    distribution_p: Tuple[float, float, int],
    distribution_q: Tuple[float, float, int],
    chunk_size: int,
) -> None:
    """Test MMD with chunk_size=None vs specific chunk_size.

    :param distribution_p: mean, std and size of samples from distribution p
    :type distribution_p: Tuple[float, float, int]
    :param distribution_q: mean, std and size of samples from distribution q
    :type distribution_q: Tuple[float, float, int]
    :param chunk_size: specific chunk size to compare with None
    :type chunk_size: int
    """
    np.random.seed(seed=RANDOM_SEED)
    X_ref = np.random.normal(*distribution_p)
    X_test = np.random.normal(*distribution_q)

    kernel = partial(
        rbf_kernel,
        sigma=DEFAULT_SIGMA,
    )

    # Detector with chunk_size=None
    detector_none = MMD(
        kernel=kernel,
        chunk_size=None,
    )
    _ = detector_none.fit(X=X_ref)
    result_none = detector_none.compare(X=X_test)[0].distance

    # Detector with specific chunk_size
    detector_chunk = MMD(
        kernel=kernel,
        chunk_size=chunk_size,
    )
    _ = detector_chunk.fit(X=X_ref)
    result_chunk = detector_chunk.compare(X=X_test)[0].distance

    assert np.isclose(result_none, result_chunk)


@pytest.mark.parametrize(
    "chunk_size",
    [
        None,
        1,
        2,
    ],
)
def test_mmd_chunk_size_valid(
    chunk_size: Optional[int],
) -> None:
    """Test MMD initialization with valid chunk sizes.

    :param chunk_size: chunk size to test
    :type chunk_size: Optional[int]
    """
    np.random.seed(seed=RANDOM_SEED)
    X_ref = np.random.normal(0, 1, 100)
    X_test = np.random.normal(0, 1, 100)

    kernel = partial(
        rbf_kernel,
        sigma=DEFAULT_SIGMA,
    )

    detector = MMD(
        kernel=kernel,
        chunk_size=chunk_size,
    )
    _ = detector.fit(X=X_ref)
    result = detector.compare(X=X_test)[0]

    assert result is not None


@pytest.mark.parametrize(
    "chunk_size",
    [
        0,
        -1,
        "invalid",
        1.5,
        [1, 2],
        {1: 2},
    ],
)
def test_mmd_chunk_size_invalid(
    chunk_size: Any,
) -> None:
    """Test MMD initialization with invalid chunk sizes.

    :param chunk_size: chunk size to test
    :type chunk_size: Any
    """
    kernel = partial(
        rbf_kernel,
        sigma=DEFAULT_SIGMA,
    )

    with pytest.raises((TypeError, ValueError)):
        MMD(
            kernel=kernel,
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    "kernel",
    [
        partial(
            rbf_kernel,
            sigma=DEFAULT_SIGMA,
        ),
        lambda X, Y: X + Y,  # simple kernel
    ],
)
def test_mmd_kernel_valid(
    kernel: Callable,  # type: ignore
) -> None:
    """Test MMD initialization with valid kernels.

    :param kernel: kernel to test
    :type kernel: Callable
    """
    np.random.seed(seed=RANDOM_SEED)
    X_ref = np.random.normal(0, 1, 100)
    X_test = np.random.normal(0, 1, 100)

    detector = MMD(
        kernel=kernel,
    )
    _ = detector.fit(X=X_ref)
    result = detector.compare(X=X_test)[0]

    assert result is not None


@pytest.mark.parametrize(
    "kernel",
    [
        None,
        "invalid",
        123,
        [1, 2],
        {1: 2},
    ],
)
def test_mmd_kernel_invalid(
    kernel: Any,
) -> None:
    """Test MMD initialization with invalid kernels.

    :param kernel: kernel to test
    :type kernel: Any
    """
    with pytest.raises((TypeError, ValueError)):
        MMD(
            kernel=kernel,
        )
