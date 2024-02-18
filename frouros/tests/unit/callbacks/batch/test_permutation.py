"""Test permutation module."""

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.utils.stats import permutation


def statistic(X: np.ndarray, Y: np.ndarray) -> float:  # noqa: N803
    """Statistic method.

    :param X: X data
    :type X: numpy.ndarray
    :param Y: Y data
    :type Y: numpy.ndarray
    :return: statistic
    :rtype: float
    """
    return np.abs(X.mean() - Y.mean())


@pytest.mark.parametrize(
    "X, Y, expected_num_permutations, expected_permutation_mean",
    [
        (np.array([1, 2, 3]), np.array([10, 20, 30]), 720, 7.6),
        (np.array([*range(1, 11)]), np.array([*range(1, 101, 10)]), 1000, 10.3654),
    ],
)
def test_permutation(
    X: np.ndarray,  # noqa: N803
    Y: np.ndarray,
    expected_num_permutations: int,
    expected_permutation_mean: float,
) -> None:
    """Test permutation method.

    :param X: X data
    :type X: numpy.ndarray
    :param Y: Y data
    :type Y: numpy.ndarray
    :param expected_num_permutations: expected number of permutations
    :type expected_num_permutations: int
    :param expected_permutation_mean: expected permutation mean
    :type expected_permutation_mean: float
    """
    permutations, _ = permutation(
        X=X,
        Y=Y,
        statistic=statistic,
        statistical_args={},
        num_permutations=1000,
        num_jobs=1,
        random_state=31,
    )

    assert len(permutations) == expected_num_permutations
    assert np.isclose(np.array(permutations).mean(), expected_permutation_mean)
