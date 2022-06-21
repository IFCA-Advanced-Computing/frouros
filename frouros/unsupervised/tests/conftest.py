"""Configuration file for the tests."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore

from frouros.datasets.real import Elec2


@pytest.fixture(scope="module")
def dataset_categorical() -> Tuple[np.ndarray, np.ndarray]:
    """Dataset using categorical variables.

    :return: dataset
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    X_ref = np.array(  # noqa: N806
        [["a", "A"], ["a", "B"], ["b", "B"], ["a", "C"], ["a", "A"], ["b", "C"]],
        dtype=object,
    )
    X_test = np.array(  # noqa: N806
        [["b", "A"], ["c", "B"], ["c", "A"], ["c", "A"], ["b", "C"], ["b", "C"]],
        dtype=object,
    )

    return X_ref, X_test


@pytest.fixture(scope="module")
def dataset_elec2() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dataset using Elec2.

    :return: dataset
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    elec2 = Elec2()
    elec2.download()
    dataset_ = elec2.load()

    # Two comprehensions lists are faster than iterating over one
    # (Python doing Python´s things).
    X = np.array(  # noqa: N806
        [sample.tolist()[:-1] for sample in dataset_], dtype=np.float16
    )
    y = np.array([sample[-1] for sample in dataset_], dtype="str")

    idx = X.shape[0] // 2
    X_ref = X[:idx]  # noqa: N806
    y_ref = y[:idx]

    X_test = X[idx:, :]  # noqa: N806

    return X_ref, y_ref, X_test


@pytest.fixture(scope="module")
def multivariate_distribution_p() -> Tuple[np.ndarray, np.ndarray]:
    """Multivariate distribution p.

    :return: mean and covariance matrix of distribution p
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    mean = np.ones(2)
    cov = 2 * np.eye(2)

    return mean, cov


@pytest.fixture(scope="module")
def multivariate_distribution_q() -> Tuple[np.ndarray, np.ndarray]:
    """Multivariate distribution q.

    :return: mean and covariance matrix of distribution q
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    mean = np.zeros(2)
    cov = 3 * np.eye(2) - 1

    return mean, cov
