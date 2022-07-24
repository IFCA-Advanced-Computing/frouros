"""Configuration file for the tests."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore

from frouros.datasets.real import Elec2
from frouros.datasets.synthetic import SEA
from frouros.metrics.prequential_error import PrequentialError


# Elec2 fixtures
@pytest.fixture(scope="function")
def elec2_raw() -> Elec2:
    """Elec2 raw dataset.

    :return: Elec2 raw dataset
    :rtype: Elec2
    """
    dataset = Elec2()
    dataset.download()
    return dataset


@pytest.fixture(scope="function")
def elec2(elec2_raw: Elec2) -> np.ndarray:  # pylint: disable=redefined-outer-name
    """Elec2 dataset.

    :param elec2_raw: Elec2 raw dataset
    :type elec2_raw: Elec2
    :return: Elec2 dataset
    :rtype: np.ndarray
    """
    return elec2_raw.load()


# SEA fixtures
@pytest.fixture(scope="function")
def sea() -> SEA:
    """SEA dataset generator.

    :return: SEA dataset generator
    :rtype: SEA
    """
    generator = SEA()
    return generator


@pytest.fixture(scope="module")
def classification_dataset() -> Tuple[np.array, np.array, np.array, np.array]:
    """Classification dataset using SEA generator.

    :return: classification dataset
    :rtype: Tuple[np.array, np.array, np.array, np.array]
    """
    concept_samples = 100
    generator = SEA(seed=31)
    concept_1 = generator.generate_dataset(block=1, noise=0.0, num_samples=concept_samples, )
    concept_2 = generator.generate_dataset(block=4, noise=0.0, num_samples=concept_samples, )
    concept_3 = generator.generate_dataset(block=3, noise=0.0, num_samples=concept_samples, )
    concept_4 = generator.generate_dataset(block=2, noise=0.0, num_samples=concept_samples, )

    X, y = [], []
    for concept in [concept_1, concept_2, concept_3, concept_4]:
        for X_sample, y_sample in concept:
            X.append(X_sample)
            y.append(y_sample)
    X = np.array(X)
    y = np.array(y)

    split_idx = concept_samples

    X_ref = X[:split_idx]  # noqa: N806
    y_ref = y[:split_idx]

    X_test = X[split_idx:]  # noqa: N806
    y_test = y[split_idx:]

    return X_ref, y_ref, X_test, y_test


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
    elec2 = Elec2()  # pylint: disable=redefined-outer-name
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


@pytest.fixture(scope="module")
def prequential_error():
    """Prequential error.

    :return: prequential error
    :rtype: PrequentialError
    """
    return PrequentialError(
        error_scorer=lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred)
    )
