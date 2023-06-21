"""Configuration file for the tests."""

from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import pytest  # type: ignore
import sklearn.linear_model  # type: ignore # pylint: disable=import-error
import sklearn.pipeline  # type: ignore # pylint: disable=import-error
import sklearn.preprocessing  # type: ignore # pylint: disable=import-error

from frouros.datasets.real import Elec2
from frouros.datasets.synthetic import Dummy, SEA
from frouros.metrics import PrequentialError


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


@pytest.fixture(scope="function")
def stream_drift() -> np.ndarray:
    """Stream with drift.

    :return: stream with drift
    :rtype: np.ndarray
    """
    np.random.seed(seed=31)

    dist_normal = np.random.normal(
        loc=0.0,
        scale=0.5,
        size=100,
    )
    dist_drift_1 = np.random.normal(
        loc=4.0,
        scale=0.2,
        size=100,
    )
    dist_drift_2 = np.random.normal(
        loc=6.0,
        scale=0.5,
        size=100,
    )

    return np.concatenate((dist_normal, dist_drift_1, dist_drift_2))


@pytest.fixture(scope="module", name="clf_dataset")
def classification_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classification dataset using SEA generator.

    :return: classification dataset
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    concept_samples = 200
    generator = SEA(seed=31)

    concepts = [
        generator.generate_dataset(
            block=block,
            noise=0.0,
            num_samples=concept_samples,
        )
        for block in [1, 4, 3, 2]
    ]

    X, y = [], []  # noqa: N806
    for concept in concepts:
        for X_sample, y_sample in concept:  # noqa: N806
            X.append(X_sample)
            y.append(y_sample)
    X = np.array(X)  # noqa: N806
    y = np.array(y)

    split_idx = concept_samples

    X_ref = X[:split_idx]  # noqa: N806
    y_ref = y[:split_idx]

    X_test = X[split_idx:]  # noqa: N806
    y_test = y[split_idx:]

    return X_ref, y_ref, X_test, y_test


@pytest.fixture(scope="module")
def categorical_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Categorical variables dataset.

    :return: dataset
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    X_ref = np.array(  # noqa: N806
        ["a", "a", "b", "a", "a", "b"],
        dtype=object,
    )
    X_test = np.array(  # noqa: N806
        ["b", "a", "c", "c", "c", "c"],
        dtype=object,
    )

    return X_ref, X_test


@pytest.fixture(scope="module")
def numerical_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Numerical variables dataset.

    :return: dataset
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    np.random.seed(seed=31)
    X_ref = np.random.normal(loc=[0, 5], scale=[1, 1], size=(10, 2))  # noqa: N806
    X_test = np.random.normal(loc=[0, 1], scale=[1, 0.5], size=(10, 2))  # noqa: N806

    return X_ref, X_test


@pytest.fixture(scope="module")
def elec2_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elec2 dataset.

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


def _generate_multivariate_normal_samples(
    mean: np.ndarray, cov: np.ndarray, num_samples: int = 100, seed: int = 31
) -> np.ndarray:
    np.random.seed(seed=seed)
    X = np.random.multivariate_normal(  # noqa: N806
        mean=mean, cov=cov, size=num_samples
    )
    return X


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
def X_ref_multivariate(  # noqa: N802
    multivariate_distribution_p: Tuple[  # pylint: disable=redefined-outer-name
        np.ndarray, np.ndarray
    ]
) -> np.ndarray:
    """Reference multivariate data.

    :param multivariate_distribution_p: multivariate distribution p
    :type multivariate_distribution_p: Tuple[numpy.ndarray, numpy.ndarray]
    :return: reference multivariate data
    :rtype: numpy.ndarray
    """
    X = _generate_multivariate_normal_samples(  # noqa: N806
        *multivariate_distribution_p
    )

    return X


@pytest.fixture(scope="module")
def X_test_multivariate(  # noqa: N802
    multivariate_distribution_q: Tuple[  # pylint: disable=redefined-outer-name
        np.ndarray, np.ndarray
    ]
) -> np.ndarray:
    """Test multivariate data.

    :param multivariate_distribution_q: multivariate distribution p
    :type multivariate_distribution_q: Tuple[numpy.ndarray, numpy.ndarray]
    :return: test multivariate data
    :rtype: numpy.ndarray
    """
    X = _generate_multivariate_normal_samples(  # noqa: N806
        *multivariate_distribution_q
    )

    return X


@pytest.fixture(scope="module")
def univariate_distribution_p() -> Tuple[float, float]:
    """Univariate distribution p.

    :return: mean and standard deviation of distribution p
    :rtype: Tuple[float, float]
    """
    mean, std = 1, 1

    return mean, std


@pytest.fixture(scope="module")
def univariate_distribution_q() -> Tuple[float, float]:
    """Univariate distribution q.

    :return: mean and standard deviation of distribution q
    :rtype: Tuple[float, float]
    """
    mean, std = 5, 2

    return mean, std


def _generate_univariate_normal_samples(
    mean: float, std: float, num_samples: int = 100, seed: int = 31
) -> np.ndarray:
    np.random.seed(seed=seed)
    X = np.random.normal(loc=mean, scale=std, size=num_samples)  # noqa: N806
    return X


@pytest.fixture(scope="module")
def X_ref_univariate(  # noqa: N802
    univariate_distribution_p: Tuple[  # pylint: disable=redefined-outer-name
        float, float
    ]
) -> np.ndarray:
    """Reference univariate data.

    :param univariate_distribution_p: univariate distribution p
    :type univariate_distribution_p: Tuple[float, float]
    :return: reference univariate data
    :rtype: numpy.ndarray
    """
    X = _generate_univariate_normal_samples(*univariate_distribution_p)  # noqa: N806

    return X


@pytest.fixture(scope="module")
def X_test_univariate(  # noqa: N802
    univariate_distribution_q: Tuple[  # pylint: disable=redefined-outer-name
        float, float
    ]
) -> np.ndarray:
    """Test multivariate data.

    :param univariate_distribution_q: univariate distribution q
    :type univariate_distribution_q: Tuple[float, float]
    :return: test univariate data
    :rtype: numpy.ndarray
    """
    X = _generate_univariate_normal_samples(*univariate_distribution_q)  # noqa: N806

    return X


@pytest.fixture(scope="module")
def prequential_error():
    """Prequential error.

    :return: prequential error
    :rtype: PrequentialError
    """
    return PrequentialError()


class DummyClassificationModel:
    """Dummy classification model class."""

    def __init__(self, num_classes: int = 2, seed: Optional[int] = None) -> None:
        """Init method.

        :param num_classes: number of classes
        :type num_classes: int
        """
        self.num_classes = num_classes
        np.random.seed(seed=seed)

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):  # noqa: N803, W0613
        """Fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :return: random class prediction
        :rtype: numpy.ndarray
        """
        _ = (X, y, args, kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict method.

        :param X: feature data
        :type X: numpy.ndarray
        :return: random class prediction
        :rtype: numpy.ndarray
        """
        prediction = np.random.randint(low=0, high=self.num_classes, size=X.shape[0])
        return prediction


@pytest.fixture(scope="module")
def train_prediction_normal(
    clf_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Train a model and use a test dataset to obtain the predictions.

    :param clf_dataset: dataset generated using SEA
    :type clf_dataset: Tuple[numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray]
    :return: test predictions from trained model
    :rtype: numpy.ndarray
    """
    X_ref, y_ref, X_test, _ = clf_dataset  # noqa: N806

    model = DummyClassificationModel(num_classes=len(np.unique(y_ref)), seed=31)
    model.fit(X=X_ref, y=y_ref)
    y_pred = model.predict(X=X_test)

    return y_pred


@pytest.fixture(scope="module", name="dataset_simple")
def concept_drift_dataset_simple() -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """Dataset with multiple concepts to induce concept drift.

    :return: dataset split in reference and test
    :rtype: Tuple[Tuple[numpy.ndarray, numpy.ndarray],
    Tuple[numpy.ndarray, numpy.ndarray]]
    """
    concept_samples = 100
    concept_types = [1, 1, 0, 1]

    generator = Dummy(seed=31)
    concepts = [
        generator.generate_dataset(
            class_=concept_type,
            num_samples=concept_samples,
        )
        for concept_type in concept_types
    ]

    X, y = [], []  # noqa: N806
    for concept in concepts:
        for X_sample, y_sample in concept:  # noqa: N806
            X.append(X_sample)
            y.append(y_sample)
    X = np.array(X)  # noqa: N806
    y = np.array(y)

    num_ref_concept_types = 1
    idx_ref = num_ref_concept_types * concept_samples
    X_ref, y_ref = X[:idx_ref], y[:idx_ref]  # noqa: N806
    X_test, y_test = X[idx_ref:], y[idx_ref:]  # noqa: N806

    return (X_ref, y_ref), (X_test, y_test)


@pytest.fixture(scope="module", name="model")
def concept_drift_model(
    dataset_simple: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> sklearn.pipeline.Pipeline:
    """Model used for concept drift.

    :param dataset_simple: dataset with concept drift
    :type dataset_simple: Tuple[Tuple[numpy.ndarray, numpy.ndarray],
    :return: trained model
    :rtype: sklearn.pipeline.Pipeline
    """
    (X_ref, y_ref), _ = dataset_simple  # noqa: N806

    pipeline = sklearn.pipeline.Pipeline(
        [
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("model", sklearn.linear_model.LogisticRegression()),
        ]
    )
    pipeline.fit(X=X_ref, y=y_ref)

    return pipeline


@pytest.fixture(scope="module", name="model_errors")
def concept_drift_model_errors_simple(
    dataset_simple: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model: sklearn.pipeline.Pipeline,
) -> List[int]:
    """Compute model errors given a dataset with concept drift.

    :param dataset_simple: dataset with concept drift
    :type dataset_simple: Tuple[Tuple[numpy.ndarray, numpy.ndarray],
    Tuple[numpy.ndarray, numpy.ndarray]]
    :param model: trained model
    :type model: sklearn.pipeline.Pipeline
    :return: model errors
    :rtype: List[int]
    """
    _, (X_test, y_test) = dataset_simple  # noqa: N806

    y_test_pred = model.predict(X_test)
    error = (1 - y_test_pred == y_test).astype(int).tolist()

    return error
