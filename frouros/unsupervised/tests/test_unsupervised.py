"""Test unsupervised methods."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore
from sklearn.gaussian_process.kernels import RBF  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from frouros.unsupervised.base import UnsupervisedBaseEstimator
from frouros.unsupervised.distance_based import EMD, PSI, MMD
from frouros.unsupervised.statistical_test import CVMTest, KSTest
from frouros.unsupervised.utils import get_statistical_test


@pytest.mark.parametrize("detector", [EMD(), PSI(), CVMTest(), KSTest()])
def test_univariate_test(
    dataset: Tuple[np.array, np.array, np.array], detector: UnsupervisedBaseEstimator
) -> None:
    """Test univariate method.

    :param dataset: Elec2 raw dataset
    :type dataset: Tuple[numpy.array, numpy.array, numpy.array]
    """
    X_ref, y_ref, X_test = dataset  # noqa: N806

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("detector", detector),
            ("model", LogisticRegression(solver="lbfgs", max_iter=1000)),
        ]
    )

    pipe.fit(X=X_ref, y=y_ref)

    _ = pipe.predict(X=np.array([*X_test]))
    _ = get_statistical_test(estimator=pipe, detector_name="detector")


@pytest.mark.parametrize(
    "detector", [MMD(num_permutations=1000, kernel=RBF(), random_state=31)]
)
def test_distance_based_multivariate_different_distribution_test(
    multivariate_distribution_p: Tuple[np.ndarray, np.ndarray],
    multivariate_distribution_q: Tuple[np.ndarray, np.ndarray],
    detector: UnsupervisedBaseEstimator,
    num_samples: int = 500,
) -> None:
    """Test c multivariate different distribution method.

    :param multivariate_distribution_p: mean and covariance matrix of distribution p
    :type multivariate_distribution_p: Tuple[numpy.ndarray, numpy.ndarray]
    :param multivariate_distribution_q: mean and covariance matrix of distribution q
    :type multivariate_distribution_q: Tuple[numpy.ndarray, numpy.ndarray]
    """
    np.random.seed(seed=31)
    X_ref = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )
    X_test = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_q, size=num_samples
    )

    detector.fit(X=X_ref)
    detector.transform(X=X_test)
    statistic, p_value = detector.test  # type: ignore

    assert np.isclose(statistic, 0.09446612)
    assert np.isclose(p_value, 0.0)


@pytest.mark.parametrize(
    "detector", [MMD(num_permutations=1000, kernel=RBF(), random_state=31)]
)
def test_distance_based_multivariate_same_distribution_test(
    multivariate_distribution_p: Tuple[np.ndarray, np.ndarray],
    detector: UnsupervisedBaseEstimator,
    num_samples: int = 500,
) -> None:
    """Test distance based multivariate same distribution method.

    :param multivariate_distribution_p: mean and covariance matrix of distribution p
    :type multivariate_distribution_p: Tuple[numpy.ndarray, numpy.ndarray]
    """
    np.random.seed(seed=31)
    X_ref = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )
    X_test = np.random.multivariate_normal(  # noqa: N806
        *multivariate_distribution_p, size=num_samples
    )

    detector.fit(X=X_ref)
    detector.transform(X=X_test)
    statistic, p_value = detector.test  # type: ignore

    assert np.isclose(statistic, 0.00256109)
    assert np.isclose(p_value, 0.894)
