"""Test unsupervised methods."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from frouros.unsupervised.base import UnsupervisedBaseEstimator
from frouros.unsupervised.distance_based import EMD
from frouros.unsupervised.statistical_test import CVMTest, KSTest
from frouros.unsupervised.utils import get_statistical_test


@pytest.mark.parametrize("detector", [EMD(), CVMTest(), KSTest()])
def test_unsupervised_method(
    dataset: Tuple[np.array, np.array, np.array], detector: UnsupervisedBaseEstimator
) -> None:
    """Test unsupervised method.

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
