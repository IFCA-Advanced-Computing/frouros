"""Test statistical test estimators module."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from frouros.unsupervised.statistical_test.base import (  # type: ignore
    StatisticalTestEstimator,
)
from frouros.unsupervised.utils import get_statistical_test
from frouros.unsupervised.statistical_test.cvm import CVMTest  # type: ignore
from frouros.unsupervised.statistical_test.ks import KSTest  # type: ignore


@pytest.mark.parametrize("detector", [CVMTest(), KSTest()])
def test_statistical_test_detector(
    dataset: Tuple[np.array, np.array, np.array], detector: StatisticalTestEstimator
) -> None:
    """Test feature detector.

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
    _ = get_statistical_test(estimator=pipe, detector_name="detector")  # pylint: disable=E1101
