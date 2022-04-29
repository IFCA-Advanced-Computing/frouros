"""Test supervised methods."""

from typing import Tuple

import pytest  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import TargetDelayEstimator
from frouros.supervised.ddm_based.base import DDMConfig, EDDMConfig
from frouros.supervised.ddm_based import DDM, EDDM
from frouros.supervised.utils import update_detector


@pytest.mark.parametrize(
    "detector",
    [
        DDM(
            estimator=LogisticRegression(solver="lbfgs", max_iter=1000),
            error_scorer=lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred),
            config=DDMConfig(
                warning_level=2.0,
                drift_level=3.0,
                min_num_instances=500,
            ),
        ),
        EDDM(
            estimator=LogisticRegression(solver="lbfgs", max_iter=1000),
            config=EDDMConfig(
                alpha=0.95,
                beta=0.9,
                level=2.0,
                min_num_misclassified_instances=500,
            ),
        ),
    ],
)
def test_supervised_method(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    detector: TargetDelayEstimator,
) -> None:
    """Test supervised dataset.

    :param classification_dataset: Elec2 raw dataset
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806

    pipe = Pipeline([("scaler", StandardScaler()), ("detector", detector)])

    pipe.fit(X=X_ref, y=y_ref)

    for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
        _ = pipe.predict(X=np.array([*X_sample]).reshape(1, -1))

        # Delayed targets arriving....
        _ = update_detector(
            estimator=pipe, y=np.array([y_sample]), detector_name="detector"
        )
