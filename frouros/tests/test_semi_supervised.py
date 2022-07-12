"""Test semi-supervised methods."""

from typing import Tuple

import pytest  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
import numpy as np  # type: ignore

from frouros.common.update import update_detector
from frouros.supervised.base import SupervisedBaseEstimator
from frouros.semi_supervised.margin_density_based import (
    MD3RS,
    MD3RSConfig,
    MD3SVM,
    MD3SVMConfig,
)


@pytest.mark.parametrize(
    "detector",
    [
        MD3RS(
            estimator=DecisionTreeClassifier(random_state=31),
            metric_scorer=accuracy_score,
            config=MD3RSConfig(
                chunk_size=2500,
                sensitivity=2.0,
                num_folds=5,
                ratio_random_features=0.5,
                margin_uncertainty=0.5,
            ),
            random_state=31,
        ),
        MD3SVM(
            metric_scorer=accuracy_score,
            config=MD3SVMConfig(
                chunk_size=2500,
                sensitivity=2.0,
                num_folds=5,
            ),
            random_state=31,
            svm_args={
                "C": 1.0,
                "penalty": "l2",
                "loss": "hinge",
            },
        ),
    ],
)
def test_semi_supervised_method(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    detector: SupervisedBaseEstimator,
) -> None:
    """Test semi-supervised dataset.

    :param classification_dataset: Elec2 raw dataset
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806

    pipe = Pipeline([("detector", detector)])

    pipe.fit(X=X_ref, y=y_ref)

    for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
        X_sample = np.array([*X_sample]).reshape(1, -1)  # noqa: N806
        _ = pipe.predict(X=X_sample)

        if pipe["detector"].drift_suspected:  # pylint: disable=no-member
            # Provide labeled samples as an oracle
            _ = update_detector(estimator=pipe, X=X_sample, y=np.array([y_sample]))
