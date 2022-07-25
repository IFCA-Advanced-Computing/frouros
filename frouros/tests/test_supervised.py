"""Test supervised methods."""

from typing import Tuple

import pytest  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import SupervisedBaseEstimator
from frouros.supervised.cusum_based import (
    CUSUM,
    CUSUMConfig,
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
    PageHinkley,
    PageHinkleyConfig,
)
from frouros.supervised.ddm_based import (
    DDM,
    DDMConfig,
    ECDDWT,
    ECDDWTConfig,
    EDDM,
    EDDMConfig,
    HDDMA,
    HDDMAConfig,
    HDDMW,
    HDDMWConfig,
    RDDM,
    RDDMConfig,
)
from frouros.supervised.statistical_test import STEPD, STEPDConfig
from frouros.supervised.window_based import ADWIN, ADWINConfig, KSWIN, KSWINConfig

from frouros.common.exceptions import OneSampleError


ESTIMATOR = DecisionTreeClassifier
ESTIMATOR_ARGS = {
    "random_state": 31,
}
MIN_NUM_INSTANCES = 500
CUMSUM_ARGS = {
    "delta": 0.005,
    "lambda_": 50,
}
HDDM_ARGS = {
    "alpha_w": 0.005,
    "alpha_d": 0.001,
}
HDDMW_ARGS = {
    **HDDM_ARGS,
    "lambda_": 0.05,
}


def error_scorer(y_true, y_pred):
    """Error scorer function."""
    return 1 - accuracy_score(y_true, y_pred)


supervised_methods = (
    [
        ADWIN(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=ADWINConfig(
                clock=32,
                delta=0.15,
                m=5,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        KSWIN(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=KSWINConfig(
                alpha=0.005,
                seed=31,
                min_num_instances=100,
                num_test_instances=30,
            ),
        ),
        CUSUM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=CUSUMConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                **CUMSUM_ARGS,
            ),
        ),
        GeometricMovingAverage(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=GeometricMovingAverageConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                lambda_=1.0,
                alpha=0.99,
            ),
        ),
        PageHinkley(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=PageHinkleyConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                **CUMSUM_ARGS,
                alpha=0.9999,
            ),
        ),
        DDM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=DDMConfig(
                warning_level=2.0,
                drift_level=3.0,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        ECDDWT(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=ECDDWTConfig(
                lambda_=0.2,
                warning_level=0.5,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        EDDM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=EDDMConfig(
                alpha=0.95,
                beta=0.9,
                level=2.0,
                min_num_misclassified_instances=MIN_NUM_INSTANCES,
            ),
        ),
        HDDMA(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=HDDMAConfig(
                **HDDM_ARGS,
                two_sided_test=False,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        HDDMA(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=HDDMAConfig(
                **HDDM_ARGS,
                two_sided_test=True,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        HDDMW(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=HDDMWConfig(
                **HDDM_ARGS,
                two_sided_test=False,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        HDDMW(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=HDDMWConfig(
                **HDDM_ARGS,
                two_sided_test=True,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        RDDM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            error_scorer=error_scorer,
            config=RDDMConfig(
                warning_level=1.773,
                drift_level=2.258,
                max_concept_size=40000,
                min_concept_size=7000,
                max_num_instances_warning=1400,
                min_num_instances=129,
            ),
        ),
        STEPD(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=STEPDConfig(
                alpha_d=0.003,
                alpha_w=0.05,
                min_num_instances=30,
            ),
        ),
    ],
)


@pytest.mark.parametrize("model_detector", *supervised_methods)
def test_supervised_method(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    model_detector: SupervisedBaseEstimator,
) -> None:
    """Test supervised dataset.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806

    model_detector.fit(X=X_ref, y=y_ref)

    for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
        _ = model_detector.predict(X=np.array([*X_sample]).reshape(1, -1))

        # Delayed targets arriving....
        _ = model_detector.update(y=np.array([y_sample]))


@pytest.mark.parametrize("model_detector", *supervised_methods)
def test_not_supported_multi_sample_predict(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    model_detector: SupervisedBaseEstimator,
) -> None:
    """Test not supported multiple sample predict.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    """
    X_ref, y_ref, X_test, _ = classification_dataset  # noqa: N806

    model_detector.fit(X=X_ref, y=y_ref)

    with pytest.raises(OneSampleError):
        model_detector.predict(X=X_test)


@pytest.mark.parametrize("model_detector", *supervised_methods)
def test_not_supported_multi_sample_update(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    model_detector: SupervisedBaseEstimator,
) -> None:
    """Test not supported multiple sample update.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806

    model_detector.fit(X=X_ref, y=y_ref)
    model_detector.predict(X=X_test[0, :].reshape(1, -1))

    with pytest.raises(OneSampleError):
        model_detector.update(y=y_test)
