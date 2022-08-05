"""Test supervised methods."""
import copy
from typing import Callable, Tuple

import pytest  # type: ignore
from sklearn.linear_model import SGDClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
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
    STEPD,
    STEPDConfig,
)
from frouros.supervised.ddm_based.modes import IncrementalLearningMode
from frouros.supervised.modes import NormalMode
from frouros.supervised.window_based import ADWIN, ADWINConfig, KSWIN, KSWINConfig

# from frouros.common.exceptions import OneSampleError


ESTIMATOR = SGDClassifier
ESTIMATOR_ARGS = {
    "loss": "log_loss",
    "random_state": 31,
}
MIN_NUM_INSTANCES = 30
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


normal_mode_methods = [
    (
        CUSUM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=CUSUMConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                **CUMSUM_ARGS,
            ),
        ),
        error_scorer,
    ),
    (
        GeometricMovingAverage(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=GeometricMovingAverageConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                alpha=0.99,
            ),
        ),
        error_scorer,
    ),
    (
        PageHinkley(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=PageHinkleyConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                alpha=0.9999,
            ),
        ),
        error_scorer,
    ),
    (
        ADWIN(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=ADWINConfig(
                clock=32,
                delta=0.15,
                m=5,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        KSWIN(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=KSWINConfig(
                alpha=0.005,
                seed=31,
                min_num_instances=100,
                num_test_instances=30,
            ),
        ),
        error_scorer,
    ),
]

incremental_mode_methods = [
    (
        DDM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=DDMConfig(
                warning_level=2.0,
                drift_level=3.0,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        ECDDWT(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=ECDDWTConfig(
                lambda_=0.2,
                warning_level=0.5,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        EDDM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=EDDMConfig(
                alpha=0.95,
                beta=0.9,
                level=2.0,
                min_num_misclassified_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        HDDMA(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=HDDMAConfig(
                **HDDM_ARGS,
                two_sided_test=False,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        HDDMA(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=HDDMAConfig(
                **HDDM_ARGS,
                two_sided_test=True,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        HDDMW(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=HDDMWConfig(
                **HDDM_ARGS,
                two_sided_test=False,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        HDDMW(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=HDDMWConfig(
                **HDDM_ARGS,
                two_sided_test=True,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        error_scorer,
    ),
    (
        RDDM(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=RDDMConfig(
                warning_level=1.773,
                drift_level=2.258,
                max_concept_size=40000,
                min_concept_size=7000,
                max_num_instances_warning=1400,
                min_num_instances=129,
            ),
        ),
        error_scorer,
    ),
    (
        STEPD(
            estimator=ESTIMATOR(**ESTIMATOR_ARGS),
            config=STEPDConfig(
                alpha_d=0.003,
                alpha_w=0.05,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        lambda y_true, y_pred: y_true == y_pred,
    ),
]

# NormalMode tests accept all methods but Incremental only support DDM-Based
supervised_methods = [
    *copy.deepcopy(incremental_mode_methods),
    *copy.deepcopy(normal_mode_methods),
]


@pytest.mark.parametrize("detector", supervised_methods)
def test_supervised_estimator_normal_mode(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    detector: Tuple[SupervisedBaseEstimator, Callable],
) -> None:
    """Test supervised estimator normal mode.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    :param detector: supervised detector
    :type detector: Tuple[SupervisedBaseEstimator, Callable]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806
    model_detector, value_func = detector

    model_detector.fit(X=X_ref, y=y_ref)
    mode = NormalMode(detector=model_detector, value_func=value_func)

    for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
        X_sample = np.array([*X_sample]).reshape(1, -1)  # noqa: N806
        y_sample = np.array([y_sample])
        y_pred = model_detector.predict(X=X_sample)

        # Update detector using auxiliary class NormalMode
        _ = mode.update(X=X_sample, y_true=y_sample, y_pred=y_pred)


@pytest.mark.parametrize("detector", supervised_methods)
def test_supervised_pipeline_normal_mode(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    detector: Tuple[SupervisedBaseEstimator, Callable],
) -> None:
    """Test supervised pipeline normal mode.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    :param detector: supervised detector
    :type detector: Tuple[SupervisedBaseEstimator, Callable]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806
    model_detector, value_func = detector

    model_detector = Pipeline(
        [("scaler", StandardScaler()), ("detector", model_detector)]
    )

    model_detector.fit(X=X_ref, y=y_ref)
    mode = NormalMode(detector=model_detector, value_func=value_func)

    for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
        X_sample = np.array([*X_sample]).reshape(1, -1)  # noqa: N806
        y_sample = np.array([y_sample])
        y_pred = model_detector.predict(X=X_sample)

        # Update detector using auxiliary class NormalMode
        _ = mode.update(X=X_sample, y_true=y_sample, y_pred=y_pred)


@pytest.mark.parametrize("detector", incremental_mode_methods)
def test_supervised_estimator_incremental_mode(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    detector: Tuple[SupervisedBaseEstimator, Callable],
) -> None:
    """Test supervised estimator incremental mode.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    :param detector: supervised detector
    :type detector: Tuple[SupervisedBaseEstimator, Callable]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806
    model_detector, value_func = detector

    model_detector.fit(X=X_ref, y=y_ref)
    mode = IncrementalLearningMode(detector=model_detector, value_func=value_func)

    for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
        X_sample = np.array([*X_sample]).reshape(1, -1)  # noqa: N806
        y_sample = np.array([y_sample])
        y_pred = model_detector.predict(X=X_sample)

        # Update detector using auxiliary class IncrementalLearningMode
        _ = mode.update(X=X_sample, y_true=y_sample, y_pred=y_pred)


@pytest.mark.parametrize("detector", incremental_mode_methods)
def test_supervised_pipeline_incremental_mode(
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    detector: Tuple[SupervisedBaseEstimator, Callable],
) -> None:
    """Test supervised pipeline incremental mode.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    :param detector: supervised detector
    :type detector: Tuple[SupervisedBaseEstimator, Callable]
    """
    X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806
    model_detector, value_func = detector

    model_detector = Pipeline(
        [("scaler", StandardScaler()), ("detector", model_detector)]
    )

    model_detector.fit(X=X_ref, y=y_ref)
    mode = IncrementalLearningMode(detector=model_detector, value_func=value_func)

    for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
        X_sample = np.array([*X_sample]).reshape(1, -1)  # noqa: N806
        y_sample = np.array([y_sample])
        y_pred = model_detector.predict(X=X_sample)

        # Update detector using auxiliary class IncrementalLearningMode
        _ = mode.update(X=X_sample, y_true=y_sample, y_pred=y_pred)


# @pytest.mark.parametrize("model_detector", *supervised_methods)
# def test_supervised_method(
#     classification_dataset: Tuple[np.array, np.array, np.array, np.array],
#     model_detector: SupervisedBaseEstimator,
# ) -> None:
#     """Test supervised dataset.
#
#     :param classification_dataset: dataset generated using SEA
#     :type classification_dataset: Tuple[numpy.array, numpy.array,
#     numpy.array, numpy.array]
#     """
#     X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806
#
#     model_detector.fit(X=X_ref, y=y_ref)
#
#     for X_sample, y_sample in zip(X_test, y_test):  # noqa: N806
#         _ = model_detector.predict(X=np.array([*X_sample]).reshape(1, -1))
#
#         # Delayed targets arriving....
#         _ = model_detector.update(y=np.array([y_sample]))


# @pytest.mark.parametrize("model_detector", *supervised_methods)
# def test_not_supported_multi_sample_predict(
#     classification_dataset: Tuple[np.array, np.array, np.array, np.array],
#     model_detector: SupervisedBaseEstimator,
# ) -> None:
#     """Test not supported multiple sample predict.
#
#     :param classification_dataset: dataset generated using SEA
#     :type classification_dataset: Tuple[numpy.array, numpy.array,
#     numpy.array, numpy.array]
#     """
#     X_ref, y_ref, X_test, _ = classification_dataset  # noqa: N806
#
#     model_detector.fit(X=X_ref, y=y_ref)
#
#     with pytest.raises(OneSampleError):
#         model_detector.predict(X=X_test)


# @pytest.mark.parametrize("model_detector", *supervised_methods)
# def test_not_supported_multi_sample_update(
#     classification_dataset: Tuple[np.array, np.array, np.array, np.array],
#     model_detector: SupervisedBaseEstimator,
# ) -> None:
#     """Test not supported multiple sample update.
#
#     :param classification_dataset: dataset generated using SEA
#     :type classification_dataset: Tuple[numpy.array, numpy.array,
#     numpy.array, numpy.array]
#     """
#     X_ref, y_ref, X_test, y_test = classification_dataset  # noqa: N806
#
#     model_detector.fit(X=X_ref, y=y_ref)
#     model_detector.predict(X=X_test[0, :].reshape(1, -1))
#
#     with pytest.raises(OneSampleError):
#         model_detector.update(y=y_test)
