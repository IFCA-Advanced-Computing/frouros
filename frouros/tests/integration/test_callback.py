"""Test callback module."""

from typing import List, Tuple

import numpy as np  # type: ignore
import pytest  # type: ignore
import sklearn  # type: ignore # pylint: disable=import-error

from frouros.callbacks.batch import (
    PermutationTestDistanceBased,
    ResetStatisticalTest,
)
from frouros.callbacks.streaming import (
    HistoryConceptDrift,
    mSPRT,
    WarningSamplesBuffer,
)
from frouros.detectors.concept_drift import (
    ADWIN,
    CUSUM,
    DDM,
    ECDDWT,
    EDDM,
    GeometricMovingAverage,
    HDDMA,
    HDDMW,
    KSWIN,
    PageHinkley,
    RDDM,
    STEPD,
)
from frouros.detectors.concept_drift.base import BaseConceptDrift
from frouros.detectors.concept_drift.streaming.statistical_process_control.base import (
    BaseSPC,
)
from frouros.detectors.data_drift.batch import (
    BhattacharyyaDistance,
    CVMTest,
    EMD,
    HellingerDistance,
    HINormalizedComplement,
    JS,
    KL,
    KSTest,
    MMD,
    PSI,
    WelchTTest,
)
from frouros.detectors.data_drift.batch.base import BaseDataDriftBatch
from frouros.detectors.data_drift.streaming import MMD as MMDStreaming  # noqa: N811


@pytest.mark.parametrize(
    "detector_class, expected_distance, expected_p_value",
    [
        (BhattacharyyaDistance, 0.55516059, 0.0),
        (EMD, 3.85346006, 0.0),
        (HellingerDistance, 0.74509099, 0.0),
        (HINormalizedComplement, 0.78, 0.0),
        (JS, 0.67010107, 0.0),
        (KL, np.inf, 0.06),
        (MMD, 0.69509004, 0.0),
        (PSI, 461.20379435, 0.0),
    ],
)
def test_batch_permutation_test_data_univariate_different_distribution(
    X_ref_univariate: np.ndarray,  # noqa: N803
    X_test_univariate: np.ndarray,
    detector_class: BaseDataDriftBatch,
    expected_distance: float,
    expected_p_value: float,
) -> None:
    """Test batch permutation test on data callback.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param detector_class: detector distance
    :type detector_class: BaseDataDriftBatch
    :param expected_distance: expected distance value
    :type expected_distance: float
    :param expected_p_value: expected p-value value
    :type expected_p_value: float
    """
    np.random.seed(seed=31)

    permutation_test_name = "permutation_test"
    detector = detector_class(  # type: ignore
        callbacks=[
            PermutationTestDistanceBased(
                num_permutations=100,
                random_state=31,
                num_jobs=-1,
                name=permutation_test_name,
            )
        ]
    )
    _ = detector.fit(X=X_ref_univariate)
    distance, callback_logs = detector.compare(X=X_test_univariate)

    assert np.isclose(distance, expected_distance)
    assert np.isclose(
        callback_logs[permutation_test_name]["p_value"],
        expected_p_value,
    )


@pytest.mark.parametrize(
    "detector_class",
    [CVMTest, KSTest, WelchTTest],
)
def test_batch_reset_on_statistical_test_data_drift(
    X_ref_univariate,  # noqa: N803
    X_test_univariate,
    detector_class: BaseDataDriftBatch,
    mocker,
) -> None:
    """Test batch reset on statistical test data drift callback.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param detector_class: detector distance
    :type detector_class: BaseDataDriftBatch
    """
    mocker.patch("frouros.detectors.data_drift.batch.base.BaseDataDriftBatch.reset")

    detector = detector_class(  # type: ignore
        callbacks=[
            ResetStatisticalTest(
                alpha=0.01,
            ),
        ],
    )
    _ = detector.fit(X=X_ref_univariate)
    _ = detector.compare(X=X_test_univariate)
    detector.reset.assert_called_once()  # type: ignore # pylint: disable=no-member


@pytest.mark.parametrize(
    "detector_class",
    [
        ADWIN,
        CUSUM,
        DDM,
        ECDDWT,
        EDDM,
        GeometricMovingAverage,
        HDDMA,
        HDDMW,
        KSWIN,
        PageHinkley,
        RDDM,
        STEPD,
    ],
)
def test_streaming_history_on_concept_drift(
    model_errors: List[int],
    detector_class: BaseConceptDrift,
):
    """Test streaming history on concept drift callback.

    :param model_errors: model errors
    :type model_errors: List[int]
    :param detector_class: concept drift detector
    :type detector_class: BaseConceptDrift
    """
    name = "history"
    detector = detector_class(callbacks=HistoryConceptDrift(name=name))  # type: ignore

    for error in model_errors:
        history = detector.update(value=error)
        status = detector.status
        if status["drift"]:
            assert history[name]["drift"][-1]
            assert not any(history[name]["drift"][:-1])
            break


def _fit_model(model, X, y):  # noqa: N803
    model.fit(X=X, y=y)
    return model


@pytest.mark.parametrize(
    "detector_class",
    [
        DDM,
        ECDDWT,
        EDDM,
        HDDMA,
        HDDMW,
        RDDM,
    ],  # pylint: disable=too-many-locals
)
def test_streaming_warning_samples_buffer_on_concept_drift(
    dataset_simple: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    model: sklearn.pipeline.Pipeline,
    detector_class: BaseSPC,
):
    """Test streaming warning samples buffer on concept drift callback.

    :param dataset_simple: dataset with concept drift
    :type dataset_simple: Tuple[Tuple[numpy.ndarray, numpy.ndarray],
    :param model: trained model
    :type model: sklearn.pipeline.Pipeline
    :param detector_class: concept drift detector
    :type detector_class: BaseSPC
    """
    _, test = dataset_simple  # noqa: N806

    detector = detector_class(
        callbacks=WarningSamplesBuffer(name="samples"),  # type: ignore
    )

    collect_example_warning_samples = False
    X_extra, y_extra = [], []  # noqa: N806

    for X, y in zip(*test):  # noqa: N806
        y_pred = model.predict(X.reshape(1, -1))
        if not collect_example_warning_samples:
            error = 1 - int(y_pred == y)
            callbacks_logs = detector.update(value=error, X=X, y=y)
        else:
            X_extra.append(X)
            y_extra.append(y)
        if detector.status["drift"]:
            y_new_ref = callbacks_logs["samples"]["y"] + y_extra
            if len([*set(y_new_ref)]) < 2:
                collect_example_warning_samples = True
            else:
                X_new_ref = callbacks_logs["samples"]["X"] + X_extra  # noqa: N806
                collect_example_warning_samples = False
                X_extra.clear()
                y_extra.clear()
                detector.reset()
                model = _fit_model(model=model, X=X_new_ref, y=y_new_ref)


@pytest.mark.parametrize(
    "detector_class,"
    " expected_drift_idx,"
    " expected_distance_mean,"
    " expected_p_value,"
    " expected_likelihood",
    [
        (MMDStreaming, 5, 0.27193007, 0.00989585, 101.05240437),
    ],
)
def test_streaming_msprt_multivariate_different_distribution(
    X_ref_multivariate: np.ndarray,  # noqa: N803
    X_test_multivariate: np.ndarray,
    detector_class: BaseDataDriftBatch,
    expected_drift_idx: int,
    expected_distance_mean: float,
    expected_p_value: float,
    expected_likelihood: float,
) -> None:
    """Test streaming mSPRT test on data callback.

    :param X_ref_multivariate: reference multivariate data
    :type X_ref_multivariate: numpy.ndarray
    :param X_test_multivariate: test multivariate data
    :type X_test_multivariate: numpy.ndarray
    :param detector_class: detector distance
    :type detector_class: BaseDataDriftBatch
    :param expected_drift_idx: expected drift index
    :type expected_drift_idx: int
    :param expected_distance_mean: expected distance mean value
    :type expected_distance_mean: float
    :param expected_p_value: expected p-value value
    :type expected_p_value: float
    :param expected_likelihood: expected likelihood value
    :type expected_likelihood: float
    """
    np.random.seed(seed=31)

    alpha = 0.01

    detector = detector_class(  # type: ignore
        callbacks=mSPRT(
            alpha=alpha,
            sigma=1,
            tau=1,
            lambda_=32,
        ),
        window_size=5,
    )
    _ = detector.fit(X=X_ref_multivariate)

    drift_idx = -1
    for i, sample in enumerate(X_test_multivariate, start=1):
        value, callbacks_logs = detector.update(value=sample)
        if value is not None:
            if callbacks_logs["mSPRT"]["p_value"] < alpha:
                drift_idx = i
                break

    assert drift_idx == expected_drift_idx
    assert np.isclose(callbacks_logs["mSPRT"]["distance_mean"], expected_distance_mean)
    assert np.isclose(callbacks_logs["mSPRT"]["p_value"], expected_p_value)
    assert np.isclose(callbacks_logs["mSPRT"]["likelihood"], expected_likelihood)
