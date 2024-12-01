"""Test callback module."""

from typing import Any

import numpy as np
import pytest

from frouros.callbacks.batch import (
    PermutationTestDistanceBased,
    ResetStatisticalTest,
)
from frouros.callbacks.streaming import (
    HistoryConceptDrift,
)
from frouros.detectors.concept_drift import (
    ADWIN,
    CUSUM,
    DDM,
    ECDDWT,
    EDDM,
    HDDMA,
    HDDMW,
    KSWIN,
    RDDM,
    STEPD,
    GeometricMovingAverage,
    PageHinkley,
)
from frouros.detectors.concept_drift.base import BaseConceptDrift
from frouros.detectors.data_drift.batch import (
    EMD,
    JS,
    KL,
    MMD,
    PSI,
    AndersonDarlingTest,
    BhattacharyyaDistance,
    BWSTest,
    CVMTest,
    EnergyDistance,
    HellingerDistance,
    HINormalizedComplement,
    KSTest,
    KuiperTest,
    MannWhitneyUTest,
    WelchTTest,
)
from frouros.detectors.data_drift.batch.base import BaseDataDriftBatch


@pytest.mark.parametrize(
    "detector_class, expected_distance, expected_p_value",
    [
        (BhattacharyyaDistance, 0.81004188, 0.0),
        (EMD, 3.85346006, 0.0),
        (EnergyDistance, 2.11059982, 0.0),
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
    """Test batch permutation test on data drift callback.

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
                method="estimate",
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
    "method, expected_p_value",
    [
        ("auto", 0.009900490107343236),
        ("conservative", 0.009900990099009901),
        ("exact", 0.009900490107343236),
        ("approximate", 0.009900990098759907),
        ("estimate", 0.0),
    ],
)
def test_batch_permutation_test_method(
    X_ref_univariate: np.ndarray,  # noqa: N803
    X_test_univariate: np.ndarray,
    method: str,
    expected_p_value: float,
) -> None:
    """Test batch permutation test on data drift callback using method.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param method: method
    :type method: str
    :param expected_p_value: expected p-value value
    :type expected_p_value: float
    """
    np.random.seed(seed=31)

    permutation_test_name = "permutation_test"
    detector = MMD(
        callbacks=[
            PermutationTestDistanceBased(
                num_permutations=100,
                method=method,
                random_state=31,
                num_jobs=-1,
                name=permutation_test_name,
            )
        ]
    )
    _ = detector.fit(X=X_ref_univariate)
    _, callback_logs = detector.compare(X=X_test_univariate)

    assert np.isclose(
        callback_logs[permutation_test_name]["p_value"],
        expected_p_value,
    )


@pytest.mark.parametrize(
    "detector_class",
    [
        AndersonDarlingTest,
        BWSTest,
        CVMTest,
        KSTest,
        KuiperTest,
        MannWhitneyUTest,
        WelchTTest,
    ],
)
def test_batch_reset_on_statistical_test_data_drift(
    X_ref_univariate: np.ndarray,  # noqa: N803
    X_test_univariate: np.ndarray,
    detector_class: BaseDataDriftBatch,
    mocker: Any,
) -> None:
    """Test batch reset on statistical test data drift callback.

    :param X_ref_univariate: reference univariate data
    :type X_ref_univariate: numpy.ndarray
    :param X_test_univariate: test univariate data
    :type X_test_univariate: numpy.ndarray
    :param detector_class: detector distance
    :type detector_class: BaseDataDriftBatch
    :param mocker: mocker
    :type mocker: Any
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
    detector.reset.assert_called_once()  # pylint: disable=no-member


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
    model_errors: list[int],
    detector_class: BaseConceptDrift,
) -> None:
    """Test streaming history on concept drift callback.

    :param model_errors: model errors
    :type model_errors: list[int]
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
