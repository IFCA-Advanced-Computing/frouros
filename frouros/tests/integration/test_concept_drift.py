"""Test concept drift detectors."""

from typing import Callable, List, Tuple

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.detectors.concept_drift import ADWIN, ADWINConfig, KSWIN, KSWINConfig
from frouros.detectors.concept_drift import (
    BOCD,
    BOCDConfig,
    CUSUM,
    CUSUMConfig,
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
    PageHinkley,
    PageHinkleyConfig,
)
from frouros.detectors.concept_drift import (
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
from frouros.detectors.concept_drift.base import BaseConceptDrift
from frouros.detectors.concept_drift.streaming.change_detection.base import (
    BaseChangeDetection,
)
from frouros.detectors.concept_drift.streaming.change_detection.bocd import (
    GaussianUnknownMean,
)

MIN_NUM_INSTANCES = 30
BOCD_ARGS = {
    "model": GaussianUnknownMean(
        prior_mean=0,
        prior_var=1,
        data_var=0.5,
    ),
    "hazard": 0.01,
}
CUMSUM_ARGS = {
    "delta": 0.005,
    "lambda_": 50,
}
GEOMETRIC_MOVING_AVERAGE_ARGS = {
    "alpha": 0.99,
}
PAGE_HINKLEY_ARGS = {
    "alpha": 0.9999,
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
    return int(1 - y_true == y_pred)


detectors = [
    (
        BOCD(
            config=BOCDConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                **BOCD_ARGS,  # type: ignore
            ),
        ),
        error_scorer,
    ),
    (
        CUSUM(
            config=CUSUMConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                **CUMSUM_ARGS,
            ),
        ),
        error_scorer,
    ),
    (
        GeometricMovingAverage(
            config=GeometricMovingAverageConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                **GEOMETRIC_MOVING_AVERAGE_ARGS,
            ),
        ),
        error_scorer,
    ),
    (
        PageHinkley(
            config=PageHinkleyConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                **PAGE_HINKLEY_ARGS,
            ),
        ),
        error_scorer,
    ),
    (
        ADWIN(
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
            config=KSWINConfig(
                alpha=0.005,
                seed=31,
                min_num_instances=100,
                num_test_instances=30,
            ),
        ),
        error_scorer,
    ),
    (
        DDM(
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
            config=STEPDConfig(
                alpha_d=0.003,
                alpha_w=0.05,
                min_num_instances=MIN_NUM_INSTANCES,
            ),
        ),
        lambda y_true, y_pred: y_true == y_pred,
    ),
]


@pytest.mark.parametrize("detector_info", detectors)
def test_streaming_detector_normal(
    clf_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    train_prediction_normal: np.ndarray,
    detector_info: Tuple[BaseConceptDrift, Callable],
) -> None:
    """Test streaming detector.

    :param clf_dataset: dataset generated using SEA
    :type clf_dataset: Tuple[numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray]
    :param train_prediction_normal: test prediction values
    :type train_prediction_normal: numpy.ndarray
    :param detector_info: concept drift detector and value function
    :type detector_info: Tuple[BaseConceptDrift, Callable]
    """
    _, _, _, y_test = clf_dataset
    y_pred = train_prediction_normal  # noqa: N806
    detector, value_func = detector_info

    for y_sample_pred, y_sample in zip(y_pred, y_test):  # noqa: N806
        value_score = value_func(y_true=y_sample, y_pred=y_sample_pred)
        detector.update(value=value_score)


CHANGE_DETECTION_MIN_NUM_INSTANCES = 1
change_detection_detectors = [
    (
        BOCD(
            config=BOCDConfig(
                min_num_instances=CHANGE_DETECTION_MIN_NUM_INSTANCES,
                **BOCD_ARGS,  # type: ignore
            ),
        ),
        [100, 203],
    ),
    (
        CUSUM(
            config=CUSUMConfig(
                min_num_instances=CHANGE_DETECTION_MIN_NUM_INSTANCES,
                **CUMSUM_ARGS,
            ),
        ),
        [113, 226],
    ),
    (
        GeometricMovingAverage(
            config=GeometricMovingAverageConfig(
                min_num_instances=CHANGE_DETECTION_MIN_NUM_INSTANCES,
                **GEOMETRIC_MOVING_AVERAGE_ARGS,
            ),
        ),
        [134],
    ),
    (
        PageHinkley(
            config=PageHinkleyConfig(
                min_num_instances=CHANGE_DETECTION_MIN_NUM_INSTANCES,
                **PAGE_HINKLEY_ARGS,
            ),
        ),
        [113, 227],
    ),
]


@pytest.mark.parametrize("detector_info", change_detection_detectors)
def test_streaming_change_detection_detector(
    stream_drift: np.ndarray,
    detector_info: Tuple[BaseChangeDetection, List[int]],
) -> None:
    """Test streaming change detection detector.

    :param stream_drift: stream with drift
    :type stream_drift: numpy.ndarray
    :param detector_info: change detection detector and list of expected drift indices
    :type detector_info: Tuple[BaseChangeDetection, List[int]]
    """
    detector, idx_drifts = detector_info
    idx_detected_drifts = []
    for i, val in enumerate(stream_drift):
        detector.update(value=val)
        if detector.status["drift"]:
            detector.reset()
            idx_detected_drifts.append(i)

    assert idx_detected_drifts == idx_drifts
