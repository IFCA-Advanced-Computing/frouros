"""Test concept drift detectors."""

from typing import Callable, Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore

from frouros.concept_drift.base import ConceptDriftBase
from frouros.concept_drift.cusum_based import (
    CUSUM,
    CUSUMConfig,
    GeometricMovingAverage,
    GeometricMovingAverageConfig,
    PageHinkley,
    PageHinkleyConfig,
)
from frouros.concept_drift.ddm_based import (
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
from frouros.concept_drift.window_based import ADWIN, ADWINConfig, KSWIN, KSWINConfig


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
    return int(1 - y_true == y_pred)


detectors = [
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
                alpha=0.99,
            ),
        ),
        error_scorer,
    ),
    (
        PageHinkley(
            config=PageHinkleyConfig(
                min_num_instances=MIN_NUM_INSTANCES,
                alpha=0.9999,
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
    classification_dataset: Tuple[np.array, np.array, np.array, np.array],
    train_prediction_normal,
    detector_info: Tuple[ConceptDriftBase, Callable],
) -> None:
    """Test streaming detector.

    :param classification_dataset: dataset generated using SEA
    :type classification_dataset: Tuple[numpy.array, numpy.array,
    numpy.array, numpy.array]
    :param detector_info: concept drift detector and value function
    :type detector_info: Tuple[ConceptDriftBase, Callable]
    """
    _, _, _, y_test = classification_dataset
    pipeline, y_pred = train_prediction_normal  # noqa: N806
    detector, value_func = detector_info

    for y_sample_pred, y_sample in zip(y_pred, y_test):  # noqa: N806
        value_score = value_func(y_true=y_sample, y_pred=y_sample_pred)
        detector.update(value=value_score)
