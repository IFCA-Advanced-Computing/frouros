"""Test change detection detectors."""

from typing import Tuple, List

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.detectors.concept_drift import (
    BOCD,
    BOCDConfig,
)
from frouros.detectors.concept_drift.streaming.change_detection.base import (
    BaseChangeDetection,
)
from frouros.detectors.concept_drift.streaming.change_detection.bocd import (
    GaussianUnknownMean,
)

detectors = [
    (
        BOCD(
            config=BOCDConfig(
                min_num_instances=1,
                model=GaussianUnknownMean(
                    prior_mean=0,
                    prior_var=1,
                    data_var=0.5,
                ),
                hazard=0.01,
            ),
        ),
        [100, 203],
    ),
]


@pytest.mark.parametrize("detector_info", detectors)
def test_change_detection_detector(
    stream_drift: np.ndarray,
    detector_info: Tuple[BaseChangeDetection, List[int]],
) -> None:
    """Test streaming detector.

    :param stream_drift: stream with drift
    :type stream_drift: numpy.ndarray
    :param detector_info: concept drift detector and value function
    :type detector_info: Tuple[BaseConceptDrift, Callable]
    """
    detector, idx_drifts = detector_info
    idx_detected_drifts = []
    for i, val in enumerate(stream_drift):
        detector.update(value=val)
        if detector.status["drift"]:
            detector.reset()
            idx_detected_drifts.append(i)

    assert idx_detected_drifts == idx_drifts
