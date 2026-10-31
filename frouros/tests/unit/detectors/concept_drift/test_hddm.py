"""Test HDDM-A and HDDM-W detectors.

Statistical thresholds were calibrated over 100 seeds (blocks of 10 seeds) and
are documented next to each assertion. Streams are generated with
numpy.random.default_rng, whose bit stream is stable across numpy versions.
"""

from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import pytest

from frouros.detectors.concept_drift import (
    HDDMA,
    HDDMW,
    HDDMAConfig,
    HDDMWConfig,
)
from frouros.detectors.concept_drift.streaming.statistical_process_control.hddm import (  # noqa: E501
    HoeffdingOneSidedTest,
    HoeffdingTwoSidedTest,
    McDiarmidOneSidedTest,
    McDiarmidTwoSidedTest,
    SampleInfo,
)

HDDM = Union[HDDMA, HDDMW]
SEEDS = range(10)
CHANGE_POINT = 1500
DETECTION_WINDOW = 500


def hddma(two_sided_test: bool) -> HDDMA:
    """Create HDDM-A detector with default configuration.

    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    :return: detector
    :rtype: HDDMA
    """
    return HDDMA(config=HDDMAConfig(two_sided_test=two_sided_test))


def hddmw(two_sided_test: bool) -> HDDMW:
    """Create HDDM-W detector with default configuration.

    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    :return: detector
    :rtype: HDDMW
    """
    return HDDMW(config=HDDMWConfig(two_sided_test=two_sided_test))


DETECTORS = [
    pytest.param(hddma, id="HDDMA"),
    pytest.param(hddmw, id="HDDMW"),
]
TWO_SIDED = [
    pytest.param(False, id="one-sided"),
    pytest.param(True, id="two-sided"),
]


def bernoulli_stream(seed: int, *probabilities: float) -> npt.NDArray[np.int_]:
    """Generate a stream of errors with a change in error rate every CHANGE_POINT.

    :param seed: random seed
    :type seed: int
    :param probabilities: error rate of each segment
    :type probabilities: float
    :return: stream of 0 (correct) and 1 (error) values
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [(rng.random(CHANGE_POINT) < p).astype(int) for p in probabilities]
    )


def drift_steps(detector: HDDM, stream: npt.NDArray[np.int_]) -> list[int]:
    """Feed a stream and return the steps where drift is detected.

    The detector is reset after each detection, as recommended for Frouros
    concept drift detectors.

    :param detector: detector
    :type detector: Union[HDDMA, HDDMW]
    :param stream: stream of values
    :type stream: numpy.ndarray
    :return: steps where drift is detected
    :rtype: list[int]
    """
    steps = []
    for i, value in enumerate(stream):
        detector.update(value=int(value))
        if detector.drift:
            steps.append(i)
            detector.reset()
    return steps


def flags(detector: HDDM, stream: npt.NDArray[np.int_]) -> list[tuple[bool, bool]]:
    """Feed a stream and return the (drift, warning) flags of every step.

    :param detector: detector
    :type detector: Union[HDDMA, HDDMW]
    :param stream: stream of values
    :type stream: numpy.ndarray
    :return: drift and warning flags per step
    :rtype: list[tuple[bool, bool]]
    """
    result = []
    for value in stream:
        detector.update(value=int(value))
        result.append((detector.drift, detector.warning))
    return result


# A. Boolean correctness


def test_mcdiarmid_two_sided_no_change_returns_false_flags() -> None:
    """Test that a two-sided McDiarmid test without change returns no flags.

    The drift flag of the decrease test used to be a one-element tuple, which
    is always truthy, so drift was reported on every call.
    """
    test = McDiarmidTwoSidedTest(alpha_d=0.001, alpha_w=0.005, lambda_=0.05)
    for _ in range(5):
        test.update_stats(value=0, alpha=0.001)

    drift, warning = test.check_changes()

    assert isinstance(drift, (bool, np.bool_)) and not drift
    assert isinstance(warning, (bool, np.bool_)) and not warning


@pytest.mark.parametrize("two_sided_test", TWO_SIDED)
@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_status_flags_are_booleans(
    detector_factory: Callable[[bool], HDDM],
    two_sided_test: bool,
) -> None:
    """Test that drift and warning flags are booleans and never both set.

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    """
    stream = bernoulli_stream(0, 0.1, 0.4, 0.05)
    for drift, warning in flags(detector_factory(two_sided_test), stream):
        assert isinstance(drift, (bool, np.bool_))
        assert isinstance(warning, (bool, np.bool_))
        assert not (drift and warning)


# B. Stationary data


@pytest.mark.parametrize("two_sided_test", TWO_SIDED)
@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_stationary_stream_false_alarms(
    detector_factory: Callable[[bool], HDDM],
    two_sided_test: bool,
) -> None:
    """Test that a stationary stream produces few false alarms.

    HDDM bounds hold for each individual test, but the test is repeated at
    every step, so a small number of false alarms is expected. Calibration
    over 100 seeds (error rate 0.2, 3000 steps) gave at most 4 alarms per
    block of 10 seeds for every configuration (the previous two-sided
    implementation gave 675-1000).

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    """
    num_alarms = sum(
        len(drift_steps(detector_factory(two_sided_test), stream))
        for stream in (bernoulli_stream(seed, 0.2, 0.2) for seed in SEEDS)
    )

    assert num_alarms <= 6


# C. Genuine drift (error rate increase)


@pytest.mark.parametrize("two_sided_test", TWO_SIDED)
@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_increase_detected_after_change_point(
    detector_factory: Callable[[bool], HDDM],
    two_sided_test: bool,
) -> None:
    """Test that an error rate increase is detected only after it happens.

    The first alarm must fall inside the detection window: a detector that
    alarms all the time would fail. Calibration: 100/100 seeds for every
    configuration (error rate 0.1 -> 0.4).

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    """
    for seed in SEEDS:
        steps = drift_steps(
            detector_factory(two_sided_test), bernoulli_stream(seed, 0.1, 0.4)
        )
        assert steps
        assert CHANGE_POINT <= steps[0] < CHANGE_POINT + DETECTION_WINDOW


# D. One-sided behaviour


@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_one_sided_ignores_decrease(
    detector_factory: Callable[[bool], HDDM],
) -> None:
    """Test that the one-sided test does not alarm on an error rate decrease.

    Calibration: no post-change alarm in 100/100 seeds (0.4 -> 0.1).

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    """
    for seed in SEEDS:
        steps = drift_steps(detector_factory(False), bernoulli_stream(seed, 0.4, 0.1))
        assert not any(step >= CHANGE_POINT for step in steps)


# E. Two-sided behaviour


@pytest.mark.parametrize(
    "detector_factory, min_detected, max_pre_change_alarms",
    [
        # Calibration per block of 10 seeds: detected 9-10, pre-change 0-4
        # (previous implementation: 312-396 pre-change alarms)
        pytest.param(hddma, 8, 6, id="HDDMA"),
        # Calibration per block of 10 seeds: detected 7-10, pre-change 7-18
        # (previous implementation: 500 pre-change alarms). HDDM-W also has
        # 6-13 pre-change alarms in one-sided mode at this error rate.
        pytest.param(hddmw, 6, 25, id="HDDMW"),
    ],
)
def test_two_sided_detects_decrease(
    detector_factory: Callable[[bool], HDDM],
    min_detected: int,
    max_pre_change_alarms: int,
) -> None:
    """Test that the two-sided test detects an error rate decrease.

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param min_detected: minimum number of seeds where the decrease is detected
    :type min_detected: int
    :param max_pre_change_alarms: maximum number of alarms before the change
    :type max_pre_change_alarms: int
    """
    detected, pre_change_alarms = 0, 0
    for seed in SEEDS:
        steps = drift_steps(detector_factory(True), bernoulli_stream(seed, 0.4, 0.1))
        pre_change_alarms += sum(step < CHANGE_POINT for step in steps)
        detected += any(
            CHANGE_POINT <= step < CHANGE_POINT + DETECTION_WINDOW for step in steps
        )

    assert detected >= min_detected
    assert pre_change_alarms <= max_pre_change_alarms


@pytest.mark.parametrize("two_sided_test", TWO_SIDED)
@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_warning_precedes_drift_on_gradual_increase(
    detector_factory: Callable[[bool], HDDM],
    two_sided_test: bool,
) -> None:
    """Test the in-control -> warning -> drift transition.

    Calibration: warning before the first drift in 100/100 seeds for every
    configuration (previous two-sided implementation: 0/100).

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    """
    probabilities = np.concatenate(
        [np.full(1000, 0.1), np.linspace(0.1, 0.6, 1000), np.full(1000, 0.6)]
    )
    for seed in SEEDS:
        stream = (np.random.default_rng(seed).random(3000) < probabilities).astype(int)
        steps = flags(detector_factory(two_sided_test), stream)
        first_drift = next(i for i, (drift, _) in enumerate(steps) if drift)
        assert any(warning for _, warning in steps[:first_drift])


def test_hoeffding_tests_no_decision_without_second_sample() -> None:
    """Test that no decision is made when the cut point is the whole sequence.

    With m = 0 the Hoeffding threshold is 0 and the comparison 0 >= 0 used to
    signal drift.
    """
    one_sided = HoeffdingOneSidedTest(alpha_d=0.001, alpha_w=0.005)
    two_sided = HoeffdingTwoSidedTest(alpha_d=0.001, alpha_w=0.005)
    for test in (one_sided, two_sided):
        for value in [1, 0, 0, 0, 0, 0]:
            test.z.update(value=value)
            test.set_initial_cut_mean()
            test.update_cut_point(
                epsilon_z=test.hoeffding_error_bound(num_values=test.z.num_values)
            )
        assert test.x.num_values == test.z.num_values
        assert test.check_cases() == (False, False)


def test_mcdiarmid_decrease_uses_decrease_samples() -> None:
    """Test that the decrease check compares the decrease monitoring samples."""
    test = McDiarmidTwoSidedTest(alpha_d=0.001, alpha_w=0.005, lambda_=0.05)
    for value in [1] * 200 + [0] * 200:
        test.update_stats(value=value, alpha=0.001)

    assert test._check_mean_decrease(alpha=0.001)


def test_mcdiarmid_empty_sample_is_not_compared() -> None:
    """Test that an empty sample never triggers a change.

    An empty sample has an EWMA of 0, which would be compared as a real mean
    when the significance level is large enough to make the bound smaller
    than 1.
    """
    test = McDiarmidOneSidedTest(alpha_d=0.001, alpha_w=0.9, lambda_=0.05)
    filled, empty = SampleInfo(lambda_=0.05), SampleInfo(lambda_=0.05)
    for _ in range(100):
        filled.update(value=1)

    assert not test._check_threshold(sample_1=empty, sample_2=filled, alpha=0.9)
    assert not test._check_threshold(sample_1=filled, sample_2=empty, alpha=0.9)


# F. Reset behaviour


@pytest.mark.parametrize("two_sided_test", TWO_SIDED)
@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_reset_equivalent_to_new_detector(
    detector_factory: Callable[[bool], HDDM],
    two_sided_test: bool,
) -> None:
    """Test that a reset detector behaves exactly like a new one.

    The warm-up has a low error rate, so statistics kept after reset would
    make the following higher error rate look like an increase.

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    """
    warm_up = bernoulli_stream(1, 0.05)
    stream = bernoulli_stream(2, 0.3)
    detector = detector_factory(two_sided_test)
    for value in warm_up:
        detector.update(value=int(value))

    detector.reset()

    assert detector.num_instances == 0
    assert not detector.drift and not detector.warning
    assert flags(detector, stream) == flags(detector_factory(two_sided_test), stream)


# G. Edge cases


@pytest.mark.parametrize("two_sided_test", TWO_SIDED)
@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_no_decision_before_min_num_instances(
    detector_factory: Callable[[bool], HDDM],
    two_sided_test: bool,
) -> None:
    """Test that a stream shorter than min_num_instances raises no flags.

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    """
    detector = detector_factory(two_sided_test)
    stream = np.r_[np.zeros(15, dtype=int), np.ones(14, dtype=int)]
    assert len(stream) < detector.config.min_num_instances
    assert flags(detector, stream) == [(False, False)] * len(stream)


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("two_sided_test", TWO_SIDED)
@pytest.mark.parametrize("detector_factory", DETECTORS)
def test_constant_stream_raises_no_flags(
    detector_factory: Callable[[bool], HDDM],
    two_sided_test: bool,
    value: int,
) -> None:
    """Test that a constant stream never raises drift or warning.

    :param detector_factory: detector factory
    :type detector_factory: Callable[[bool], Union[HDDMA, HDDMW]]
    :param two_sided_test: whether to use the two-sided test
    :type two_sided_test: bool
    :param value: constant value of the stream
    :type value: int
    """
    stream = np.full(2000, value)
    assert flags(detector_factory(two_sided_test), stream) == [(False, False)] * 2000


@pytest.mark.parametrize(
    "config_class, kwargs",
    [
        (HDDMAConfig, {"alpha_d": 0.0}),
        (HDDMAConfig, {"alpha_d": 0.01, "alpha_w": 0.005}),
        (HDDMAConfig, {"alpha_w": 1.5}),
        (HDDMAConfig, {"two_sided_test": 1}),
        (HDDMWConfig, {"lambda_": 1.5}),
        (HDDMWConfig, {"alpha_d": 0.01, "alpha_w": 0.01}),
    ],
)
def test_invalid_config_raises_value_error(
    config_class: Union[type[HDDMAConfig], type[HDDMWConfig]],
    kwargs: dict[str, Union[float, int]],
) -> None:
    """Test that invalid configuration values are rejected.

    :param config_class: configuration class
    :type config_class: Union[type[HDDMAConfig], type[HDDMWConfig]]
    :param kwargs: invalid configuration arguments
    :type kwargs: dict[str, Union[float, int]]
    """
    with pytest.raises(ValueError):
        config_class(**kwargs)  # type: ignore[arg-type]
