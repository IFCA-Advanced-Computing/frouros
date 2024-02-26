"""Test persistence module."""

import pickle

import pytest  # type: ignore

from frouros.callbacks import HistoryConceptDrift, PermutationTestDistanceBased
from frouros.callbacks.base import BaseCallback
from frouros.detectors.base import BaseDetector
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.detectors.data_drift import MMD
from frouros.utils import load, save


@pytest.fixture(
    scope="module",
    params=[
        DDM(
            config=DDMConfig(),
        ),
        MMD(),
    ],
)
def detector(
    request: pytest.FixtureRequest,
) -> BaseDetector:
    """Fixture for detector.

    :param request: Request
    :type request: pytest.FixtureRequest
    :return: Detector
    :rtype: BaseDetector
    """
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        HistoryConceptDrift(),
        PermutationTestDistanceBased(
            num_permutations=2,
        ),
    ],
)
def callback(
    request: pytest.FixtureRequest,
) -> BaseCallback:
    """Fixture for callback.

    :param request: Request
    :type request: pytest.FixtureRequest
    :return: Callback
    :rtype: BaseCallback
    """
    return request.param


def test_save_load_with_valid_detector(
    detector: BaseDetector,
) -> None:
    """Test save and load with valid detector.

    :param detector: Detector
    :type detector: BaseDetector
    """
    filename = "/tmp/detector.pkl"
    save(detector, filename)
    loaded_detector = load(filename)
    assert isinstance(loaded_detector, detector.__class__)


def test_save_load_with_valid_callback(
    callback: BaseCallback,
) -> None:
    """Test save and load with valid callback.

    :param callback: Callback
    :type callback: BaseCallback
    """
    filename = "/tmp/callback.pkl"
    save(callback, filename)
    loaded_callback = load(filename)
    assert isinstance(loaded_callback, BaseCallback)


def test_save_with_invalid_object() -> None:
    """Test save with invalid object.

    :raises TypeError: Type error exception
    """
    invalid_object = "invalid"
    filename = "/tmp/invalid.pkl"
    with pytest.raises(TypeError):
        save(invalid_object, filename)


def test_save_with_invalid_protocol(
    detector: BaseDetector,
) -> None:
    """Test save with invalid protocol.

    :param detector: Detector
    :type detector: BaseDetector
    :raises ValueError: Value error exception
    """
    filename = "/tmp/detector.pkl"
    invalid_protocol = pickle.HIGHEST_PROTOCOL + 1
    with pytest.raises(ValueError):
        save(detector, filename, invalid_protocol)


def test_load_with_non_existent_file() -> None:
    """Test load with non-existent file.

    :raises FileNotFoundError: File not found error exception
    """
    filename = "/tmp/non_existent.pkl"
    with pytest.raises(FileNotFoundError):
        load(filename)
