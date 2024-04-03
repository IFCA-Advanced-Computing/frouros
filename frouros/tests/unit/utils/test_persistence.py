"""Test persistence module."""

import pickle

import pytest

from frouros.callbacks import HistoryConceptDrift, PermutationTestDistanceBased
from frouros.callbacks.base import BaseCallback
from frouros.detectors.base import BaseDetector
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.detectors.data_drift import MMD  # type: ignore
from frouros.utils import load, save
from frouros.utils.decorators import set_os_filename


@pytest.fixture(
    scope="session",
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
    scope="session",
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


@set_os_filename("detector.pkl")
def test_save_load_with_valid_detector(
    request: pytest.FixtureRequest,
    detector: BaseDetector,
) -> None:
    """Test save and load with valid detector.

    :param request: Request
    :type request: pytest.FixtureRequest
    :param detector: Detector
    :type detector: BaseDetector
    """
    filename = request.node.get_closest_marker("filename").args[0]
    save(
        obj=detector,
        filename=filename,
    )
    loaded_detector = load(
        filename=filename,
    )
    assert isinstance(loaded_detector, detector.__class__)


@set_os_filename("callback.pkl")
def test_save_load_with_valid_callback(
    request: pytest.FixtureRequest,
    callback: BaseCallback,
) -> None:
    """Test save and load with valid callback.

    :param request: Request
    :type request: pytest.FixtureRequest
    :param callback: Callback
    :type callback: BaseCallback
    """
    filename = request.node.get_closest_marker("filename").args[0]
    save(
        obj=callback,
        filename=filename,
    )
    loaded_callback = load(
        filename=filename,
    )
    assert isinstance(loaded_callback, BaseCallback)


@set_os_filename("invalid.pkl")
def test_save_with_invalid_object(
    request: pytest.FixtureRequest,
) -> None:
    """Test save with invalid object.

    :param request: Request
    :type request: pytest.FixtureRequest
    :raises TypeError: Type error exception
    """
    invalid_object = "invalid"
    filename = request.node.get_closest_marker("filename").args[0]
    with pytest.raises(TypeError):
        save(invalid_object, filename)


@set_os_filename("detector.pkl")
def test_save_with_invalid_protocol(
    request: pytest.FixtureRequest,
    detector: BaseDetector,
) -> None:
    """Test save with invalid protocol.

    :param request: Request
    :type request: pytest.FixtureRequest
    :param detector: Detector
    :type detector: BaseDetector
    :raises ValueError: Value error exception
    """
    invalid_protocol = pickle.HIGHEST_PROTOCOL + 1
    filename = request.node.get_closest_marker("filename").args[0]
    with pytest.raises(ValueError):
        save(detector, filename, invalid_protocol)


@set_os_filename("non_existent.pkl")
def test_load_with_non_existent_file(
    request: pytest.FixtureRequest,
) -> None:
    """Test load with non-existent file.

    :param request: Request
    :type request: pytest.FixtureRequest
    :raises FileNotFoundError: File not found error exception
    """
    filename = request.node.get_closest_marker("filename").args[0]
    with pytest.raises(FileNotFoundError):
        load(filename)
