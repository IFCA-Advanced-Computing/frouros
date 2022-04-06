"""Configuration file for the tests."""

import pytest  # type: ignore
import numpy as np  # type: ignore

from frouros.datasets.real import Elec2


# Elec2 fixtures
@pytest.fixture(scope="function")
def elec2_raw() -> Elec2:
    """Elec2 raw dataset.

    :return: Elec2 raw dataset
    :rtype: Elec2
    """
    dataset = Elec2()
    dataset.download()
    return dataset


@pytest.fixture(scope="function")
def elec2(elec2_raw: Elec2) -> np.ndarray:  # pylint: disable=redefined-outer-name
    """Elec2 dataset.

    :param elec2_raw: Elec2 raw dataset
    :type elec2_raw: Elec2
    :return: Elec2 dataset
    :rtype: np.ndarray
    """
    return elec2_raw.load()
