"""Test stats module."""

from typing import List, Union

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.utils.stats import CircularMean, Mean


@pytest.mark.parametrize(
    "size, values, expected_mean_steps",
    [
        (1, [5, 10, 6, 4, 14], [5.0, 10.0, 6.0, 4.0, 14.0]),
        (3, [5, 10, 6, 4, 14], [5.0, 7.5, 7.0, 6.66666667, 8.0]),
        (6, [5, 10, 6, 4, 14], [5.0, 7.5, 7, 6.25, 7.8]),
    ],
)
def test_circular_mean(
    size: int,
    values: List[Union[int, float]],
    expected_mean_steps: List[Union[int, float]],
) -> None:
    """Test circular mean.

    :param size: size value
    :type size: int
    :param values: values
    :type values: List[Union[int, float]]
    :param expected_mean_steps: expected mean step values
    :type expected_mean_steps: List[Union[int, float]]
    """
    mean = CircularMean(size=size)

    for value, expected_mean_step in zip(values, expected_mean_steps):
        mean.update(value=value)
        assert np.isclose(mean.get(), expected_mean_step)


@pytest.mark.parametrize(
    "values, expected_mean_steps",
    [
        ([5, 10, 6, 4, 14], [5.0, 7.5, 7, 6.25, 7.8]),
        ([-5, 10, -6, 4, -14], [-5.0, 2.5, -0.33333334, 0.75, -2.2]),
    ],
)
def test_mean(
    values: List[Union[int, float]],
    expected_mean_steps: List[Union[int, float]],
) -> None:
    """Test mean.

    :param values: values
    :type values: List[Union[int, float]]
    :param expected_mean_steps: expected mean step values
    :type expected_mean_steps: List[Union[int, float]]
    """
    mean = Mean()

    for value, expected_mean_step in zip(values, expected_mean_steps):
        mean.update(value=value)
        assert np.isclose(mean.get(), expected_mean_step)
