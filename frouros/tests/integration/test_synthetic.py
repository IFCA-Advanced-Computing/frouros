"""Test synthetic datasets module."""

from typing import Any

import pytest  # type: ignore

from frouros.datasets.exceptions import InvalidBlockError
from frouros.datasets.synthetic import SEA


# SEA tests
@pytest.mark.parametrize("seed", [-1, "a"])
def test_sea_invalid_seed_error(seed: Any) -> None:
    """Test SEA invalid seed error.

    :param seed: seed value
    :type seed: Any
    """
    with pytest.raises((TypeError, ValueError)):
        _ = SEA(seed=seed)


@pytest.mark.parametrize("block", [0, 5])
def test_sea_invalid_block_error(sea: SEA, block: int) -> None:
    """Test SEA invalid block error.

    :param sea: SEA generator
    :type sea: SEA
    :param block: block to generate samples from
    :type block: int
    """
    with pytest.raises(InvalidBlockError):
        sea.generate_dataset(block=block)


@pytest.mark.parametrize("noise", [-0.1, 1.1])
def test_sea_invalid_noise_error(sea: SEA, noise: float) -> None:
    """Test SEA invalid noise error.

    :param sea: SEA generator
    :type sea: SEA
    :param noise: ratio of samples with a noisy class
    :type noise: float
    """
    with pytest.raises(ValueError):
        sea.generate_dataset(block=1, noise=noise)


@pytest.mark.parametrize("num_samples", [-1, 0])
def test_sea_invalid_num_samples_error(sea: SEA, num_samples: int) -> None:
    """Test SEA invalid number of samples error.

    :param sea: SEA generator
    :type sea: SEA
    :param num_samples: number of samples to generate
    :type num_samples: int
    """
    with pytest.raises(ValueError):
        sea.generate_dataset(block=1, num_samples=num_samples)
