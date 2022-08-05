"""Synthetic datasets module."""

from typing import Optional, Tuple, Iterator

import numpy as np  # type: ignore

from frouros.datasets.base import Generator
from frouros.datasets.exceptions import InvalidBlockError


class SEA(Generator):
    """SEA generator class."""

    block_map = {1: 8.0, 2: 9.0, 3: 7.0, 4: 9.5}

    def __init__(self, seed: Optional[int] = None) -> None:
        """Init method.

        :param seed: seed value
        :type seed: Optional[int]
        """
        try:
            np.random.seed(seed=seed)
        except (TypeError, ValueError) as e:
            raise e

    @staticmethod
    def _generate_sample(threshold: float, noise: float) -> Tuple[np.ndarray, int]:
        X = np.random.uniform(low=0.0, high=10.0, size=(3,))  # noqa: N806
        if np.random.random() < noise:
            y = np.random.randint(2)
        else:
            y = 1 if X[0] + X[1] <= threshold else 0
        return X, y

    def generate_dataset(
        self, block: int, noise: float = 0.1, num_samples: int = 12500
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Generate dataset.

        :param block: block to generate samples from
        :type block: int
        :param noise: ratio of samples with a noisy class
        :type noise: float
        :param num_samples: number of samples to generate
        :type num_samples: int
        :return: generator with the samples
        :rtype: Iterator[Tuple[np.ndarray, int]]
        """
        try:
            threshold = self.block_map[block]
        except KeyError as e:
            raise InvalidBlockError("block must be 1, 2, 3 or 4.") from e
        if num_samples < 1:
            raise ValueError("num_samples must be greater than 0.")
        if not 0 <= noise <= 1:
            raise ValueError("noise must be in the range [0, 1].")
        dataset = (
            self._generate_sample(threshold=threshold, noise=noise)
            for _ in range(num_samples)
        )
        return dataset


class Dummy(Generator):
    """Dummy generator class."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Init method.

        :param seed: seed value
        :type seed: Optional[int]
        """
        try:
            np.random.seed(seed=seed)
        except (TypeError, ValueError) as e:
            raise e

    @staticmethod
    def _generate_sample(class_: int) -> Tuple[np.ndarray, int]:
        X = np.random.uniform(low=0.0, high=10.0, size=(2,))  # noqa: N806
        y = class_ if X[0] + X[1] < 10.0 else 1 - class_
        return X, y

    def generate_dataset(
        self, class_: int = 1, num_samples: int = 12500
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Generate dataset.

        :param class_: class value
        :type class_: int
        :param num_samples: number of samples to generate
        :type num_samples: int
        :return: generator with the samples
        :rtype: Iterator[Tuple[np.ndarray, int]]
        """
        if class_ not in [1, 0]:
            raise ValueError("class_ must be 1 or 0.")
        if num_samples < 1:
            raise ValueError("num_samples must be greater than 0.")
        dataset = (self._generate_sample(class_=class_) for _ in range(num_samples))
        return dataset
