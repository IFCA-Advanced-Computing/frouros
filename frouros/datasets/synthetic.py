"""Synthetic datasets module."""

from typing import Iterator, Optional, Tuple

import numpy as np

from frouros.datasets.base import BaseDatasetGenerator
from frouros.datasets.exceptions import InvalidBlockError


class SEA(BaseDatasetGenerator):
    """SEA generator [street2001streaming]_.

    :param seed: seed value, defaults to None
    :type seed: Optional[int]

    :References:

    .. [street2001streaming] Street, W. Nick, and YongSeog Kim.
        "A streaming ensemble algorithm (SEA) for large-scale classification."
        Proceedings of the seventh ACM SIGKDD international conference on Knowledge
        discovery and data mining. 2001.

    :Example:

    >>> from frouros.datasets.synthetic import SEA
    >>> sea = SEA(seed=31)
    >>> dataset = sea.generate_dataset(block=1, noise=0.1, num_samples=5)
    >>> for X, y in dataset:
    ...     print(X, y)
    [2.86053822 9.58105567 7.70312932] 0
    [2.08165462 1.36917049 9.08373802] 0
    [8.36483632 1.12172604 8.3489916 ] 0
    [2.44680795 1.36231348 7.22094455] 1
    [1.28477715 2.20364007 5.19211202] 1
    """

    def __init__(  # noqa: D107
        self,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            seed=seed,
        )
        self._block_map = {1: 8.0, 2: 9.0, 3: 7.0, 4: 9.5}

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

        :param block: block to generate samples from, must be 1, 2, 3 or 4
        :type block: int
        :param noise: ratio of samples with a noisy class, defaults to 0.1
        :type noise: float
        :param num_samples: number of samples to generate, defaults to 12500
        :type num_samples: int
        :return: generator with the samples
        :rtype: Iterator[Tuple[np.ndarray, int]]
        """
        try:
            threshold = self._block_map[block]
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


class Dummy(BaseDatasetGenerator):
    """Dummy generator class."""

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
