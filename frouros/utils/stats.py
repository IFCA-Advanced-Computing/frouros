"""Stats module."""

import abc
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, Optional, List, Union

import numpy as np  # type: ignore


class Stat(abc.ABC):
    """Abstract class representing an statistic."""

    @abc.abstractmethod
    def get(self) -> float:
        """Get method."""


class IncrementalStat(Stat):
    """Abstract class representing an incremental statistic."""

    @abc.abstractmethod
    def update(self, value: Union[int, float]) -> None:
        """Update abstract method."""

    @abc.abstractmethod
    def get(self) -> float:
        """Get method."""


class Mean(IncrementalStat):
    """Incremental mean class."""

    def __init__(self) -> None:
        """Init method."""
        self.mean = 0.0
        self.num_values = 0

    @property
    def mean(self) -> float:
        """Mean property.

        :return: mean value
        :rtype: float
        """
        return self._mean

    @mean.setter
    def mean(self, value: float) -> None:
        """Mean setter.

        :param value: value to be set
        :type value: float
        """
        self._mean = value

    @property
    def num_values(self) -> int:
        """Number of values property.

        :return: number of values
        :rtype: int
        """
        return self._num_values

    @num_values.setter
    def num_values(self, value: int) -> None:
        """Number of values setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_values must be greater of equal than 0.")
        self._num_values = value

    def update(self, value: Union[int, float]) -> None:
        """Update the mean value sequentially.

        :param value: value to use to update the mean
        :type value: int
        :raises TypeError: Type error exception
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self.num_values += 1
        self.mean += (value - self.mean) / self.num_values

    def get(self) -> float:
        """Get method."""
        return self.mean


class EWMA(IncrementalStat):
    """EWMA (Exponential Weighted Moving Average) class."""

    def __init__(self, alpha: float) -> None:
        """Init method.

        :param alpha:
        :type alpha: float
        """
        self.alpha = alpha
        self.one_minus_alpha = 1.0 - self.alpha
        self.mean = 0

    @property
    def alpha(self) -> float:
        """Alpha property.

        :return: alpha value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Alpha setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("alpha must be in the range [0, 1].")
        self._alpha = value

    @property
    def mean(self) -> float:
        """Mean property.

        :return: mean value
        :rtype: float
        """
        return self._mean

    @mean.setter
    def mean(self, value: float) -> None:
        """Mean setter.

        :param value: value to be set
        :type value: float
        """
        self._mean = value

    def update(self, value: Union[int, float]) -> None:
        """Update the mean value sequentially.

        :param value: value to use to update the mean
        :type value: int
        :raises TypeError: Type error exception
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self.mean = self.alpha * value + self.one_minus_alpha * self.mean

    def get(self) -> float:
        """Get method."""
        return self.mean


def permutation_test(
    X: np.ndarray,  # noqa: N803
    Y: np.ndarray,
    statistic: Callable,
    statistical_args: Dict[str, Any],
    num_permutations: int,
    num_jobs: int,
    random_state: Optional[int] = None,
) -> List[float]:
    """Permutation test method.

    :param X: reference data
    :type X: numpy.ndarray
    :param Y: test data
    :type Y: numpy.ndarray
    :param statistic: statistic to use
    :type statistic: Callable
    :param statistical_args: args to pass to statistic method
    :type statistical_args: Dict[str, Any]
    :param num_permutations: number of permutations to use
    :type num_permutations: int
    :param num_jobs: number of jobs to use
    :type num_jobs: int
    :param random_state: random state value
    :type random_state: Optional[int]
    :return: permuted statistics
    :rtype: List[float]
    """
    np.random.seed(seed=random_state)
    X_num_samples, Y_num_samples = X.shape[0], Y.shape[0]  # noqa: N806
    data = np.concatenate([X, Y])

    permutations = []
    for _ in range(num_permutations):
        permuted_data = np.random.permutation(data)
        permutations.append(
            (permuted_data[X_num_samples:], permuted_data[:Y_num_samples])
        )

    with Pool(processes=num_jobs) as pool:
        permuted_statistics = pool.starmap_async(
            partial(statistic, **statistical_args), iterable=permutations
        ).get()

    return permuted_statistics
