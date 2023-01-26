"""MMD (Maximum Mean Discrepancy) module."""

from typing import Callable

import numpy as np  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore

from frouros.data_drift.base import NumericalData, MultivariateData
from frouros.data_drift.batch.distance_based.base import (
    DistanceBasedBase,
    DistanceResult,
)


def rbf_kernel(
    X: np.ndarray, Y: np.ndarray, std: float = 1.0  # noqa: N803
) -> np.ndarray:
    """Radial basis function kernel between X and Y matrices.

    :param X: X matrix
    :type X: numpy.ndarray
    :param Y: Y matrix
    :type Y: numpy.ndarray
    :param std: standard deviation value
    :type std: float

    :return: Radial basis kernel matrix
    :rtype: numpy.ndarray
    """
    return np.exp(-cdist(X, Y, "sqeuclidean") / 2 * std**2)


class MMD(DistanceBasedBase):
    """MMD (Maximum Mean Discrepancy) algorithm class."""

    def __init__(
        self,
        kernel: Callable = rbf_kernel,
    ) -> None:
        """Init method.

        :param kernel: kernel function to use
        :type kernel: Callable
        """
        super().__init__(data_type=NumericalData(), statistical_type=MultivariateData())
        self.kernel = kernel

    @property
    def kernel(self) -> Callable:
        """Kernel property.

        :return: kernel function to use
        :rtype: Kernel
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value: Callable) -> None:
        """Kernel method setter.

        :param value: value to be set
        :type value: Callable
        :raises TypeError: Type error exception
        """
        if not isinstance(value, Callable):  # type: ignore
            raise TypeError("value must be of type Callable.")
        self._kernel = value

    def _distance_measure(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceResult:
        mmd_statistic = self._mmd(X_ref_=X_ref_, X=X, **kwargs)
        distance_test = DistanceResult(distance=mmd_statistic)
        return distance_test

    def _mmd(self, X_ref_: np.ndarray, X: np.ndarray) -> float:  # noqa: N803
        X_ref_num_samples = X_ref_.shape[0]  # noqa: N806
        X_num_samples = X.shape[0]  # noqa: N806
        X_concat = np.vstack((X_ref_, X))  # noqa: N806

        mmd_statistic = self._mmd_statistic(
            X=X_concat,
            X_num_samples=X_num_samples,
            X_ref_num_samples=X_ref_num_samples,
        )
        return mmd_statistic

    def _mmd_statistic(
        self, X: np.ndarray, X_num_samples: int, X_ref_num_samples: int  # noqa: N803
    ) -> float:
        k_matrix = self.kernel(X=X, Y=X)
        k_x = k_matrix[:X_ref_num_samples, :X_ref_num_samples]
        k_y = k_matrix[X_num_samples:, X_num_samples:]
        k_xy = k_matrix[:X_ref_num_samples, X_num_samples:]
        mmd = (
            k_x.sum() / (X_ref_num_samples * (X_ref_num_samples - 1))
            + k_y.sum() / (X_num_samples * (X_num_samples - 1))
            - 2 * k_xy.sum() / (X_ref_num_samples * X_num_samples)
        )
        return mmd
