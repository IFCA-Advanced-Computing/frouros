"""MMD (Maximum Mean Discrepancy) module."""

from typing import Callable, Optional, List, Union

import numpy as np  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore

from frouros.callbacks import Callback
from frouros.data_drift.base import MultivariateData
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
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param kernel: kernel function to use
        :type kernel: Callable
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(
            statistical_type=MultivariateData(),
            statistical_method=self._mmd,
            statistical_kwargs={"kernel": kernel},
            callbacks=callbacks,
        )
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
        self,
        X_ref_: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> DistanceResult:
        mmd = self._mmd(X=X_ref_, Y=X, kernel=self.kernel)
        distance_test = DistanceResult(distance=mmd)
        return distance_test

    @staticmethod
    def _mmd(
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
        *,
        kernel: Callable,
    ) -> float:  # noqa: N803
        X_num_samples = X.shape[0]  # noqa: N806
        Y_num_samples = Y.shape[0]  # noqa: N806
        data = np.concatenate([X, Y])  # noqa: N806
        if X.ndim == 1:
            data = np.expand_dims(data, axis=1)

        k_matrix = kernel(X=data, Y=data)
        k_x = k_matrix[:X_num_samples, :X_num_samples]
        k_y = k_matrix[Y_num_samples:, Y_num_samples:]
        k_xy = k_matrix[:X_num_samples, Y_num_samples:]
        mmd = (
            k_x.sum() / (X_num_samples * (X_num_samples - 1))
            + k_y.sum() / (Y_num_samples * (Y_num_samples - 1))
            - 2 * k_xy.sum() / (X_num_samples * Y_num_samples)
        )
        return mmd
