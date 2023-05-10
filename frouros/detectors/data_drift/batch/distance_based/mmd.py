"""MMD (Maximum Mean Discrepancy) module."""

import itertools
import math
from typing import Callable, Generator, Optional, List, Union

import numpy as np  # type: ignore
import tqdm  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.data_drift.base import MultivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
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
    """MMD (Maximum Mean Discrepancy) [gretton2012kernel]_ detector.

    :References:

    .. [gretton2012kernel] Gretton, Arthur, et al.
        "A kernel two-sample test."
        The Journal of Machine Learning Research 13.1 (2012): 723-773.
    """

    def __init__(
        self,
        kernel: Callable = rbf_kernel,
        chunk_size: Optional[int] = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param kernel: kernel function
        :type kernel: Callable
        :param chunk_size: chunk size value
        :type chunk_size: Optional[int]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(
            statistical_type=MultivariateData(),
            statistical_method=self._mmd,
            statistical_kwargs={
                "kernel": kernel,
            },
            callbacks=callbacks,
        )
        self.kernel = kernel
        self.chunk_size = chunk_size
        self._chunk_size_x = None
        self.X_chunks_combinations = None
        self.X_num_samples = None
        self.expected_k_xx = None

    @property
    def chunk_size(self) -> Optional[int]:
        """Chunk size property.

        :return: chunk size to use
        :rtype: int
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value: Optional[int]) -> None:
        """Chunk size method setter.

        :param value: value to be set
        :type value: Optional[int]
        :raises TypeError: Type error exception
        """
        if value is not None:
            if isinstance(value, int):  # type: ignore
                if value <= 0:
                    raise ValueError("chunk_size must be greater than 0 or None.")
            else:
                raise TypeError("chunk_size must be of type int or None.")
        self._chunk_size = value

    @property
    def kernel(self) -> Callable:
        """Kernel property.

        :return: kernel function to use
        :rtype: Callable
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
            raise TypeError("kernel must be of type Callable.")
        self._kernel = value

    def _distance_measure(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> DistanceResult:
        mmd = self._mmd(X=X_ref, Y=X, kernel=self.kernel, **kwargs)
        distance_test = DistanceResult(distance=mmd)
        return distance_test

    def _fit(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> None:
        super()._fit(X=X)
        # Add dimension only for the kernel calculation (if dim == 1)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)  # noqa: N806
        self.X_num_samples = len(self.X_ref)  # type: ignore # noqa: N806

        self._chunk_size_x = (
            self.X_num_samples
            if self.chunk_size is None
            else self.chunk_size  # type: ignore
        )

        X_chunks = self._get_chunks(  # noqa: N806
            data=X,
            chunk_size=self._chunk_size_x,  # type: ignore
        )
        xx_chunks_combinations = itertools.product(X_chunks, repeat=2)  # noqa: N806

        if kwargs.get("verbose", False):
            num_chunks = (
                math.ceil(self.X_num_samples / self._chunk_size_x) ** 2  # type: ignore
            )
            xx_chunks_combinations = tqdm.tqdm(
                xx_chunks_combinations,
                total=num_chunks,
            )

        k_xx_sum = (
            self._compute_kernel(
                chunk_combinations=xx_chunks_combinations,  # type: ignore
                kernel=self.kernel,
            )
            # Remove diagonal (j!=i case)
            - self.X_num_samples  # type: ignore
        )

        self.expected_k_xx = k_xx_sum / (  # type: ignore
            self.X_num_samples * (self.X_num_samples - 1)  # type: ignore
        )

    @staticmethod
    def _compute_kernel(chunk_combinations: Generator, kernel: Callable) -> float:
        k_sum = np.array([kernel(*chunk).sum() for chunk in chunk_combinations]).sum()
        return k_sum

    @staticmethod
    def _get_chunks(data: np.ndarray, chunk_size: int) -> Generator:
        chunks = (
            data[i : i + chunk_size]  # noqa: E203
            for i in range(0, len(data), chunk_size)
        )
        return chunks

    def _mmd(  # pylint: disable=too-many-locals
        self,
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
        *,
        kernel: Callable,
        **kwargs,
    ) -> float:  # noqa: N803
        # Only check for X dimension (X == Y dim comparison has been already made)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)  # noqa: N806
            Y = np.expand_dims(Y, axis=1)  # noqa: N806

        X_chunks = self._get_chunks(  # noqa: N806
            data=X,
            chunk_size=self._chunk_size_x,  # type: ignore
        )
        y_num_samples = len(Y)  # noqa: N806
        chunk_size_y = y_num_samples if self.chunk_size is None else self.chunk_size
        y_chunks, y_chunks_copy = itertools.tee(  # noqa: N806
            self._get_chunks(
                data=Y,
                chunk_size=chunk_size_y,  # type: ignore
            ),
            2,
        )
        y_chunks_combinations = itertools.product(  # noqa: N806
            y_chunks,
            repeat=2,
        )
        xy_chunks_combinations = itertools.product(  # noqa: N806
            X_chunks,
            y_chunks_copy,
        )

        if kwargs.get("verbose", False):
            num_chunks_y = math.ceil(y_num_samples / chunk_size_y)  # type: ignore
            num_chunks_y_combinations = num_chunks_y**2
            num_chunks_xy = (
                math.ceil(len(X) / self._chunk_size_x) * num_chunks_y  # type: ignore
            )
            y_chunks_combinations = tqdm.tqdm(
                y_chunks_combinations,
                total=num_chunks_y_combinations,
            )
            xy_chunks_combinations = tqdm.tqdm(
                xy_chunks_combinations,
                total=num_chunks_xy,
            )

        k_yy_sum = (
            self._compute_kernel(
                chunk_combinations=y_chunks_combinations,  # type: ignore
                kernel=kernel,
            )
            # Remove diagonal (j!=i case)
            - y_num_samples  # type: ignore
        )
        k_xy_sum = self._compute_kernel(
            chunk_combinations=xy_chunks_combinations,  # type: ignore
            kernel=kernel,
        )
        mmd = (
            self.expected_k_xx  # type: ignore
            + k_yy_sum / (y_num_samples * (y_num_samples - 1))
            - 2 * k_xy_sum / (self.X_num_samples * y_num_samples)  # type: ignore
        )
        return mmd
