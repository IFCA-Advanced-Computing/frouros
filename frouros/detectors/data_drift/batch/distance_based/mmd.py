"""MMD (Maximum Mean Discrepancy) module."""

import itertools
import math
from typing import Callable, Generator, Optional, List, Union

import numpy as np  # type: ignore
import tqdm  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import MultivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBased,
    DistanceResult,
)
from frouros.utils.kernels import rbf_kernel


class MMD(BaseDistanceBased):
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
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
    ) -> None:
        """Init method.

        :param kernel: kernel function
        :type kernel: Callable
        :param chunk_size: chunk size value
        :type chunk_size: Optional[int]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
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
        mmd = self._mmd(
            X=X_ref,
            Y=X,
            kernel=self.kernel,
            chunk_size=self.chunk_size,
            **kwargs,
        )
        distance_test = DistanceResult(distance=mmd)
        return distance_test

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

    @staticmethod
    def _mmd(  # pylint: disable=too-many-locals
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

        x_num_samples = len(X)  # noqa: N806
        chunk_size_x = (
            kwargs["chunk_size"]
            if "chunk_size" in kwargs and kwargs["chunk_size"] is not None
            else x_num_samples
        )
        x_chunks, x_chunks_copy = itertools.tee(  # noqa: N806
            MMD._get_chunks(
                data=X,
                chunk_size=chunk_size_x,  # type: ignore
            ),
            2,
        )
        y_num_samples = len(Y)  # noqa: N806
        chunk_size_y = (
            kwargs["chunk_size"]
            if "chunk_size" in kwargs and kwargs["chunk_size"] is not None
            else y_num_samples
        )
        y_chunks, y_chunks_copy = itertools.tee(  # noqa: N806
            MMD._get_chunks(
                data=Y,
                chunk_size=chunk_size_y,  # type: ignore
            ),
            2,
        )
        x_chunks_combinations = itertools.product(  # noqa: N806
            x_chunks,
            repeat=2,
        )
        y_chunks_combinations = itertools.product(  # noqa: N806
            y_chunks,
            repeat=2,
        )
        xy_chunks_combinations = itertools.product(  # noqa: N806
            x_chunks_copy,
            y_chunks_copy,
        )

        if kwargs.get("verbose", False):
            num_chunks_x = math.ceil(x_num_samples / chunk_size_x)  # type: ignore
            num_chunks_y = math.ceil(y_num_samples / chunk_size_y)  # type: ignore
            num_chunks_x_combinations = num_chunks_x**2
            num_chunks_y_combinations = num_chunks_y**2
            num_chunks_xy = (
                math.ceil(len(X) / chunk_size_x) * num_chunks_y  # type: ignore
            )
            x_chunks_combinations = tqdm.tqdm(
                x_chunks_combinations,
                total=num_chunks_x_combinations,
            )
            y_chunks_combinations = tqdm.tqdm(
                y_chunks_combinations,
                total=num_chunks_y_combinations,
            )
            xy_chunks_combinations = tqdm.tqdm(
                xy_chunks_combinations,
                total=num_chunks_xy,
            )

        k_xx_sum = (
            MMD._compute_kernel(
                chunk_combinations=x_chunks_combinations,  # type: ignore
                kernel=kernel,
            )
            # Remove diagonal (j!=i case)
            - x_num_samples  # type: ignore
        )
        k_yy_sum = (
            MMD._compute_kernel(
                chunk_combinations=y_chunks_combinations,  # type: ignore
                kernel=kernel,
            )
            # Remove diagonal (j!=i case)
            - y_num_samples  # type: ignore
        )
        k_xy_sum = MMD._compute_kernel(
            chunk_combinations=xy_chunks_combinations,  # type: ignore
            kernel=kernel,
        )
        mmd = (
            +k_xx_sum / (x_num_samples * (x_num_samples - 1))
            + k_yy_sum / (y_num_samples * (y_num_samples - 1))
            - 2 * k_xy_sum / (x_num_samples * y_num_samples)  # type: ignore
        )
        return mmd
