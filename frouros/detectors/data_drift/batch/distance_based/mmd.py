"""MMD (Maximum Mean Discrepancy) module."""

import itertools
from typing import Any, Callable, Generator, Optional, Union

import numpy as np

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import MultivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBased,
    DistanceResult,
)
from frouros.utils.kernels import rbf_kernel


class MMD(BaseDistanceBased):
    """MMD (Maximum Mean Discrepancy) [gretton2012kernel]_ detector.

    :param kernel: kernel function, defaults to :func:`rbf_kernel() <frouros.utils.kernels.rbf_kernel>`
    :type kernel: Callable
    :param chunk_size: chunk size value, defaults to None
    :type chunk_size: Optional[int]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :References:

    .. [gretton2012kernel] Gretton, Arthur, et al.
        "A kernel two-sample test."
        The Journal of Machine Learning Research 13.1 (2012): 723-773.

    :Example:

    >>> from functools import partial
    >>> from frouros.detectors.data_drift import MMD
    >>> from frouros.utils.kernels import rbf_kernel
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.multivariate_normal(mean=[1, 1], cov=[[2, 0], [0, 2]], size=100)
    >>> Y = np.random.multivariate_normal(mean=[0, 0], cov=[[2, 1], [1, 2]], size=100)
    >>> detector = MMD(kernel=partial(rbf_kernel, sigma=0.5))
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    DistanceResult(distance=0.02146955300299802)
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        kernel: Callable = rbf_kernel,  # type: ignore
        chunk_size: Optional[int] = None,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
    ) -> None:
        super().__init__(
            statistical_type=MultivariateData(),
            statistical_method=self._mmd,
            statistical_kwargs={
                "kernel": kernel,
                "chunk_size": chunk_size,
            },
            callbacks=callbacks,
        )
        self.kernel = kernel
        self.chunk_size = chunk_size
        self._expected_k_xx = None

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
            if isinstance(value, int):
                if value <= 0:
                    raise ValueError("chunk_size must be greater than 0 or None.")
            else:
                raise TypeError("chunk_size must be of type int or None.")
        self._chunk_size = value

    @property
    def kernel(self) -> Callable:  # type: ignore
        """Kernel property.

        :return: kernel function to use
        :rtype: Callable
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value: Callable) -> None:  # type: ignore
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
        **kwargs: Any,
    ) -> DistanceResult:
        mmd = self._mmd(
            X=X_ref,
            Y=X,
            kernel=self.kernel,
            chunk_size=self.chunk_size,
            expected_k_xx=self._expected_k_xx,
            **kwargs,
        )
        distance_test = DistanceResult(distance=mmd)
        return distance_test

    def _fit(
        self,
        X: np.ndarray,  # noqa: N803
    ) -> None:
        super()._fit(X=X)
        # Add dimension only for the kernel calculation (if dim == 1)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)  # noqa: N806
        x_num_samples = len(self.X_ref)  # type: ignore

        chunk_size_x = x_num_samples if self.chunk_size is None else self.chunk_size

        x_chunks = self._get_chunks(  # noqa: N806
            data=X,
            chunk_size=chunk_size_x,
        )
        x_chunks_combinations = itertools.product(x_chunks, repeat=2)  # noqa: N806

        k_xx_sum = (
            self._compute_kernel(
                chunk_combinations=x_chunks_combinations,  # type: ignore
                kernel=self.kernel,
            )
            # Remove diagonal (j!=i case)
            - x_num_samples
        )

        self._expected_k_xx = k_xx_sum / (  # type: ignore
            x_num_samples * (x_num_samples - 1)
        )

    @staticmethod
    def _compute_kernel(
        chunk_combinations: Generator,  # type: ignore
        kernel: Callable,  # type: ignore
    ) -> float:
        k_sum = np.array([kernel(*chunk).sum() for chunk in chunk_combinations]).sum()
        return k_sum

    @staticmethod
    def _get_chunks(data: np.ndarray, chunk_size: int) -> Generator:  # type: ignore
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
        kernel: Callable,  # type: ignore
        **kwargs: Any,
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

        # If expected_k_xx is provided, we don't need to compute it again
        if "expected_k_xx" in kwargs:
            x_chunks_copy = MMD._get_chunks(  # noqa: N806
                data=X,
                chunk_size=chunk_size_x,
            )
            expected_k_xx = kwargs["expected_k_xx"]
        else:
            # Compute expected_k_xx
            x_chunks, x_chunks_copy = itertools.tee(  # type: ignore
                MMD._get_chunks(
                    data=X,
                    chunk_size=chunk_size_x,
                ),
                2,
            )
            x_chunks_combinations = itertools.product(
                x_chunks,
                repeat=2,
            )
            k_xx_sum = (
                MMD._compute_kernel(
                    chunk_combinations=x_chunks_combinations,  # type: ignore
                    kernel=kernel,
                )
                # Remove diagonal (j!=i case)
                - x_num_samples
            )
            expected_k_xx = k_xx_sum / (x_num_samples * (x_num_samples - 1))

        y_num_samples = len(Y)  # noqa: N806
        chunk_size_y = (
            kwargs["chunk_size"]
            if "chunk_size" in kwargs and kwargs["chunk_size"] is not None
            else y_num_samples
        )
        y_chunks, y_chunks_copy = itertools.tee(  # noqa: N806
            MMD._get_chunks(
                data=Y,
                chunk_size=chunk_size_y,
            ),
            2,
        )
        y_chunks_combinations = itertools.product(  # noqa: N806
            y_chunks,
            repeat=2,
        )
        xy_chunks_combinations = itertools.product(  # noqa: N806
            x_chunks_copy,
            y_chunks_copy,
        )

        k_yy_sum = (
            MMD._compute_kernel(
                chunk_combinations=y_chunks_combinations,  # type: ignore
                kernel=kernel,
            )
            # Remove diagonal (j!=i case)
            - y_num_samples
        )
        k_xy_sum = MMD._compute_kernel(
            chunk_combinations=xy_chunks_combinations,  # type: ignore
            kernel=kernel,
        )
        mmd = (
            +expected_k_xx
            + k_yy_sum / (y_num_samples * (y_num_samples - 1))
            - 2 * k_xy_sum / (x_num_samples * y_num_samples)
        )
        return mmd
