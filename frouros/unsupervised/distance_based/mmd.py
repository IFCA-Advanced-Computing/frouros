"""MMD (Maximum Mean Discrepancy) module."""

from typing import Optional, Tuple

import numpy as np  # type: ignore
from sklearn.gaussian_process.kernels import Kernel, RBF  # type: ignore

from frouros.unsupervised.base import NumericalData, MultivariateType
from frouros.unsupervised.distance_based.base import (
    DistanceBasedEstimator,
    DistanceTestResult,
)


class MMD(DistanceBasedEstimator):
    """MMD (Maximum Mean Discrepancy) algorithm class."""

    def __init__(
        self,
        num_permutations: int,
        kernel: Kernel = RBF,
        random_state: Optional[int] = None,
    ) -> None:
        """Init method.

        :param num_permutations: number of permutations to obtain the p-value
        :type num_permutations: int
        :param kernel: kernel to use
        :type kernel: Kernel
        :param random_state: random state value
        :type random_state: Optional[int]
        """
        super().__init__(data_type=NumericalData(), statistical_type=MultivariateType())
        self.kernel = kernel
        self.num_permutations = num_permutations
        self.random_state = random_state

    @property
    def kernel(self) -> Kernel:
        """Kernel property.

        :return: kernel function to use
        :rtype: Kernel
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value: Kernel) -> None:
        """Kernel method setter.

        :param value: value to be set
        :type value: Kernel
        :raises TypeError: Type error exception
        """
        if not isinstance(value, Kernel):
            raise TypeError("value must be of type Kernel.")
        self._kernel = value

    @property
    def num_permutations(self) -> int:
        """Number of permutations property.

        :return: number of permutation to obtain the p-value
        :rtype: int
        """
        return self._num_permutations

    @num_permutations.setter
    def num_permutations(self, value: int) -> None:
        """Number of permutations method setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("value must be greater of equal than 1.")
        self._num_permutations = value

    def _distance_measure(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceTestResult:
        mmd_statistic, p_value = self._mmd(X_ref_=X_ref_, X=X, **kwargs)
        distance_test = DistanceTestResult(distance=mmd_statistic, p_value=p_value)
        return distance_test

    def _mmd(
        self, X_ref_: np.ndarray, X: np.ndarray  # noqa: N803
    ) -> Tuple[float, float]:
        X_ref_num_samples = X_ref_.shape[0]  # noqa: N806
        X_num_samples = X.shape[0]  # noqa: N806
        X_concat = np.vstack((X_ref_, X))  # noqa: N806

        mmd_statistic = self._mmd_statistic(
            X=X_concat,
            X_num_samples=X_num_samples,
            X_ref_num_samples=X_ref_num_samples,
        )
        p_value = self._calculate_p_value(
            X=X_concat,
            X_ref_num_samples=X_ref_num_samples,
            mmd_statistic=mmd_statistic,
            num_permutations=self.num_permutations,
        )
        return mmd_statistic, p_value

    def _calculate_p_value(
        self,
        X: np.ndarray,  # noqa: N803
        X_ref_num_samples: int,
        mmd_statistic: float,
        num_permutations: int,
    ) -> float:
        np.random.seed(seed=self.random_state)
        mmd_permutations = []
        for _ in range(num_permutations):
            X_permuted = X[np.random.permutation(X.shape[0])]  # noqa: N806
            X_permuted_ref_ = X_permuted[:X_ref_num_samples]  # noqa: N806
            X_permuted_ref_num_samples = X_permuted_ref_.shape[0]  # noqa: N806
            X_permuted_ = X_permuted[X_ref_num_samples:]  # noqa: N806
            X_permuted_num_samples = X_permuted_.shape[0]  # noqa: N806
            X_permuted_concat = np.vstack((X_permuted_ref_, X_permuted_))  # noqa: N806
            mmd_permutations.append(
                self._mmd_statistic(
                    X=X_permuted_concat,
                    X_ref_num_samples=X_permuted_ref_num_samples,
                    X_num_samples=X_permuted_num_samples,
                )
            )
        p_value = (mmd_statistic < mmd_permutations).mean()  # type: ignore
        return p_value

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
