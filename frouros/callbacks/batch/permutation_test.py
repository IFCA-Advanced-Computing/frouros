"""Permutation test batch callback module."""

import multiprocessing
from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.stats import binom

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.utils.stats import permutation

MAX_NUM_PERM: int = 1000000  # maximum number of permutations


class PermutationTestDistanceBased(BaseCallbackBatch):
    """Permutation test callback class that can be applied to :mod:`data_drift.batch.distance_based <frouros.detectors.data_drift.batch.distance_based>` detectors.

    :param num_permutations: number of permutations to obtain the p-value
    :type num_permutations: int
    :param total_num_permutations: total number of permutations to obtain the p-value, defaults to None. If None, the total number of permutations will be set to the maximum number of permutations, the minimum between all possible permutations or the global maximum number of permutations
    :type total_num_permutations: Optional[int]
    :param num_jobs: number of jobs, defaults to -1
    :type num_jobs: int
    :param method: method to compute the p-value, defaults to "auto".
        "`auto`": if the number of permutations is greater than the maximum number of permutations, the method will be set to "approximate". Otherwise, the method will be set to "exact".
        "`conservative`": p-value is computed as (number of permutations greater or equal than the observed statistic + 1) / (number of permutations + 1).
        "`exact`": p-value is computed as the mean of the binomial cumulative distribution function as stated :cite:`phipson2010permutation`.
        "`approximate`": p-value is computed using the integral of the binomial cumulative distribution function as stated :cite:`phipson2010permutation`.
        "`estimate`": p-value is computed as the mean of the extreme statistic. p-value can be zero.
    :type method: str
    :param random_state: random state, defaults to None
    :type random_state: Optional[int]
    :param verbose: verbose flag, defaults to False
    :type verbose: bool
    :param name: name value, defaults to None. If None, the name will be set to `PermutationTestDistanceBased`.
    :type name: Optional[str]

    :Note:
    Callbacks logs are updated with the following variables:

    - `observed_statistic`: observed statistic obtained from the distance-based detector. Same distance value returned by the `compare` method
    - `permutation_statistic`: list of statistics obtained from the permutations
    - `p_value`: p-value obtained from the permutation test

    :References:

    .. [phipson2010permutation] Phipson, Belinda, and Gordon K. Smyth.
        "Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn."
        Statistical applications in genetics and molecular biology 9.1 (2010).

    :Example:

    >>> from frouros.callbacks import PermutationTestDistanceBased
    >>> from frouros.detectors.data_drift import MMD
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.multivariate_normal(mean=[1, 1], cov=[[2, 0], [0, 2]], size=100)
    >>> Y = np.random.multivariate_normal(mean=[0, 0], cov=[[2, 1], [1, 2]], size=100)
    >>> detector = MMD(callbacks=PermutationTestDistanceBased(num_permutations=1000, random_state=31))
    >>> _ = detector.fit(X=X)
    >>> distance, callbacks_log = detector.compare(X=Y)
    >>> distance
    DistanceResult(distance=0.05643613752975596)
    >>> callbacks_log["PermutationTestDistanceBased"]["p_value"]
    0.0009985010823343311
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107  # pylint: disable=too-many-arguments
        self,
        num_permutations: int,
        total_num_permutations: Optional[int] = None,
        num_jobs: int = -1,
        method: str = "auto",
        random_state: Optional[int] = None,
        verbose: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_permutations = num_permutations
        self.total_num_permutations = total_num_permutations
        self.num_jobs = num_jobs
        self.method = method
        self.random_state = random_state
        self.verbose = verbose

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
        if value > MAX_NUM_PERM:
            raise ValueError(f"value must be less than or equal to {MAX_NUM_PERM}.")
        self._num_permutations = value

    @property
    def total_num_permutations(self) -> Optional[int]:
        """Number of total permutations' property.

        :return: number of total permutations
        :rtype: Optional[int]
        """
        return self._total_permutations

    @total_num_permutations.setter
    def total_num_permutations(self, value: Optional[int]) -> None:
        """Number of total permutations method setter.

        :param value: value to be set
        :type value: Optional[int]
        :raises ValueError: Value error exception
        """
        if value is not None:
            if value < 1:
                raise ValueError("value must be greater of equal than 1.")
            if value > MAX_NUM_PERM:
                raise ValueError(f"value must be less than or equal to {MAX_NUM_PERM}.")
        self._total_permutations = value

    @property
    def num_jobs(self) -> int:
        """Number of jobs property.

        :return: number of jobs to use
        :rtype: int
        """
        return self._num_jobs

    @num_jobs.setter
    def num_jobs(self, value: int) -> None:
        """Number of jobs method setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value == 0 or value < -1:
            raise ValueError("value must be greater than 0 or -1.")
        self._num_jobs = multiprocessing.cpu_count() if value == -1 else value

    @property
    def method(self) -> str:
        """Method to compute the p-value property.

        :return: method to compute the p-value
        :rtype: str
        """
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Method to compute the p-value setter.

        :param value: value to be set
        :type value: str
        :raises ValueError: Value error exception
        """
        if value not in ["auto", "conservative", "exact", "approximate", "estimate"]:
            raise ValueError(
                "value must be one of "
                "'auto', 'conservative', 'exact', 'approximate', 'estimate'.",
            )
        self._method = value

    @property
    def verbose(self) -> bool:
        """Verbose flag property.

        :return: verbose flag
        :rtype: bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Verbose flag setter.

        :param value: value to be set
        :type value: bool
        :raises TypeError: Type error exception
        """
        if not isinstance(value, bool):
            raise TypeError("value must of type bool.")
        self._verbose = value

    @staticmethod
    def _calculate_p_value(  # pylint: disable=too-many-arguments
        X_ref: np.ndarray,  # noqa: N803
        X_test: np.ndarray,
        statistic: Callable,  # type: ignore
        statistic_args: dict[str, Any],
        observed_statistic: float,
        num_permutations: int,
        total_num_permutations: Optional[int],
        num_jobs: int,
        method: str,
        random_state: Optional[int],
        verbose: bool,
    ) -> Tuple[np.ndarray, float]:
        permuted_statistic, max_num_permutations = permutation(
            X=X_ref,
            Y=X_test,
            statistic=statistic,
            statistical_args=statistic_args,
            num_permutations=num_permutations,
            num_jobs=num_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        permuted_statistic = np.array(permuted_statistic)
        extreme_statistic = permuted_statistic >= observed_statistic  # type: ignore

        if total_num_permutations is None:
            # Set the total number of permutations to the maximum number of
            # permutations, the minimum between all possible permutations or
            # the global maximum number of permutations
            total_num_permutations = np.min([max_num_permutations, MAX_NUM_PERM])

        if method == "auto":
            method = "approximate" if num_permutations > MAX_NUM_PERM else "exact"
        if method == "conservative":
            p_value = PermutationTestDistanceBased._compute_conservative(
                num_permutations=num_permutations,
                observed_statistic=observed_statistic,
                permuted_statistic=permuted_statistic,
            )
        elif method == "exact":
            p_value = PermutationTestDistanceBased._compute_exact(
                extreme_statistic=extreme_statistic,
                total_num_permutations=total_num_permutations,
                permuted_statistic=permuted_statistic,
            )
        elif method == "approximate":
            p_value = PermutationTestDistanceBased._compute_approximate(
                extreme_statistic=extreme_statistic,
                total_num_permutations=total_num_permutations,
                permuted_statistic=permuted_statistic,
            )
        else:  # method == "estimate"
            p_value = PermutationTestDistanceBased._compute_estimate(
                extreme_statistic=extreme_statistic,
            )

        return permuted_statistic, p_value

    @staticmethod
    def _compute_estimate(
        extreme_statistic: np.ndarray,
    ) -> float:
        p_value = extreme_statistic.mean()
        return p_value

    @staticmethod
    def _compute_approximate(
        extreme_statistic: np.ndarray,
        total_num_permutations: int,
        permuted_statistic: np.ndarray,
    ) -> float:
        num_extreme_statistic = extreme_statistic.sum()
        num_permutations = len(permuted_statistic)
        integral, _ = quad(
            func=lambda p: binom.cdf(
                num_extreme_statistic,
                num_permutations,
                p,
            ),
            a=0,
            b=0.5 / total_num_permutations,
        )
        p_value = (num_extreme_statistic + 1) / (
            num_permutations + 1
        ) - 0.5 / total_num_permutations * integral
        return p_value

    @staticmethod
    def _compute_exact(
        extreme_statistic: np.ndarray,
        total_num_permutations: int,
        permuted_statistic: np.ndarray,
    ) -> float:
        num_extreme_statistic = extreme_statistic.sum()
        num_permutations = len(permuted_statistic)
        p = np.arange(1, total_num_permutations + 1) / total_num_permutations
        y = binom.cdf(
            num_extreme_statistic,
            num_permutations,
            p,
        )
        p_value = np.mean(y)
        return p_value

    @staticmethod
    def _compute_conservative(
        num_permutations: int,
        observed_statistic: float,
        permuted_statistic: np.ndarray,
    ) -> float:
        p_value = ((permuted_statistic >= observed_statistic).sum() + 1) / (
            num_permutations + 1
        )
        return p_value

    def on_compare_end(
        self,
        result: Any,
        X_ref: np.ndarray,  # noqa: N803
        X_test: np.ndarray,
    ) -> None:
        """On compare end method.

        :param result: result obtained from the `compare` method
        :type result: Any
        :param X_ref: reference data
        :type X_ref: numpy.ndarray
        :param X_test: test data
        :type X_test: numpy.ndarray
        """
        observed_statistic = result.distance
        permuted_statistics, p_value = self._calculate_p_value(
            X_ref=X_ref,
            X_test=X_test,
            statistic=self.detector.statistical_method,  # type: ignore
            statistic_args=self.detector.statistical_kwargs,  # type: ignore
            observed_statistic=observed_statistic,
            num_permutations=self.num_permutations,
            total_num_permutations=self.total_num_permutations,
            num_jobs=self.num_jobs,
            method=self.method,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.logs.update(
            {
                "observed_statistic": observed_statistic,
                "permuted_statistics": permuted_statistics,
                "p_value": p_value,
            },
        )

    # FIXME: set_detector method as a workaround to  # pylint: disable=fixme
    #  avoid circular import problem. Make it an abstract method and
    #  uncomment commented code when it is solved

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, BaseDataDriftBatch):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type BaseDataDriftBatch."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""
