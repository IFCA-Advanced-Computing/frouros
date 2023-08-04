"""Permutation test batch callback module."""

import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.utils.stats import permutation


class PermutationTestDistanceBased(BaseCallbackBatch):
    """Permutation test callback class that can be applied to :mod:`data_drift.batch.distance_based <frouros.detectors.data_drift.batch.distance_based>` detectors.

    :param num_permutations: number of permutations to obtain the p-value
    :type num_permutations: int
    :param num_jobs: number of jobs, defaults to -1
    :type num_jobs: int
    :param verbose: verbose flag, defaults to False
    :type verbose: bool
    :param name: name value, defaults to None. If None, the name will be set to `PermutationTestDistanceBased`.
    :type name: Optional[str]

    :Note:
    Callbacks logs are updated with the following variables:

    - `observed_statistic`: observed statistic obtained from the distance-based detector. Same distance value returned by the `compare` method
    - `permutation_statistic`: list of statistics obtained from the permutations
    - `p_value`: p-value obtained from the permutation test

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
    0.0
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        num_permutations: int,
        num_jobs: int = -1,
        verbose: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self.num_permutations = num_permutations
        self.num_jobs = num_jobs
        self.verbose = verbose
        self.permutation_kwargs = kwargs

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
        statistic: Callable,
        statistic_args: Dict[str, Any],
        observed_statistic: float,
        num_permutations: int,
        num_jobs: int,
        random_state: int,
        verbose: bool,
    ) -> Tuple[List[float], float]:
        permuted_statistic = permutation(
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
        p_value = (permuted_statistic >= observed_statistic).mean()  # type: ignore
        return permuted_statistic, p_value

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
            num_jobs=self.num_jobs,
            verbose=self.verbose,
            **self.permutation_kwargs,
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
