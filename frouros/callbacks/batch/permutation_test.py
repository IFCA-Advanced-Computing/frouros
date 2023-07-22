"""Permutation test batch callback module."""

import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.utils.stats import permutation


class PermutationTestDistanceBased(BaseCallbackBatch):
    """Permutation test on distance based batch callback class."""

    def __init__(
        self,
        num_permutations: int,
        num_jobs: int = -1,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Init method.

        :param num_permutations: number of permutations
        :type num_permutations: int
        :param num_jobs: number of jobs
        :type num_jobs: int
        :param name: name to be use
        :type name: Optional[str]
        """
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

    def on_compare_end(self, **kwargs) -> None:
        """On compare end method."""
        X_ref, X_test = kwargs["X_ref"], kwargs["X_test"]  # noqa: N806
        observed_statistic = kwargs["result"][0]
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
