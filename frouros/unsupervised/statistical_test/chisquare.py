"""ChiSquareTest (Chi-square test) module."""

import collections
from typing import List, Tuple

import numpy as np  # type: ignore
from scipy.stats import chi2_contingency  # type: ignore

from frouros.unsupervised.base import CategoricalData, UnivariateType
from frouros.unsupervised.statistical_test.base import (  # type: ignore
    StatisticalTestBaseEstimator,
    TestResult,
)


class ChiSquareTest(StatisticalTestBaseEstimator):
    """ChiSquareTest (Chi-square test) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=CategoricalData(), statistical_type=UnivariateType())

    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> TestResult:
        f_exp, f_obs = self._calculate_frequencies(X_ref_=X_ref_, X=X)
        statistic, p_value, _, _ = chi2_contingency(
            observed=np.array([f_obs, f_exp]), **kwargs
        )

        test = TestResult(statistic=statistic, p_value=p_value)
        return test

    @staticmethod
    def _calculate_frequencies(
        X_ref_: np.ndarray, X: np.ndarray  # noqa: N803
    ) -> Tuple[List[int], List[int]]:
        X_ref_counter, X_counter = [  # noqa: N806
            *map(collections.Counter, [X_ref_, X])  # noqa: N806
        ]
        possible_values = set([*X_ref_counter.keys()] + [*X_counter.keys()])
        f_exp, f_obs = {}, {}
        for value in possible_values:
            f_exp[value] = X_ref_counter.get(value, 0)
            f_obs[value] = X_counter.get(value, 0)
        f_exp_values, f_obs_values = [
            *map(list, [f_exp.values(), f_obs.values()])  # type: ignore
        ]
        return f_exp_values, f_obs_values  # type: ignore
