"""ChiSquaredTest (Chi-squared test) module."""

import collections
from typing import List, Tuple

import numpy as np  # type: ignore
from scipy.stats import chisquare  # type: ignore

from frouros.unsupervised.base import CategoricalData, UnivariateTestType
from frouros.unsupervised.statistical_test.base import (  # type: ignore
    StatisticalTestBaseEstimator,
)
from frouros.unsupervised.exceptions import DistinctNumberSamplesError


class ChiSquaredTest(StatisticalTestBaseEstimator):
    """ChiSquaredTest (Chi-squared test) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=CategoricalData(), test_type=UnivariateTestType())

    def _specific_checks(self, X: np.ndarray) -> None:  # noqa: N803
        self._check_equal_number_samples(X_ref_=self.X_ref_, X=X)

    @staticmethod
    def _check_equal_number_samples(
        X_ref_: np.ndarray, X: np.ndarray  # noqa: N803
    ) -> None:
        if X_ref_.shape[0] != X.shape[0]:
            raise DistinctNumberSamplesError("Number of samples must be equal.")

    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        f_exp, f_obs = self._calculate_frequencies(X_ref_=X_ref_, X=X)

        test = chisquare(
            f_obs=f_obs,
            f_exp=f_exp,
            ddof=0,
            axis=kwargs.get("axis", 0),
        )
        return test.statistic, test.pvalue

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
