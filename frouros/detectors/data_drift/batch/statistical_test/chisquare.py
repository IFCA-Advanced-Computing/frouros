"""ChiSquareTest (Chi-square test) module."""

import collections
from typing import Optional, List, Tuple, Union

import numpy as np  # type: ignore
from scipy.stats import chi2_contingency  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.data_drift.base import CategoricalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (  # type: ignore
    StatisticalTestBase,
    StatisticalResult,
)


class ChiSquareTest(StatisticalTestBase):
    """ChiSquareTest (Chi-square test) algorithm class."""

    def __init__(
        self, callbacks: Optional[Union[Callback, List[Callback]]] = None
    ) -> None:
        """Init method.

        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(
            data_type=CategoricalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )

    def _statistical_test(
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> StatisticalResult:
        f_exp, f_obs = self._calculate_frequencies(X_ref=X_ref, X=X)
        statistic, p_value, _, _ = chi2_contingency(
            observed=np.array([f_obs, f_exp]), **kwargs
        )

        test = StatisticalResult(statistic=statistic, p_value=p_value)
        return test

    @staticmethod
    def _calculate_frequencies(
        X_ref: np.ndarray, X: np.ndarray  # noqa: N803
    ) -> Tuple[List[int], List[int]]:
        X_ref_counter, X_counter = [  # noqa: N806
            *map(collections.Counter, [X_ref, X])  # noqa: N806
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
