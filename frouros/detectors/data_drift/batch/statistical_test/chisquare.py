"""ChiSquareTest (Chi-square test) module."""

import collections
from typing import Optional, List, Tuple, Union

import numpy as np  # type: ignore
from scipy.stats import chi2_contingency  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import CategoricalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (  # type: ignore
    BaseStatisticalTest,
    StatisticalResult,
)


class ChiSquareTest(BaseStatisticalTest):
    """ChiSquareTest (Chi-square test) [pearson1900x]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]

    :Note:
    - Passing additional arguments to `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`__ can be done using :func:`compare` kwargs.

    :References:

    .. [pearson1900x] Pearson, Karl.
        "X. On the criterion that a given system of deviations from the probable in the
        case of a correlated system of variables is such that it can be reasonably
        supposed to have arisen from random sampling."
        The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
        Science 50.302 (1900): 157-175.

    :Example:

    >>> from frouros.detectors.data_drift import ChiSquareTest
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
    >>> Y = np.random.choice(a=[0, 1], size=100, p=[0.8, 0.2])
    >>> detector = ChiSquareTest()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    StatisticalResult(statistic=9.81474665685192, p_value=0.0017311812135839511)
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
    ) -> None:
        super().__init__(
            data_type=CategoricalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )

    @staticmethod
    def _statistical_test(
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs,
    ) -> StatisticalResult:
        f_exp, f_obs = ChiSquareTest._calculate_frequencies(
            X_ref=X_ref,
            X=X,
        )
        statistic, p_value, _, _ = chi2_contingency(
            observed=np.array([f_obs, f_exp]),
            **kwargs,
        )

        test = StatisticalResult(
            statistic=statistic,
            p_value=p_value,
        )
        return test

    @staticmethod
    def _calculate_frequencies(
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
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
