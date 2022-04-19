"""KSTest (Kolmogorov-Smirnov test) module."""
from typing import List, Tuple

import numpy as np  # type: ignore
from scipy.stats import ks_2samp  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

from frouros.unsupervised.statistical_test.base import StatisticalTestEstimator


class KSTest(BaseEstimator, TransformerMixin, StatisticalTestEstimator):
    """KSTest (Kolmogorov-Smirnov test) algorithm class."""

    @staticmethod
    def _statistical_test(
        X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> List[Tuple[float, float]]:
        test = ks_2samp(
            data1=X_ref_,
            data2=X,
            alternative=kwargs.get("alternative", "two-sided"),
            mode=kwargs.get("method", "auto"),
        )
        return test
