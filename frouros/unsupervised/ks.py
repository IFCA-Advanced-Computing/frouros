"""KSTest (Kolmogorov-Smirnov test) module."""
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
from scipy.stats import ks_2samp  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.unsupervised.exceptions import MisMatchDimensionError
from frouros.unsupervised.base import TestEstimator


class KSTest(BaseEstimator, TransformerMixin, TestEstimator):
    """KSTest (Kolmogorov-Smirnov test) algorithm class."""

    @TestEstimator.X_ref_.setter  # type: ignore[attr-defined]
    def X_ref_(self, value: Optional[np.ndarray]) -> None:  # noqa: N802
        """Reference data setter.

        :param value: value to be set
        :type value: Optional[numpy.ndarray]
        """
        self._X_ref_ = check_array(value) if value is not None else value  # noqa: N806

    def _validation_checks(self, X: np.ndarray) -> None:  # noqa: N803
        check_is_fitted(self, attributes="X_ref_")
        X = check_array(X)  # noqa: N806
        self.X_ref_: np.ndarray
        if self.X_ref_.shape[1] != X.shape[1]:
            raise MisMatchDimensionError(
                f"Dimensions of X_ref ({self.X_ref_.shape[1]}) "
                f"and X ({X.shape[1]}) must be equal"
            )

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
