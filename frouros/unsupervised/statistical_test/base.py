"""Unsupervised statistical test base module."""

import abc

from typing import Tuple
import numpy as np  # type: ignore

from frouros.unsupervised.base import UnsupervisedBaseEstimator


class StatisticalTestEstimator(UnsupervisedBaseEstimator):
    """Abstract class representing a statistical test estimator."""

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        statistical_test = self._statistical_test(X_ref_=X_ref_, X=X, **kwargs)
        return statistical_test

    @staticmethod
    @abc.abstractmethod
    def _statistical_test(
        X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        pass
