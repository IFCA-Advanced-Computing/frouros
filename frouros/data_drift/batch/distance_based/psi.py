"""PSI (Population Stability Index) module."""

import sys

import numpy as np  # type: ignore

from frouros.data_drift.batch.distance_based.base import (
    DistanceBinsBasedBase,
    DistanceResult,
)


class PSI(DistanceBinsBasedBase):
    """PSI (Population Stability Index) algorithm class."""

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceResult:
        distance = self._distance_measure(X_ref_=X_ref_, X=X, **kwargs)
        return distance

    def _distance_measure_bins(
        self,
        X_ref_: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        # Replace 0.0 values with the smallest number possible
        # in order to avoid division by zero
        X_ref_[X_ref_ == 0.0] = sys.float_info.min
        X[X == 0.0] = sys.float_info.min
        distance = self._psi(
            X_ref_=X_ref_,
            X=X,
        )
        return distance

    @staticmethod
    def _psi(
        X_ref_: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        psi = np.sum((X - X_ref_) * np.log(X / X_ref_))
        return psi
