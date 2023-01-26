"""Bhattacharyya distance module."""

import numpy as np  # type: ignore

from frouros.data_drift.batch.distance_based.base import (
    DistanceBinsBasedBase,
)


class Bhattacharyya(DistanceBinsBasedBase):
    """Bhattacharyya algorithm class."""

    def _distance_measure_bins(
        self,
        X_ref_: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        distance = self._bhattacharyya(
            X_ref_=X_ref_,
            X=X,
        )
        return distance

    @staticmethod
    def _bhattacharyya(X_ref_: np.ndarray, X: np.ndarray) -> float:  # noqa: N803
        distance = 1 - np.sum(np.sqrt(X_ref_ * X))
        return distance
