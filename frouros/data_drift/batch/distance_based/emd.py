"""EMD (Earth Mover's Distance) module."""

import numpy as np  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore

from frouros.data_drift.base import NumericalData, UnivariateData
from frouros.data_drift.batch.distance_based.base import (
    DistanceBasedBase,
    DistanceResult,
)


class EMD(DistanceBasedBase):
    """EMD (Earth Mover's Distance) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=NumericalData(), statistical_type=UnivariateData())

    def _distance_measure(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceResult:
        distance = wasserstein_distance(
            u_values=X_ref_,
            v_values=X,
            u_weights=kwargs.get("u_weights", None),
            v_weights=kwargs.get("v_weights", None),
        )
        distance = DistanceResult(distance=distance)
        return distance
