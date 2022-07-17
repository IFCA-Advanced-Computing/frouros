"""EMD (Earth Mover's Distance) module."""

import numpy as np  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore

from frouros.unsupervised.base import NumericalData, UnivariateTestType
from frouros.unsupervised.distance_based.base import (  # type: ignore
    DistanceBasedEstimator,
)


class EMD(DistanceBasedEstimator):
    """EMD (Earth Mover's Distance) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=NumericalData(), test_type=UnivariateTestType())

    def _distance(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> float:
        distance = wasserstein_distance(
            u_values=X_ref_,
            v_values=X,
            u_weights=kwargs.get("u_weights", None),
            v_weights=kwargs.get("v_weights", None),
        )
        return distance
