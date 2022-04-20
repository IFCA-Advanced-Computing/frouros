"""EMD (Earth Mover's Distance) module."""

import numpy as np  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

from frouros.unsupervised.distance_based.base import (  # type: ignore
    DistanceBasedEstimator,
)


class EMD(BaseEstimator, TransformerMixin, DistanceBasedEstimator):
    """EMD (Earth Mover's Distance) algorithm class."""

    @staticmethod
    def _distance(X_ref_: np.ndarray, X: np.ndarray, **kwargs) -> float:  # noqa: N803
        distance = wasserstein_distance(
            u_values=X_ref_,
            v_values=X,
            u_weights=kwargs.get("u_weights", None),
            v_weights=kwargs.get("v_weights", None),
        )
        return distance
