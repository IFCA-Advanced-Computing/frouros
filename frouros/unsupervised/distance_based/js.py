"""JS (Jensen-Shannon distance) module."""

import numpy as np  # type: ignore
from scipy.spatial.distance import jensenshannon  # type: ignore

from frouros.unsupervised.distance_based.base import (
    DistanceProbabilityBasedEstimator,
    DistanceResult,
)


class JS(DistanceProbabilityBasedEstimator):
    """JS (Jensen-Shannon distance) algorithm class."""

    def _distance_measure(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceResult:
        X_ref_rvs, X_rvs = self._calculate_probabilities(  # noqa: N806
            X_ref_=X_ref_, X=X
        )
        distance = jensenshannon(p=X_ref_rvs, q=X_rvs, base=kwargs.get("base", None))
        distance = DistanceResult(distance=distance)
        return distance
