"""Energy Distance module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.stats import energy_distance

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import UnivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBased,
    DistanceResult,
)


class EnergyDistance(BaseDistanceBased):
    """EnergyDistance [szekely2013energy]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]
    :param kwargs: additional keyword arguments to pass to scipy.stats.energy_distance
    :type kwargs: Dict[str, Any]

    :References:

    .. [szekely2013energy] Székely, Gábor J., and Maria L. Rizzo.
        "Energy statistics: A class of statistics based on distances."
        Journal of statistical planning and inference 143.8 (2013): 1249-1272.

    :Example:

    >>> from frouros.detectors.data_drift import EnergyDistance
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = EnergyDistance()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    DistanceResult(distance=0.8359206395514527)
    """  # noqa: E501

    def __init__(  # noqa: D107
        self,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            statistical_type=UnivariateData(),
            statistical_method=self._energy_distance,
            statistical_kwargs=kwargs,
            callbacks=callbacks,
        )
        self.kwargs = kwargs

    def _distance_measure(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        **kwargs: Any,
    ) -> DistanceResult:
        emd = self._energy_distance(X=X_ref, Y=X, **self.kwargs)
        distance = DistanceResult(distance=emd)
        return distance

    @staticmethod
    def _energy_distance(
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        energy = energy_distance(
            u_values=X.flatten(),
            v_values=Y.flatten(),
            **kwargs,
        )
        return energy
