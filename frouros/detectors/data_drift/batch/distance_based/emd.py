"""EMD (Earth Mover's Distance) module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.stats import wasserstein_distance

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import UnivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBased,
    DistanceResult,
)


class EMD(BaseDistanceBased):
    """EMD (Earth Mover's Distance) [rubner2000earth]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]
    :param kwargs: additional keyword arguments to pass to scipy.stats.wasserstein_distance
    :type kwargs: Dict[str, Any]

    :References:

    .. [rubner2000earth] Rubner, Yossi, Carlo Tomasi, and Leonidas J. Guibas.
        "The earth mover's distance as a metric for image retrieval."
        International journal of computer vision 40.2 (2000): 99.

    :Example:

    >>> from frouros.detectors.data_drift import EMD
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = EMD()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    DistanceResult(distance=1.0686078744674332)
    """  # noqa: E501

    def __init__(  # noqa: D107
        self,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            statistical_type=UnivariateData(),
            statistical_method=self._emd,
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
        emd = self._emd(X=X_ref, Y=X, **self.kwargs)
        distance = DistanceResult(distance=emd)
        return distance

    @staticmethod
    def _emd(X: np.ndarray, Y: np.ndarray, **kwargs: Any) -> float:  # noqa: N803
        emd = wasserstein_distance(
            u_values=X.flatten(),
            v_values=Y.flatten(),
            **kwargs,
        )
        return emd
