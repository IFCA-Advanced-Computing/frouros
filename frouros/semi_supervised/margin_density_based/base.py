"""Semi-supervised margin density based base module."""

import abc
from typing import Any, Dict, Tuple, Union

import numpy as np  # type: ignore

from frouros.semi_supervised.base import (
    SemiSupervisedBaseEstimator,
    SemiSupervisedBaseConfig,
)


class MarginDensityBasedConfig(SemiSupervisedBaseConfig):
    """Class representing a margin density based configuration class."""


class MarginDensityBasedEstimator(SemiSupervisedBaseEstimator):
    """Abstract class representing a margin density based estimator."""

    def update(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Update drift detector.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: input data
        :type y: numpy.ndarray
        """
        self.X_samples.extend(X)
        self.y_samples.extend(y)

    def _calculate_md_ref(self, X: np.ndarray) -> float:  # noqa: N803
        return np.sum(self._calculate_margin_signal(X)) / X.shape[0]

    @staticmethod
    def _get_predict_response(
        drift_suspected: bool,
        drift_confirmed: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        response = {
            "drift_suspected": drift_suspected,
            "drift_confirmed": drift_confirmed,
        }
        response.update(**kwargs)  # type: ignore
        return response

    @abc.abstractmethod
    def predict(
        self, X: np.ndarray  # noqa: N803
    ) -> Tuple[np.ndarray, Dict[str, Union[bool, float]]]:
        """Predict abstract method."""

    @abc.abstractmethod
    def _calculate_margin_signal(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        pass
