"""CUSUM module."""


import numpy as np  # type: ignore

from frouros.supervised.cusum_based.base import (
    CUSUMBaseEstimator,
    CUSUMBaseConfig,
)


class CUSUMConfig(CUSUMBaseConfig):
    """CUSUM configuration class."""


class CUSUM(CUSUMBaseEstimator):
    """CUSUM algorithm class."""

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = np.maximum(
            0,
            self.sum_
            + error_rate
            - self.mean_error_rate
            - self.config.delta,  # type: ignore
        )
