"""CUSUM module."""

import numpy as np  # type: ignore

from frouros.detectors.concept_drift.cusum_based.base import (
    CUSUMBase,
    CUSUMBaseConfig,
    DeltaConfig,
)


class CUSUMConfig(CUSUMBaseConfig, DeltaConfig):
    """CUSUM configuration class."""

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param delta: delta value
        :type delta: float
        :param lambda_: delta value
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        CUSUMBaseConfig.__init__(
            self, lambda_=lambda_, min_num_instances=min_num_instances
        )
        DeltaConfig.__init__(self, delta=delta)


class CUSUM(CUSUMBase):
    """CUSUM algorithm class."""

    config_type = CUSUMConfig  # type: ignore

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = np.maximum(
            0,
            self.sum_
            + error_rate
            - self.mean_error_rate.mean
            - self.config.delta,  # type: ignore
        )