"""CUSUM module."""

import numpy as np  # type: ignore

from frouros.detectors.concept_drift.streaming.change_detection.base import (
    BaseCUSUM,
    BaseCUSUMConfig,
    DeltaConfig,
)


class CUSUMConfig(BaseCUSUMConfig, DeltaConfig):
    """CUSUM [page1954continuous]_ configuration.

    :References:

    .. [page1954continuous] Page, Ewan S.
        "Continuous inspection schemes."
        Biometrika 41.1/2 (1954): 100-115.
    """

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
        BaseCUSUMConfig.__init__(
            self, lambda_=lambda_, min_num_instances=min_num_instances
        )
        DeltaConfig.__init__(self, delta=delta)


class CUSUM(BaseCUSUM):
    """CUSUM [page1954continuous]_ detector.

    :References:

    .. [page1954continuous] Page, Ewan S.
        "Continuous inspection schemes."
        Biometrika 41.1/2 (1954): 100-115.
    """

    config_type = CUSUMConfig  # type: ignore

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = np.maximum(
            0,
            self.sum_
            + error_rate
            - self.mean_error_rate.mean
            - self.config.delta,  # type: ignore
        )
