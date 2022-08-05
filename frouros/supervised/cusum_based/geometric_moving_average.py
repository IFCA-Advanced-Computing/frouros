"""Geometric Moving Average module."""


from frouros.supervised.cusum_based.base import (
    CUSUMBaseEstimator,
    CUSUMBaseConfig,
    AlphaConfig,
)


class GeometricMovingAverageConfig(CUSUMBaseConfig, AlphaConfig):
    """Geometric Moving Average configuration class."""

    def __init__(
        self,
        alpha: float = 0.99,
        lambda_: float = 1.0,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha: forgetting factor value
        :type alpha: float
        :param lambda_: delta value
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        CUSUMBaseConfig.__init__(
            self, lambda_=lambda_, min_num_instances=min_num_instances
        )
        AlphaConfig.__init__(self, alpha=alpha)


class GeometricMovingAverage(CUSUMBaseEstimator):
    """Geometric Moving Average algorithm class."""

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = self.config.alpha * self.sum_ + (  # type: ignore
            1 - self.config.alpha  # type: ignore
        ) * (error_rate - self.mean_error_rate.mean)
