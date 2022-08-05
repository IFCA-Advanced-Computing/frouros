"""Page Hinkley module."""


from frouros.supervised.cusum_based.base import (
    CUSUMBaseEstimator,
    CUSUMBaseConfig,
    DeltaConfig,
    AlphaConfig,
)


class PageHinkleyConfig(CUSUMBaseConfig, DeltaConfig, AlphaConfig):
    """Page Hinkley configuration class."""

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        min_num_instances: int = 30,
        alpha: float = 0.9999,
    ) -> None:
        """Init method.

        :param delta: delta value
        :type delta: float
        :param lambda_: lambda value
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        :param alpha: forgetting factor value
        :type alpha: float
        """
        CUSUMBaseConfig.__init__(
            self, min_num_instances=min_num_instances, lambda_=lambda_
        )
        DeltaConfig.__init__(self, delta=delta)
        AlphaConfig.__init__(self, alpha=alpha)


class PageHinkley(CUSUMBaseEstimator):
    """Page Hinkley algorithm class."""

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = self.config.alpha * self.sum_ + (  # type: ignore
            error_rate - self.mean_error_rate.mean - self.config.delta  # type: ignore
        )
