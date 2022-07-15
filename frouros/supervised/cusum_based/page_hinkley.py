"""Page Hinkley module."""


from frouros.supervised.cusum_based.base import (
    CUSUMBaseEstimator,
    CUSUMBaseConfig,
)


class PageHinkleyConfig(CUSUMBaseConfig):
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
        super().__init__(
            min_num_instances=min_num_instances, delta=delta, lambda_=lambda_
        )
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        """Forgetting factor property.

        :return: forgetting factor value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Forgetting factor setter.

        :param value: forgetting factor value
        :type value: float
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("alpha must be in the range [0, 1].")
        self._alpha = value


class PageHinkley(CUSUMBaseEstimator):
    """Page Hinkley algorithm class."""

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = self.config.alpha * self.sum_ + (  # type: ignore
            error_rate - self.mean_error_rate - self.config.delta  # type: ignore
        )
