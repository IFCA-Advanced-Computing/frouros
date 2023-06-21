"""Geometric Moving Average module."""

from frouros.detectors.concept_drift.streaming.change_detection.base import (
    BaseCUSUM,
    BaseCUSUMConfig,
    AlphaConfig,
)


class GeometricMovingAverageConfig(BaseCUSUMConfig, AlphaConfig):
    """Geometric Moving Average [robertst1959control]_ configuration.

    :References:

    .. [robertst1959control] Roberts, S. W.
        “Control Chart Tests Based on Geometric Moving Averages.”
        Technometrics, vol. 1, no. 3, 1959, pp. 239–50.
        JSTOR, https://doi.org/10.2307/1266443.
    """

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
        BaseCUSUMConfig.__init__(
            self, lambda_=lambda_, min_num_instances=min_num_instances
        )
        AlphaConfig.__init__(self, alpha=alpha)


class GeometricMovingAverage(BaseCUSUM):
    """Geometric Moving Average [robertst1959control]_ detector.

    :References:

    .. [robertst1959control] Roberts, S. W.
        “Control Chart Tests Based on Geometric Moving Averages.”
        Technometrics, vol. 1, no. 3, 1959, pp. 239–50.
        JSTOR, https://doi.org/10.2307/1266443.
    """

    config_type = GeometricMovingAverageConfig  # type: ignore

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = self.config.alpha * self.sum_ + (  # type: ignore
            1 - self.config.alpha  # type: ignore
        ) * (error_rate - self.mean_error_rate.mean)
