"""Geometric Moving Average module."""

from typing import Optional, Union

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.change_detection.base import (
    AlphaConfig,
    BaseCUSUM,
    BaseCUSUMConfig,
)


class GeometricMovingAverageConfig(BaseCUSUMConfig, AlphaConfig):
    """Geometric Moving Average [robertst1959control]_ configuration.

    :param alpha: forgetting factor value, defaults to 0.99
    :type alpha: float
    :param lambda_: delta value, defaults to 1.0
    :type lambda_: float
    :param min_num_instances: minimum numbers of instances to start looking for changes, defaults to 30
    :type min_num_instances: int

    :References:

    .. [robertst1959control] Roberts, S. W.
        “Control Chart Tests Based on Geometric Moving Averages.”
        Technometrics, vol. 1, no. 3, 1959, pp. 239–50.
        JSTOR, https://doi.org/10.2307/1266443.
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        alpha: float = 0.99,
        lambda_: float = 1.0,
        min_num_instances: int = 30,
    ) -> None:
        BaseCUSUMConfig.__init__(
            self, lambda_=lambda_, min_num_instances=min_num_instances
        )
        AlphaConfig.__init__(self, alpha=alpha)


class GeometricMovingAverage(BaseCUSUM):
    """Geometric Moving Average [robertst1959control]_ detector.

    :param config: configuration object of the detector, defaults to None. If None, the default configuration of :class:`GeometricMovingAverageConfig` is used.
    :type config: Optional[GeometricMovingAverageConfig]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]]

    :References:

    .. [robertst1959control] Roberts, S. W.
        “Control Chart Tests Based on Geometric Moving Averages.”
        Technometrics, vol. 1, no. 3, 1959, pp. 239–50.
        JSTOR, https://doi.org/10.2307/1266443.

    :Example:

    >>> from frouros.detectors.concept_drift import GeometricMovingAverage, GeometricMovingAverageConfig
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> dist_a = np.random.normal(loc=0.2, scale=0.01, size=1000)
    >>> dist_b = np.random.normal(loc=0.8, scale=0.04, size=1000)
    >>> stream = np.concatenate((dist_a, dist_b))
    >>> detector = GeometricMovingAverage(config=GeometricMovingAverageConfig(lambda_=0.3))
    >>> for i, value in enumerate(stream):
    ...     _ = detector.update(value=value)
    ...     if detector.drift:
    ...         print(f"Change detected at step {i}")
    ...         break
    Change detected at step 1071
    """  # noqa: E501  # pylint: disable=line-too-long

    config_type = GeometricMovingAverageConfig  # type: ignore

    def __init__(  # noqa: D107
        self,
        config: Optional[GeometricMovingAverageConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        super().__init__(
            config=config,
            callbacks=callbacks,
        )

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = self.config.alpha * self.sum_ + (  # type: ignore
            1 - self.config.alpha  # type: ignore
        ) * (error_rate - self.mean_error_rate.mean)
