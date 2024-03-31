"""CUSUM module."""

from typing import Optional, Union

import numpy as np

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.change_detection.base import (
    BaseCUSUM,
    BaseCUSUMConfig,
    DeltaConfig,
)


class CUSUMConfig(BaseCUSUMConfig, DeltaConfig):
    """CUSUM [page1954continuous]_ configuration.

    :param delta: delta value, defaults to 0.005
    :type delta: float
    :param lambda_: delta value, defaults to 50.0
    :type lambda_: float
    :param min_num_instances: minimum numbers of instances to start looking for changes, defaults to 30
    :type min_num_instances: int

    :References:

    .. [page1954continuous] Page, Ewan S.
        "Continuous inspection schemes."
        Biometrika 41.1/2 (1954): 100-115.
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        min_num_instances: int = 30,
    ) -> None:
        BaseCUSUMConfig.__init__(
            self, lambda_=lambda_, min_num_instances=min_num_instances
        )
        DeltaConfig.__init__(self, delta=delta)


class CUSUM(BaseCUSUM):
    """CUSUM [page1954continuous]_ detector.

    :param config: configuration object of the detector, defaults to None. If None, the default configuration of :class:`CUSUMConfig` is used.
    :type config: Optional[CUSUMConfig]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]]

    :References:

    .. [page1954continuous] Page, Ewan S.
        "Continuous inspection schemes."
        Biometrika 41.1/2 (1954): 100-115.

    :Example:

    >>> from frouros.detectors.concept_drift import CUSUM
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> dist_a = np.random.normal(loc=0.2, scale=0.01, size=1000)
    >>> dist_b = np.random.normal(loc=0.8, scale=0.04, size=1000)
    >>> stream = np.concatenate((dist_a, dist_b))
    >>> detector = CUSUM()
    >>> for i, value in enumerate(stream):
    ...     _ = detector.update(value=value)
    ...     if detector.drift:
    ...         print(f"Change detected at step {i}")
    ...         break
    Change detected at step 1086
    """  # noqa: E501  # pylint: disable=line-too-long

    config_type = CUSUMConfig  # type: ignore

    def __init__(  # noqa: D107
        self,
        config: Optional[CUSUMConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        super().__init__(
            config=config,
            callbacks=callbacks,
        )

    def _update_sum(self, error_rate: float) -> None:
        self.sum_ = np.maximum(
            0,
            self.sum_ + error_rate - self.mean_error_rate.mean - self.config.delta,  # type: ignore # noqa: E501
        )
