"""MMD (Maximum Mean Discrepancy) module."""

from typing import Any, Callable, Dict, Optional, List, Tuple, Union

import numpy as np  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.data_drift.base import NumericalData, MultivariateData
from frouros.detectors.data_drift.batch import MMD as MMDBatch  # noqa: N811
from frouros.detectors.data_drift.batch.distance_based.mmd import rbf_kernel
from frouros.detectors.data_drift.streaming.distance_based.base import (
    BaseDistanceBased,
    DistanceResult,
)
from frouros.utils.data_structures import CircularQueue


class MMD(BaseDistanceBased):
    """MMD (Maximum Mean Discrepancy) [gretton2012kernel]_ detector.

    :References:

    .. [gretton2012kernel] Gretton, Arthur, et al.
        "A kernel two-sample test."
        The Journal of Machine Learning Research 13.1 (2012): 723-773.
    """

    def __init__(
        self,
        window_size: int,
        kernel: Callable = rbf_kernel,
        chunk_size: Optional[int] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param window_size: window size
        :type window_size: int
        :param kernel: kernel function
        :type kernel: Callable
        :param chunk_size: chunk size value
        :type chunk_size: Optional[int]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        super().__init__(
            data_type=NumericalData(),
            statistical_type=MultivariateData(),
            callbacks=callbacks,
        )
        self.mmd = MMDBatch(
            kernel=kernel,
            chunk_size=chunk_size,
        )
        self.window_size = window_size
        self.X_queue = CircularQueue(max_len=self.window_size)

    @property
    def window_size(self) -> int:
        """Window size property.

        :return: window size
        :rtype: int
        """
        return self._window_size

    @window_size.setter
    def window_size(self, value: int) -> None:
        """Window size setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("window_size value must be greater than 0.")
        self._window_size = value

    def _fit(self, X: np.ndarray) -> None:  # noqa: N803
        self.mmd.fit(X=X)
        self.X_ref = self.mmd.X_ref

    def _reset(self) -> None:
        super()._reset()
        self.mmd.reset()

    def _update(self, value: Union[int, float]) -> Optional[DistanceResult]:
        self.X_queue.enqueue(value=value)

        if self.num_instances < self.window_size:
            return None

        # FIXME: Handle callback logs. Now are ignored.  # pylint: disable=fixme
        distance, _ = self.mmd.compare(X=np.array(self.X_queue))
        return distance

    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
    ) -> Tuple[Optional[DistanceResult], Dict[str, Any]]:  # noqa: N803
        return self.mmd.compare(X=X)
