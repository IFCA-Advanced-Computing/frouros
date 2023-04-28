"""MMD (Maximum Mean Discrepancy) module."""

from typing import Callable, Optional, List, Union

import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.data_drift.batch import MMD as MMDBatch  # noqa: N811
from frouros.detectors.data_drift.batch.distance_based.mmd import rbf_kernel
from frouros.detectors.data_drift.base import NumericalData, MultivariateData
from frouros.detectors.data_drift.streaming.distance_based.base import (
    DistanceBasedBase,
    DistanceResult,
)
from frouros.utils.data_structures import CircularQueue


class MMD(DistanceBasedBase):
    """MMD (Maximum Mean Discrepancy) [gretton2012kernel]_ detector.

    :References:

    .. [gretton2012kernel] Gretton, Arthur, et al.
        "A kernel two-sample test."
        The Journal of Machine Learning Research 13.1 (2012): 723-773.
    """

    def __init__(
        self,
        kernel: Callable = rbf_kernel,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
        window_size: int = 10,
    ) -> None:
        """Init method.

        :param kernel: kernel function
        :type kernel: Callable
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        :param window_size: window size
        :type window_size: int
        """
        super().__init__(
            data_type=NumericalData(),
            statistical_type=MultivariateData(),
            callbacks=callbacks,
        )
        self.mmd = MMDBatch(kernel=kernel)
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
        self.mmd.reset()

    def _update(self, value: Union[int, float]) -> Optional[DistanceResult]:
        self.X_queue.enqueue(value=value)

        if self.num_instances < self.window_size:
            return None

        # FIXME: Handle callback logs. Now are ignored.  # pylint: disable=fixme
        distance, _ = self.mmd.compare(X=np.array(self.X_queue))
        return distance
