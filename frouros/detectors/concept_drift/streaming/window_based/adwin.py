"""ADWIN (ADaptive WINdowing) module."""

from collections import deque
from typing import List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.window_based.base import (
    BaseWindowConfig,
    BaseWindow,
)


class Bucket:
    """Class representing a bucket."""

    def __init__(self, m: int) -> None:
        """Init method.

        :param m: controls the amount of memory used and
        the closeness of the cutpoints checked
        :type m: int
        """
        self.array_size = m + 1
        self.total = np.zeros(self.array_size)
        self.variance = np.zeros(self.array_size)
        self.idx = 0

    @property
    def array_size(self) -> int:
        """Array size property.

        :return: array size
        :rtype: int
        """
        return self._array_size

    @array_size.setter
    def array_size(self, value: int) -> None:
        """Array size setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 2:
            raise ValueError("array_size value must be greater than 1.")
        self._array_size = value

    @property
    def total(self) -> np.ndarray:
        """Total array property.

        :return: total array
        :rtype: numpy.ndarray
        """
        return self._total

    @total.setter
    def total(self, value: np.ndarray) -> None:
        """Total array setter.

        :param value: value to be set
        :type value: numpy.ndarray
        :raises TypeError: Type error exception
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("value must be of type numpy.ndarray.")
        self._total = value

    @property
    def variance(self) -> np.ndarray:
        """Variance array property.

        :return: variance array
        :rtype: numpy.ndarray
        """
        return self._variance

    @variance.setter
    def variance(self, value: np.ndarray) -> None:
        """Variance array setter.

        :param value: value to be set
        :type value: numpy.ndarray
        :raises TypeError: Type error exception
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("value must be of type numpy.ndarray.")
        self._variance = value

    @property
    def idx(self) -> int:
        """Current index property.

        :return: current index
        :rtype: int
        """
        return self._idx

    @idx.setter
    def idx(self, value: int):
        """Current index setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("idx value must be greater or equal than 0.")
        self._idx = value

    def reset(self) -> None:
        """Reset bucket statistics."""
        self.total = np.zeros(self.array_size)
        self.variance = np.zeros(self.array_size)

    def insert_data(self, value: float, variance: float) -> None:
        """Insert data at the end.

        :param value: value to be stored
        :type value: float
        :param variance: variance to be stored
        :type variance: float
        """
        self.total[self.idx] = value
        self.variance[self.idx] = variance
        self.idx += 1

    def remove(self) -> None:
        """Remove first."""
        self.compress(num_items_deleted=1)

    def compress(self, num_items_deleted: int) -> None:
        """Compress bucket index.

        :param num_items_deleted: number of items to delete
        :type num_items_deleted: int
        """
        for i in range(num_items_deleted, self.array_size):
            idx = i - num_items_deleted
            self.total[idx] = self.total[i]
            self.variance[idx] = self.variance[i]

        idx_start = self.array_size - num_items_deleted
        # fmt: off
        self.total[idx_start:self.array_size] = 0.0
        self.variance[idx_start:self.array_size] = 0.0
        # fmt: on

        self.idx -= num_items_deleted


class ADWINConfig(BaseWindowConfig):
    """ADWIN (ADaptive WINdowing) [bifet2007learning]_ configuration.

    :References:

    .. [bifet2007learning] Bifet, Albert, and Ricard Gavalda.
        "Learning from time-changing data with adaptive windowing."
        Proceedings of the 2007 SIAM international conference on data mining.
        Society for Industrial and Applied Mathematics, 2007.
    """

    def __init__(
        self,
        clock: int = 32,
        delta: float = 0.002,
        m: int = 5,
        min_window_size: int = 5,
        min_num_instances: int = 10,
    ) -> None:
        """Init method.

        :param clock: clock value
        :type clock: int
        :param delta: confidence value
        :type delta: float
        :param m: controls the amount of memory used and
        the closeness of the cutpoints checked
        :type m: int
        :param min_window_size: minimum numbers of instances
        per window to start looking for changes
        :type min_window_size: int
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.clock = clock
        self.delta = delta
        self.m = m
        self.min_window_size = min_window_size

    @property
    def clock(self) -> int:
        """Clock value property.

        :return: confidence interval to determine if drift is occurring
        :rtype: int
        """
        return self._clock

    @clock.setter
    def clock(self, value: int) -> None:
        """Clock value setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("clock value must be greater than 0.")
        self._clock = value

    @property
    def delta(self) -> float:
        """Delta value property.

        :return: confidence interval to determine if drift is occurring
        :rtype: float
        """
        return self._delta

    @delta.setter
    def delta(self, value: float) -> None:
        """Delta value setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0 < value < 1:
            raise ValueError("delta value must be in the range (0, 1).")
        self._delta = value

    @property
    def m(self) -> int:
        """M value property.

        :return: controls the amount of memory used and the closeness
        of the cutpoints checked
        :rtype: int
        """
        return self._m

    @m.setter
    def m(self, value: int) -> None:
        """M value setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("m value must be greater than 0.")
        self._m = value

    @property
    def min_window_size(self) -> int:
        """Minimum window size value property.

        :return: minimum window size value per each window
        :rtype: int
        """
        return self._min_window_size

    @min_window_size.setter
    def min_window_size(self, value: int) -> None:
        """Minimum window size value setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("min_window_size value must be greater than 0.")
        self._min_window_size = value


class ADWIN(BaseWindow):
    """ADWIN (ADaptive WINdowing) [bifet2007learning]_ detector.

    :References:

    .. [bifet2007learning] Bifet, Albert, and Ricard Gavalda.
        "Learning from time-changing data with adaptive windowing."
        Proceedings of the 2007 SIAM international conference on data mining.
        Society for Industrial and Applied Mathematics, 2007.
    """

    config_type = ADWINConfig

    def __init__(
        self,
        config: Optional[ADWINConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[ADWINConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        num_buckets = 0
        self.additional_vars = {
            "buckets": deque([Bucket(m=self.config.m)]),  # type: ignore
            "total": 0.0,
            "variance": 0.0,
            "width": 0,
            "num_buckets": num_buckets,
            "num_max_buckets": num_buckets,
        }
        self._set_additional_vars_callback()
        self._min_instances = self.config.min_num_instances + 1

    @property
    def buckets(self) -> deque:
        """Buckets queue property.

        :return: buckets queue
        :rtype: deque
        """
        return self._additional_vars["buckets"]

    @buckets.setter
    def buckets(self, value: deque):
        """Buckets queue setter.

        :param value: value to be set
        :type value: int
        :raises TypeError: Type error exception
        """
        if not isinstance(value, deque):
            raise TypeError("value must be of type deque.")
        self._additional_vars["buckets"] = value

    @property
    def total(self) -> float:
        """Total value property.

        :return: total value
        :rtype: float
        """
        return self._additional_vars["total"]

    @total.setter
    def total(self, value: float) -> None:
        """Total value setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0.0:
            raise ValueError("total value must be greater or equal than 0.0.")
        self._additional_vars["total"] = value

    @property
    def variance(self) -> float:
        """Variance value property.

        :return: variance value
        :rtype: float
        """
        return self._additional_vars["variance"]

    @variance.setter
    def variance(self, value: float) -> None:
        """Variance value setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        # FIXME: Workaround to avoid precision problems  # pylint: disable=fixme
        # if value < 0.0:
        #     raise ValueError("variance value must be greater or equal than 0.0.")
        self._additional_vars["variance"] = value

    @property
    def variance_window(self) -> float:
        """Variance in window value property.

        :return: variance in window value
        :rtype: float
        """
        return self.variance / self.width

    @property
    def width(self) -> int:
        """Width value property.

        :return: width value
        :rtype: int
        """
        return self._additional_vars["width"]

    @width.setter
    def width(self, value: int) -> None:
        """Width value setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("width value must be greater or equal than 0.")
        self._additional_vars["width"] = value

    @property
    def num_buckets(self) -> int:
        """Number of buckets property.

        :return: number of buckets
        :rtype: int
        """
        return self._additional_vars["num_buckets"]

    @num_buckets.setter
    def num_buckets(self, value: int) -> None:
        """Number of buckets setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_buckets value must be greater or equal than 0.")
        self._additional_vars["num_buckets"] = value

    @property
    def num_max_buckets(self) -> int:
        """Maximum number of buckets property.

        :return: maximum number of buckets
        :rtype: int
        """
        return self._additional_vars["num_max_buckets"]

    @num_max_buckets.setter
    def num_max_buckets(self, value: int) -> None:
        """Maximum number of buckets setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_max_buckets value must be greater or equal than 0.")
        self._additional_vars["num_max_buckets"] = value

    def _insert_bucket(self, value: float) -> None:
        self._insert_bucket_data(variance=0.0, value=value, bucket=self.buckets[0])
        self.width += 1
        incremental_variance = (
            (self.width - 1)
            * (value - self.total / (self.width - 1))
            * (value - self.total / (self.width - 1))
            / self.width
            if self.width > 1
            else 0.0
        )
        self.variance += incremental_variance
        self.total += value
        self._compress_buckets()

    def _insert_bucket_data(
        self, value: float, variance: float, bucket: Bucket
    ) -> None:
        bucket.insert_data(value=value, variance=variance)
        self.num_buckets += 1
        if self.num_buckets > self.num_max_buckets:
            self.num_max_buckets = self.num_buckets

    @staticmethod
    def _bucket_size(index: int) -> int:
        return np.power(2, index)

    def _delete_bucket(self) -> int:
        bucket = self.buckets[-1]
        bucket_size = self._bucket_size(index=len(self.buckets) - 1)
        self.width -= bucket_size
        self.total -= bucket.total[0]
        bucket_mean = bucket.total[0] / bucket_size
        window_mean = self.total / self.width
        incremental_variance = bucket.variance[0] + bucket_size * self.width * (
            bucket_mean - window_mean
        ) * (bucket_mean - window_mean) / (bucket_size + self.width)
        self.variance -= incremental_variance

        bucket.remove()
        self.num_buckets -= 1
        if bucket.idx == 0:
            self.buckets.pop()

        return bucket_size

    def _compress_buckets(self) -> None:
        bucket = self.buckets[0]
        idx = 0
        while bucket is not None:
            idx_next = idx + 1
            if bucket.idx == bucket.array_size:
                try:
                    next_bucket = self.buckets[idx_next]
                except IndexError:
                    self.buckets.append(Bucket(m=self.config.m))  # type: ignore
                    next_bucket = self.buckets[-1]
                bucket_size = self._bucket_size(index=idx)
                bucket_1_mean, bucket_2_mean = bucket.total[0:2] / bucket_size
                incremental_variance = (
                    bucket_size
                    * bucket_size
                    * (bucket_1_mean - bucket_2_mean)
                    * (bucket_1_mean - bucket_2_mean)
                    / (bucket_size * 2)
                )
                total = np.sum(bucket.total[0:2])
                variance = np.sum(bucket.variance[0:2]) + incremental_variance
                next_bucket.insert_data(value=total, variance=variance)
                self.num_buckets += 1
                bucket.compress(num_items_deleted=2)

                if next_bucket.idx <= self.config.m:  # type: ignore
                    break
            else:
                break

            try:
                bucket = self.buckets[idx_next]
            except IndexError:
                bucket = None
            idx += 1

    def _calculate_threshold(self, w0_instances: int, w1_instances: int) -> float:
        delta_prime = np.log(2 * np.log(self.width) / self.config.delta)  # type: ignore
        # Has highlighted in river library, the use of the reciprocal
        # of m allows to avoid extra divisions
        min_window_size = self.config.min_window_size + 1  # type: ignore
        m_reciprocal = 1 / (w0_instances - min_window_size) + 1 / (
            w1_instances - min_window_size
        )
        epsilon = (
            np.sqrt(2 * m_reciprocal * self.variance_window * delta_prime)
            + 2 / 3 * delta_prime * m_reciprocal
        )
        return epsilon

    def _update(self, value: Union[int, float], **kwargs) -> None:
        # pylint: disable=too-many-locals, too-many-nested-blocks
        # NOTE: Refactor function
        self.num_instances += 1
        self._insert_bucket(value=value)

        if (
            self.num_instances % self.config.clock == 0  # type: ignore
            and self.width > self.config.min_num_instances  # type: ignore
        ):
            flag_reduce_width = True

            while flag_reduce_width:
                flag_reduce_width = False
                flag_exit = False
                w0_instances = 0
                w1_instances = self.width
                w0_total = 0
                w1_total = self.total

                for i in range(len(self.buckets) - 1, -1, -1):
                    if flag_exit:
                        break
                    bucket = self.buckets[i]
                    for j in range(bucket.idx - 1):
                        bucket_size = self._bucket_size(index=i)

                        w0_instances += bucket_size
                        w1_instances -= bucket_size
                        w0_total += bucket.total[j]
                        w1_total -= bucket.total[j]

                        if i == 0 and j == bucket.idx - 1:
                            flag_exit = True
                            break

                        if (
                            w1_instances > self.config.min_window_size  # type: ignore
                            and (
                                w0_instances
                                > self.config.min_window_size  # type: ignore
                            )
                        ):
                            w0_mean = w0_total / w0_instances
                            w1_mean = w1_total / w1_instances
                            threshold = self._calculate_threshold(
                                w0_instances=w0_instances, w1_instances=w1_instances
                            )
                            if np.abs(w0_mean - w1_mean) > threshold:
                                # Drift detected
                                flag_reduce_width = True
                                self.drift = True
                                if self.width > 0:
                                    w0_instances -= self._delete_bucket()
                                    flag_exit = True
                                    break

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.buckets = deque([Bucket(m=self.config.m)])  # type: ignore
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.num_buckets = 0
        self.num_max_buckets = self.num_buckets
