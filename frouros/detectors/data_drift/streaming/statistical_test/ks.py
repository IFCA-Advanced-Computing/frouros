"""IncrementalKSTest (IncrementalKolmogorov-Smirnov test) module."""

# FIXME: There seem to be a bug on the treap DS. Uncomment all  # pylint: disable=fixme
#  the commented code lines when that is solved.

# from copy import deepcopy
from typing import Optional, List, Union

import numpy as np  # type: ignore
from scipy.stats._stats_py import (  # type: ignore
    _compute_prob_outside_square,
    _compute_outer_prob_inside_method,
)
from scipy.stats.distributions import kstwo  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.streaming.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)
from frouros.utils.data_structures import CircularQueue

# from frouros.utils.data_structures import Treap

# Value used in scipy
MAX_AUTO_N = 10000


class IncrementalKSTest(BaseStatisticalTest):
    """IncrementalKSTest (Incremental Kolmogorov-Smirnov test) [dos2016fast]_ detector.

    :References:

    .. [dos2016fast] dos Reis, Denis Moreira, et al.
        "Fast unsupervised online drift detection using incremental kolmogorov-smirnov
        test."
        Proceedings of the 22nd ACM SIGKDD international conference on knowledge
        discovery and data mining. 2016.
    """

    def __init__(
        self,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
        window_size: int = 10,
    ) -> None:
        """Init method.

        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        :param window_size: window size
        :type window_size: int
        """
        super().__init__(
            data_type=NumericalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )
        self.window_size = window_size
        self.gcd = None
        # self.treap = None
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
        if max(X.shape[0], self.window_size) <= MAX_AUTO_N:
            self.gcd = np.gcd(X.shape[0], self.window_size)

        # r = X.shape[0] / self.window_size
        # self.treap = Treap(r=r)  # type: ignore
        # for obs in X:
        #     self.treap.insert(obs=obs, group=0)  # type: ignore
        self.X_ref = np.sort(X)

    def _reset(self) -> None:
        # self.treap = None
        self.gcd = None
        self.X_queue.clear()

    def _update(self, value: Union[int, float]) -> Optional[StatisticalResult]:
        self.X_queue.enqueue(value=value)
        # self.treap.insert(obs=value, group=1)  # type: ignore

        if self.num_instances < self.window_size:
            return None

        test = self._statistical_test(
            X_ref=self.X_ref,
            X=np.array(self.X_queue),
            gcd=self.gcd,
            window_size=self.window_size,
            # treap=self.treap,  # type: ignore
            # p_value=self._target_p_value
        )
        # self.treap.remove(obs=self.X_queue[0], group=1)  # type: ignore
        return test

    @staticmethod
    def _statistical_test(
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs,
    ) -> StatisticalResult:
        # treap, p_value = kwargs["treap"], kwargs["p_value"]

        # Resampling if |A| > |B|
        # samples_diff = max(X_ref.shape[0] - X.shape[0], 0)
        # if samples_diff > 0:
        #     treap_resample = deepcopy(treap)
        #     for _ in range(samples_diff):
        #         sample = np.random.choice(X, replace=True)
        #         treap_resample.insert(obs=sample, group=1)
        # statistic = max(
        #     treap_resample.max,
        #     -treap_resample.min
        # ) / treap_resample.num_samples[0]
        # else:
        #     statistic = max(treap.max, -treap.min) / treap.num_samples[0]
        # n, m = treap.num_samples
        # FIXME: Implement a CircularQueue that maintain   # pylint: disable=fixme
        #  values ordered
        X_sorted = np.sort(X)  # noqa: N806

        statistic = IncrementalKSTest._calculate_statistic(X_ref=X_ref, X=X_sorted)

        p_value = (
            IncrementalKSTest._calculate_p_value_exact(
                X_ref_num_samples=X_ref.shape[0],
                statistic=statistic,
                gcd=kwargs["gcd"],
                window_size=kwargs["window_size"],
            )
            if max(X_ref.shape[0], X_sorted.shape[0]) <= MAX_AUTO_N
            else IncrementalKSTest._calculate_p_value_aprox(
                X_ref_num_samples=X_ref.shape[0],
                X_num_samples=X.shape[0],
                statistic=statistic,
            )
        )

        test = StatisticalResult(statistic=statistic, p_value=p_value)
        return test

    @staticmethod
    def _calculate_p_value_aprox(
        X_ref_num_samples: int,  # noqa: N803
        X_num_samples: int,
        statistic: float,
    ) -> float:
        # Uses scipy code adaptation to calculate approximate p-value
        n, m = float(X_ref_num_samples), float(X_num_samples)
        en = n * m / (n + m)
        p_value = kstwo.sf(statistic, np.round(en))[0]
        return p_value

    @staticmethod
    def _calculate_statistic(X_ref: np.ndarray, X: np.ndarray) -> float:  # noqa: N803
        # Uses scipy code adaptation to calculate statistic (distance)
        data_all = np.concatenate([X_ref, X])
        cdf1 = np.searchsorted(X_ref, data_all, side="right") / X_ref.shape[0]
        cdf2 = np.searchsorted(X, data_all, side="right") / X.shape[0]
        cddiffs = cdf1 - cdf2
        argmin_s = np.argmin(cddiffs)
        argmax_s = np.argmax(cddiffs)
        min_s = np.clip(-cddiffs[argmin_s], 0, 1)
        max_s = cddiffs[argmax_s]
        statistic = min_s if min_s > max_s else max_s
        return statistic

    @staticmethod
    def _calculate_p_value_exact(
        X_ref_num_samples: int,  # noqa: N803
        statistic: float,
        gcd: Union[int, float],
        window_size: int,
    ) -> float:
        # Uses scipy code adaptation to calculate exact p-value
        lcm = (X_ref_num_samples // gcd) * window_size
        h = int(np.round(statistic * lcm))
        if h == 0:
            return 1.0
        try:
            with np.errstate(invalid="raise", over="raise"):
                p_value = (
                    _compute_prob_outside_square(X_ref_num_samples, h)
                    if X_ref_num_samples == window_size
                    else _compute_outer_prob_inside_method(
                        X_ref_num_samples, window_size, gcd, h
                    )
                )
        except (FloatingPointError, OverflowError):
            return np.nan
        return p_value
