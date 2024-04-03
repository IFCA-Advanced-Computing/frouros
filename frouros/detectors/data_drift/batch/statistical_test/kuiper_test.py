"""Kuiper's test module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.special import comb, factorial
from scipy.stats import ks_2samp

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class KuiperTest(BaseStatisticalTest):
    """Kuiper's test [kuiper1960tests]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :References:

    .. [kuiper1960tests] Kuiper, Nicolaas H.
        "Tests concerning random points on a circle."
        Nederl. Akad. Wetensch. Proc. Ser. A. Vol. 63. No. 1. 1960.

    :Example:

    >>> from frouros.detectors.data_drift import KuiperTest
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = KuiperTest()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    StatisticalResult(statistic=0.55, p_value=5.065664859580971e-13)
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
    ) -> None:
        super().__init__(
            data_type=NumericalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )

    @staticmethod
    def _statistical_test(
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs: Any,
    ) -> StatisticalResult:
        statistic, p_value = KuiperTest._kuiper(
            X=X_ref,
            Y=X,
            **kwargs,
        )
        test = StatisticalResult(
            statistic=statistic,
            p_value=p_value,
        )
        return test

    @staticmethod
    def _false_positive_probability(
        D: float,  # noqa: N803
        N: float,
    ) -> float:
        """
        Compute the false positive probability for the Kuiper's test.

        NOTE: This function is a modified version of the
        `kuiper_false_positive_probability` function from the
        `astropy.stats` <https://docs.astropy.org/en/stable/api/astropy.stats.kuiper_false_positive_probability.html#astropy.stats.kuiper_false_positive_probability> package.

        :param D: Kuiper's statistic
        :param D: float
        :param N: effective size of the sample
        :type N: float
        :return: false positive probability
        :rtype: float
        """  # noqa: E501
        if D < 2.0 / N:
            return 1.0 - factorial(N) * (D - 1.0 / N) ** (N - 1)

        if D < 3.0 / N:
            k = -(N * D - 1.0) / 2.0
            r = np.sqrt(k**2 - (N * D - 2.0) ** 2 / 2.0)
            a, b = -k + r, -k - r
            return 1 - (
                factorial(N - 1)
                * (b ** (N - 1) * (1 - a) - a ** (N - 1) * (1 - b))
                / N ** (N - 2)
                / (b - a)
            )

        if (D > 0.5 and N % 2 == 0) or (D > (N - 1.0) / (2.0 * N) and N % 2 == 1):
            # NOTE: the upper limit of this sum is taken from Stephens 1965
            t = np.arange(np.floor(N * (1 - D)) + 1)
            y = D + t / N
            Tt = y ** (t - 3) * (  # noqa: N806
                y**3 * N
                - y**2 * t * (3 - 2 / N)
                + y * t * (t - 1) * (3 - 2 / N) / N
                - t * (t - 1) * (t - 2) / N**2
            )
            term1 = comb(N, t)
            term2 = (1 - D - t / N) ** (N - t - 1)
            # term1 is formally finite, but is approximated by numpy as np.inf for
            # large values, so we set them to zero manually when they would be
            # multiplied by zero anyway
            term1[(term1 == np.inf) & (term2 == 0)] = 0.0
            return (Tt * term1 * term2).sum()

        z = D * np.sqrt(N)
        # When m*z>18.82 (sqrt(-log(finfo(double))/2)), exp(-2m**2z**2)
        # underflows.  Cutting off just before avoids triggering a (pointless)
        # underflow warning if `under="warn"`.
        ms = np.arange(1, 18.82 / z)
        S1 = (  # noqa: N806
            2 * (4 * ms**2 * z**2 - 1) * np.exp(-2 * ms**2 * z**2)
        ).sum()
        S2 = (  # noqa: N806
            ms**2 * (4 * ms**2 * z**2 - 3) * np.exp(-2 * ms**2 * z**2)
        ).sum()
        return S1 - 8 * D / 3 * S2

    @staticmethod
    def _kuiper(
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
    ) -> tuple[float, float]:
        """Kuiper's test.

        :param X: reference data
        :type X: numpy.ndarray
        :param Y: test data
        :type Y: numpy.ndarray
        :return: Kuiper's statistic and p-value
        :rtype: tuple[float, float]
        """
        X = np.sort(X)  # noqa: N806
        Y = np.sort(Y)  # noqa: N806
        X_size = len(X)  # noqa: N806
        Y_size = len(Y)  # noqa: N806

        statistic = ks_2samp(
            data1=X,
            data2=Y,
            alternative="two-sided",
        ).statistic
        sample_effective_size = X_size * Y_size / float(X_size + Y_size)

        p_value = KuiperTest._false_positive_probability(
            D=statistic,
            N=sample_effective_size,
        )

        return statistic, p_value
