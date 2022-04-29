"""PSI (Population Stability Index) module."""

import sys

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

from frouros.unsupervised.distance_based.base import (  # type: ignore
    DistanceBasedEstimator,
)


class PSI(DistanceBasedEstimator, BaseEstimator, TransformerMixin):
    """PSI (Population Stability Index) algorithm class."""

    def __init__(self, num_buckets: int = 10) -> None:
        """Init method.

        :param num_buckets: number of buckets
        :type num_buckets: int
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.X_ref_num = None  # pylint: disable=invalid-name

    @property
    def num_buckets(self) -> int:
        """Number of buckets.

        :return: number of buckets
        :rtype: int
        """
        return self._num_buckets

    @num_buckets.setter
    def num_buckets(self, value: int) -> None:
        """Number of buckets setter.

        :param value: value to be set
        :type value: Optional[List[Tuple[float, float]]]
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("num buckets must be greater than 0.")
        self._num_buckets = value

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray = None,  # pylint: disable=W0613
    ):
        """Fit estimator.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :return fitted estimator
        :rtype: self
        """
        super().fit(X=X, y=y)
        self.X_ref_num = self.X_ref_.shape[0]  # type: ignore
        return self

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> float:
        distance = self._distance(
            X_ref_=X_ref_,
            X=X,
            func=self._psi,
            num_buckets=self.num_buckets,
            X_ref_num=self.X_ref_num,
            **kwargs
        )
        return distance

    @staticmethod
    def _distance(X_ref_: np.ndarray, X: np.ndarray, **kwargs) -> float:  # noqa: N803
        func = kwargs.pop("func")
        distance = func(X_ref_=X_ref_, X=X, **kwargs)
        return distance

    @staticmethod
    def _psi(
        X_ref_: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        X_ref_num: int,  # noqa: N803  # pylint: disable=invalid-name
        num_buckets: int,
    ) -> float:
        X_ref_percents = (  # noqa: N806  # pylint: disable=invalid-name
            np.histogram(a=X_ref_, bins=num_buckets)[0] / X_ref_num
        )
        X_percents = np.histogram(  # noqa: N806  # pylint: disable=invalid-name
            a=X, bins=num_buckets
        )[0] / len(
            X  # noqa: N806
        )

        # Replace 0.0 values with the smallest number possible
        # in order to avoid division by zero
        X_ref_percents[X_ref_percents == 0.0] = sys.float_info.min
        X_percents[X_percents == 0.0] = sys.float_info.min

        psi = np.sum(
            (X_percents - X_ref_percents) * np.log(X_percents / X_ref_percents)
        )
        return psi