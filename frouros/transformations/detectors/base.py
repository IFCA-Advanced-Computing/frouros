"""Abstract class .... module."""

import abc

from typing import List, Optional
from joblib import Parallel  # type: ignore

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.utils.fixes import delayed  # type: ignore
import numpy as np  # type: ignore

from frouros.unsupervised.base import UnsupervisedBaseEstimator


class BaseDetectors(abc.ABC, BaseEstimator, TransformerMixin):
    """Abstract class ."""

    def __init__(
        self,
        detectors: List[UnsupervisedBaseEstimator],
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        """Init method.

        :param detectors: detectors to use
        :type detectors: List[UnsupervisedBaseEstimator]
        :param n_jobs: number of jobs to use
        :type n_jobs: Optional[int]
        :param verbose: verbosity level
        :type verbose: int
        """
        self.detectors = detectors
        self.n_jobs = n_jobs
        self.verbose = verbose

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
        :return: fitted estimator
        :rtype: self
        """
        X_preprocessed = self.preprocess_x(X=X)  # noqa: N806
        _ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(delayed(detector.fit)(X=X_preprocessed, y=y) for detector in self.detectors)
        return self

    def transform(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray = None,  # pylint: disable=W0613
        **kwargs,
    ) -> np.ndarray:
        """Transform values.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :return: transformed feature data
        :rtype: numpy.ndarray
        """
        X_preprocessed = self.preprocess_x(X=X)  # noqa: N806
        _ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads",)(
            delayed(detector.transform)(X=X_preprocessed, **kwargs)
            for detector in self.detectors
        )
        return X

    @property
    def detectors(self) -> List[UnsupervisedBaseEstimator]:
        """Detectors property.

        :return: detectors to use
        :rtype: List[UnsupervisedBaseEstimator]
        """
        return self._detectors

    @detectors.setter
    def detectors(self, value: List[UnsupervisedBaseEstimator]) -> None:
        """Detectors method setter.

        :param value: value to be set
        :type value: List[UnsupervisedBaseEstimator]
        :raises TypeError: Type error exception
        """
        if not all(isinstance(x, UnsupervisedBaseEstimator) for x in value):
            raise TypeError("value elements must be of type UnsupervisedBaseEstimator.")
        self._detectors = value

    @property
    def n_jobs(self) -> Optional[int]:
        """Number of jobs property.

        :return: number of jobs to use
        :rtype: Optional[int]
        """
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: Optional[int]) -> None:
        """Number of jobs method setter.

        :param value: value to be set
        :type value: Optional[int]
        :raises TypeError: Type error exception
        """
        if value is not None and not isinstance(value, int):
            raise TypeError("value must be of type None or int.")
        self._n_jobs = value

    @property
    def verbose(self) -> int:
        """Verbose level property.

        :return: verbosity level
        :rtype: int
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: int) -> None:
        """Verbose level method setter.

        :param value: value to be set
        :type value: int
        :raises TypeError: Type error exception
        """
        if not isinstance(value, int):
            raise TypeError("value must be of type int.")
        self._verbose = value

    @abc.abstractmethod
    def preprocess_x(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Abstract preprocess X method.

        :param X: feature data
        :type X: numpy.ndarray
        :return: preprocessed feature data
        :rtype: numpy.ndarray
        """
