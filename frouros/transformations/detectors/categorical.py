"""Categorical detectors module."""

from typing import List, Optional

import numpy as np  # type: ignore

from frouros.transformations.detectors.base import BaseDetectors
from frouros.unsupervised.base import UnsupervisedBaseEstimator


class CategoricalDetectors(BaseDetectors):
    """Categorical detectors transformation."""

    def __init__(
        self,
        detectors: List[UnsupervisedBaseEstimator],
        columns: List[int],
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        """Init method.

        :param detectors: detectors to use
        :type detectors: List[Tuple[UnsupervisedBaseEstimator, List[int]]]
        :param columns: columns on which to apply the detectors
        :type columns: List[int]
        :param n_jobs: number of jobs to use
        :type n_jobs: Optional[int]
        :param verbose: verbosity level
        :type verbose: int
        """
        super().__init__(detectors=detectors, n_jobs=n_jobs, verbose=verbose)
        self.columns = columns

    def preprocess_x(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Preprocess X method.

        :param X: feature data
        :type X: numpy.ndarray
        :return: preprocessed feature data
        :rtype: numpy.ndarray
        """
        return X[:, self.columns]
