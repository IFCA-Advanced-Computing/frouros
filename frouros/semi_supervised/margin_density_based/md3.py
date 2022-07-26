"""MD3 (Margin Density Drift Detection) module."""

from typing import Callable, Dict, Optional, List, Union

import numpy as np  # type: ignore
from sklearn.ensemble import BaggingClassifier  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore

from frouros.semi_supervised.margin_density_based.base import (
    MarginDensityBasedEstimator,
    MarginDensityBasedConfig,
)


class MD3Config(MarginDensityBasedConfig):
    """MD3 (Margin Density Drift Detection) configuration class."""

    def __init__(
        self,
        chunk_size: int = 500,
        sensitivity: float = 2.0,
        num_folds: int = 5,
        num_training_samples: int = None,
    ) -> None:
        """Init method.

        :param chunk_size: chunk size used for lambda value
        :type chunk_size: int
        :param sensitivity: sensitivity value
        :type sensitivity: float
        :param num_folds: number of folds for K-Fold
        :type num_folds: int
        :param num_training_samples: number of training samples to used for retrain
        :type num_training_samples: int
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.forgetting_factor = (self.chunk_size - 1) / self.chunk_size  # lambda
        self.num_folds = num_folds
        self.num_training_samples = (
            num_training_samples if num_training_samples else self.chunk_size
        )
        self.sensitivity = sensitivity
        self.signal_factor = 1 - self.forgetting_factor  # 1 - lambda


class MD3SVMConfig(MD3Config):
    """MD3-SVM (Margin Density Drift Detection SVM) configuration class."""


class MD3RSConfig(MD3Config):
    """MD3-RS (Margin Density Drift Detection Random Subspace) configuration class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        chunk_size: int = 500,
        sensitivity: float = 2.0,
        num_folds: int = 5,
        num_training_samples: int = None,
        num_estimators: int = 20,
        ratio_random_features: float = 0.5,
        margin_uncertainty: float = 0.5,
    ) -> None:
        """Init method.

        :param chunk_size: chunk size used for lambda value
        :type chunk_size: int
        :param sensitivity: sensitivity value
        :type sensitivity: float
        :param num_folds: number of folds for K-Fold
        :type num_folds: int
        :param num_training_samples: number of training samples to used for retrain
        :type num_training_samples: int
        :param ratio_random_features: ratio of random features to used for train
        :type ratio_random_features: float
        :param margin_uncertainty: margin of uncertainty to be used by the signal
        :type margin_uncertainty: float
        """
        super().__init__(
            chunk_size=chunk_size,
            sensitivity=sensitivity,
            num_folds=num_folds,
            num_training_samples=num_training_samples,
        )
        self.num_estimators = num_estimators
        self.ratio_random_features = ratio_random_features
        self.margin_uncertainty = margin_uncertainty

    @property
    def margin_uncertainty(self) -> float:
        """Margin of uncertainty property.

        :return: margin of uncertainty
        :rtype: float
        """
        return self._margin_uncertainty

    @margin_uncertainty.setter
    def margin_uncertainty(self, value: float) -> None:
        """Margin of uncertainty setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 < value < 1.0:
            raise ValueError("margin_uncertainty value must be in the range (0, 1).")
        self._margin_uncertainty = value

    @property
    def num_estimators(self) -> int:
        """Number of estimators property.

        :return: number of estimators
        :rtype: int
        """
        return self._num_estimators

    @num_estimators.setter
    def num_estimators(self, value: int) -> None:
        """Number of estimators setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("num_estimators value must be greater than 1.")
        self._num_estimators = value

    @property
    def ratio_random_features(self) -> float:
        """Ratio of random features property.

        :return: ratio of random features
        :rtype: float
        """
        return self._ratio_random_features

    @ratio_random_features.setter
    def ratio_random_features(self, value: float) -> None:
        """Ratio of random features setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 < value < 1.0:
            raise ValueError("ratio_random_features value must be in the range (0, 1).")
        self._ratio_random_features = value


class MD3SVM(MarginDensityBasedEstimator):
    """MD3-SVM (Margin Density Drift Detection SVM) algorithm class."""

    def __init__(
        self,
        config: MD3Config,
        metric_scorer: Callable,
        random_state: int,
        svm_args: Optional[Dict[str, Union[str, int, float]]] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: MD3Config
        :param metric_scorer: metric scorer function
        :type metric_scorer: Callable
        :param random_state: random state value
        :type random_state: int
        :param svm_args: LinearSVC arguments
        :type svm_args: Optional[Dict[str, Union[str, int, float]]]
        """
        self.svm_args = (
            {
                "C": 1.0,
                "penalty": "l2",
                "loss": "hinge",
            }
            if svm_args is None
            else svm_args
        )
        super().__init__(
            estimator=LinearSVC(**self.svm_args),
            config=config,
            metric_scorer=metric_scorer,
            random_state=random_state,
        )
        self.w = None
        self.b = None

    def fit(
        self,
        X: np.array,  # noqa: N803
        y: np.array,
        sample_weight: Optional[Union[List[int], List[float]]] = None,
    ):
        """Fit estimator.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :param sample_weight: assigns weights to each sample
        :type sample_weight: Optional[Union[List[int], List[float]]]
        :return: fitted estimator
        :rtype: self
        """
        self.sample_weight = sample_weight
        self.estimator.fit(X=X, y=y, sample_weight=self.sample_weight)
        self.w = self.estimator.coef_[0]
        self.b = self.estimator.intercept_[0]
        self._calculate_reference_distribution(X=X, y=y)
        return self

    @property
    def svm_args(self) -> Dict[str, Union[str, int, float]]:
        """SVM arguments property.

        :return: SVM arguments
        :rtype: Dict[str, Union[str, int, float]]
        """
        return self._svm_args

    @svm_args.setter
    def svm_args(self, value: Dict[str, Union[str, int, float]]) -> None:
        """SVM arguments setter.

        :param value: value to be set
        :type value: Dict[str, Union[str, int, float]]
        """
        self._svm_args = value

    def _calculate_margin_signal(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return (np.abs(np.dot(X, self.w) + self.b) <= 1).astype(int)


class MD3RS(MarginDensityBasedEstimator):
    """MD3-RS (Margin Density Drift Detection RS) algorithm class."""

    def __init__(
        self,
        estimator,
        config: MD3Config,
        metric_scorer: Callable,
        random_state: int,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: MD3Config
        :param metric_scorer: metric scorer function
        :type metric_scorer: Callable
        :param random_state: random state value
        :type random_state: int
        """
        super().__init__(
            estimator=BaggingClassifier(
                base_estimator=estimator,
                max_features=config.ratio_random_features,  # type: ignore
                n_estimators=config.num_estimators,  # type: ignore
                random_state=random_state,
            ),
            config=config,
            metric_scorer=metric_scorer,
            random_state=random_state,
        )

    def fit(
        self,
        X: np.array,  # noqa: N803
        y: np.array,
        sample_weight: Optional[Union[List[int], List[float]]] = None,
    ):
        """Fit estimator.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :param sample_weight: assigns weights to each sample
        :type sample_weight: Optional[Union[List[int], List[float]]]
        :return: fitted estimator
        :rtype: self
        """
        self.sample_weight = sample_weight
        self.estimator.fit(X=X, y=y, sample_weight=self.sample_weight)
        self._calculate_reference_distribution(X=X, y=y)
        return self

    def _calculate_margin_signal(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        y_prob_pred = self.estimator.predict_proba(X=X)
        return (
            np.abs(np.diff(y_prob_pred))
            <= self.config.margin_uncertainty  # type: ignore
        ).astype(int)
