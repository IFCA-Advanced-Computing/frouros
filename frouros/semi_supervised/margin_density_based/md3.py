"""MD3 (Margin Density Drift Detection) module."""

from typing import Callable, Dict, Optional, List, Tuple, Union

import numpy as np  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.semi_supervised.margin_density_based.base import (
    MarginDensityBasedEstimator,
    MarginDensityBasedConfig,
)
from frouros.utils.decorators import check_func_parameters
from frouros.utils.logger import logger


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

    @property
    def chunk_size(self) -> int:
        """Chunk size property.

        :return: chunk size
        :rtype: int
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        """Chunk size setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("chunk_size value must be greater than 0.")
        self._chunk_size = value

    @property
    def forgetting_factor(self) -> float:
        """Forgetting factor property.

        :return: forgetting factor
        :rtype: float
        """
        return self._forgetting_factor

    @forgetting_factor.setter
    def forgetting_factor(self, value: float) -> None:
        """Forgetting factor setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0.0:
            raise ValueError(
                "forgetting_factor value must be greater or equal than 0.0."
            )
        self._forgetting_factor = value

    @property
    def num_folds(self) -> int:
        """Number of folds property.

        :return: number of folds
        :rtype: int
        """
        return self._num_folds

    @num_folds.setter
    def num_folds(self, value: int) -> None:
        """Number of folds setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 2:
            raise ValueError("num_folds value must be greater than 1.")
        self._num_folds = value

    @property
    def num_training_samples(self) -> int:
        """Number of training samples property.

        :return: number of training samples
        :rtype: int
        """
        return self._num_training_samples

    @num_training_samples.setter
    def num_training_samples(self, value: int) -> None:
        """Number of training samples setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_training_samples value must be greater than 0.")
        self._num_training_samples = value

    @property
    def sensitivity(self) -> float:
        """Sensitivity property.

        :return: sensitivity
        :rtype: float
        """
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        """Sensitivity setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0.0:
            raise ValueError("sensitivity value must be greater or equal than 0.0.")
        self._sensitivity = value

    @property
    def signal_factor(self) -> float:
        """Signal factor property.

        :return: signal factor
        :rtype: float
        """
        return self._signal_factor

    @signal_factor.setter
    def signal_factor(self, value: float) -> None:
        """Signal factor setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0.0:
            raise ValueError("signal_factor value must be greater or equal than 0.0.")
        self._signal_factor = value


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
        super().__init__(estimator=LinearSVC(**self.svm_args), config=config)
        self.metric_scorer = metric_scorer  # type: ignore
        self.md_ref = None
        self.md_current = None
        self.sigma_ref = None
        self.metric_ref = None
        self.sigma_metric = None
        self.drift_suspected = False
        self.threshold_suspected = None
        self.threshold_confirmed = None
        self.w = None
        self.b = None
        self.random_state = random_state
        self.metrics = None
        self.X_samples, self.y_samples = [], []  # type: ignore

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
        :raises TrainingEstimatorError: Training estimator exception
        :return fitted estimator
        :rtype: self
        """
        self.sample_weight = sample_weight
        self.estimator.fit(X=X, y=y, sample_weight=self.sample_weight)
        self.w = self.estimator.coef_[0]
        self.b = self.estimator.intercept_[0]
        self._calculate_reference_distribution(X=X, y=y)
        return self

    @property
    def metric_scorer(self) -> Callable:
        """Metric scorer property.

        :return: metric scorer
        :rtype: Callable
        """
        return self._metric_scorer

    @metric_scorer.setter  # type: ignore
    @check_func_parameters
    def metric_scorer(self, value: Callable) -> None:
        """Metric scorer setter.

        :param value: value to be set
        :type value: Callable
        """
        self._metric_scorer = value

    @property
    def random_state(self) -> int:
        """Random state property.

        :return: random state
        :rtype: int
        """
        return self._random_state

    @random_state.setter
    def random_state(self, value: int) -> None:
        """Random state setter.

        :param value: value to be set
        :type value: int
        """
        self._random_state = value

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

    def _calculate_reference_distribution(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> None:
        folds = KFold(
            n_splits=self.config.num_folds,  # type: ignore
            shuffle=True,
            random_state=self.random_state,
        )
        fold_metrics, fold_md_refs, fold_weights = [], [], []
        for train_idx, test_idx in folds.split(X=X):
            X_fold_train, y_fold_train, X_fold_test, y_fold_test = (  # noqa: N806
                X[train_idx],
                y[train_idx],
                X[test_idx],
                y[test_idx],
            )
            fold_metric, fold_md_ref = self._fold_iteration(
                X_train=X_fold_train,
                y_train=y_fold_train,
                X_test=X_fold_test,
                y_test=y_fold_test,
            )
            fold_metrics.append(fold_metric)
            fold_md_refs.append(fold_md_ref)
            fold_weights.append(len(test_idx) / X.shape[0])
        self.metric_ref = np.average(a=fold_metrics, weights=fold_weights)
        self.sigma_metric = np.sqrt(
            np.average(
                a=(fold_metrics - self.metric_ref) ** 2,  # type: ignore
                weights=fold_weights,
            )
        )
        self.threshold_confirmed = (
            self.config.sensitivity * self.sigma_metric  # type: ignore
        )
        self.md_ref = np.average(a=fold_md_refs, weights=fold_weights)
        self.sigma_ref = np.sqrt(
            np.average(
                a=(fold_md_refs - self.md_ref) ** 2,  # type: ignore
                weights=fold_weights,
            )
        )
        self.threshold_suspected = (
            self.config.sensitivity * self.sigma_ref  # type: ignore
        )
        self.md_current = self.md_ref

    def _fold_iteration(
        self,
        X_train: np.ndarray,  # noqa: N803
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[float, float]:
        estimator = LinearSVC(**self.svm_args).fit(X=X_train, y=y_train)
        y_pred = estimator.predict(X=X_test)
        metric = self.metric_scorer(y_true=y_test, y_pred=y_pred)
        md_ref = self._calculate_md_ref(X=X_test)
        return metric, md_ref

    def predict(
        self, X: np.ndarray  # noqa: N803
    ) -> Tuple[np.ndarray, Dict[str, Union[bool, float]]]:
        """Predict values.

        :param X: input data
        :type X: numpy.ndarray
        :return predicted values and response dict
        :rtype: Tuple[np.ndarray, Dict[str, Union[bool, float]]]
        """
        check_is_fitted(self.estimator)
        X = check_array(X)  # noqa: N806
        y_pred = self.estimator.predict(X=X)
        drift_confirmed = False
        if self.drift_suspected:
            if len(self.X_samples) == self.config.num_training_samples:  # type: ignore
                X_samples, y_samples = [  # type: ignore # noqa: N806
                    *map(np.array, [self.X_samples, self.y_samples])
                ]
                y_samples_pred = self.estimator.predict(X=X_samples)
                metric_performance = self.metric_scorer(
                    y_true=y_samples, y_pred=y_samples_pred
                )
                if self.metric_ref - metric_performance > self.threshold_confirmed:
                    # Out-of-control
                    logger.warning(
                        "Drift confirmed threshold has been exceeded. "
                        "Model will be retrained."
                    )
                    self.estimator.fit(
                        X=X_samples, y=y_samples, sample_weight=self.sample_weight
                    )
                    drift_confirmed = True
                # Update statistics
                self._calculate_reference_distribution(X=X_samples, y=y_samples)
                self.drift_suspected = False
                self.X_samples, self.y_samples = [], []
        else:
            margin_signal = self._calculate_margin_signal(X=X).item()
            self.md_current = (
                self.config.forgetting_factor * self.md_current  # type: ignore
                + self.config.signal_factor * margin_signal  # type: ignore
            )
            if (
                np.abs(self.md_current - self.md_ref)  # type: ignore
                > self.threshold_suspected
            ):
                logger.warning(
                    "Drift suspected threshold has been exceeded. "
                    "Label samples will need to be provided in order to continue."
                )
                self.drift_suspected = True
        response = self._get_predict_response(
            drift_suspected=self.drift_suspected,
            drift_confirmed=drift_confirmed,
            metric_ref=self.metric_ref,
            sigma_metric=self.sigma_metric,
            md_current=self.md_current,
            md_ref=self.md_ref,
            sigma_ref=self.sigma_ref,
        )
        return y_pred, response

    def _calculate_margin_signal(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return (np.abs(np.dot(X, self.w) + self.b) <= 1).astype(int)
