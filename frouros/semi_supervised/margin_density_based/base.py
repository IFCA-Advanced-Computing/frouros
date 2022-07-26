"""Semi-supervised margin density based base module."""

import abc
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.semi_supervised.base import (
    SemiSupervisedBaseEstimator,
    SemiSupervisedBaseConfig,
)
from frouros.utils.decorators import check_func_parameters
from frouros.utils.logger import logger


class MarginDensityBasedConfig(SemiSupervisedBaseConfig):
    """Class representing a margin density based configuration class."""


class MarginDensityBasedEstimator(SemiSupervisedBaseEstimator):
    """Abstract class representing a margin density based estimator."""

    def __init__(
        self,
        estimator,
        config: MarginDensityBasedConfig,
        metric_scorer: Callable,
        random_state: Optional[int] = None,
    ) -> None:
        """Init method.

        :param estimator:
        :type estimator:
        :param config: configuration parameters
        :type config: MarginDensityBasedConfig
        :param metric_scorer: metric scorer function
        :type metric_scorer: Callable
        :param random_state: random state value
        :type random_state: Optional[int]
        """
        super().__init__(estimator=estimator, config=config)
        self.metric_scorer = metric_scorer  # type: ignore
        self.md_ref = None
        self.md_current = None
        self.sigma_ref = None
        self.metric_ref = None
        self.sigma_metric = None
        self.drift_suspected = False
        self.threshold_suspected = None
        self.threshold_confirmed = None
        self.X_samples, self.y_samples = [], []  # type: ignore
        self.random_state = random_state  # type: ignore

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

    def update(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Update drift detector.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: input data
        :type y: numpy.ndarray
        """
        self.X_samples.extend(X)
        self.y_samples.extend(y)

    def _calculate_md_ref(self, X: np.ndarray) -> float:  # noqa: N803
        return np.sum(self._calculate_margin_signal(X)) / X.shape[0]

    @staticmethod
    def _get_predict_response(
        drift_suspected: bool,
        drift_confirmed: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        response = {
            "drift_suspected": drift_suspected,
            "drift_confirmed": drift_confirmed,
        }
        response.update(**kwargs)  # type: ignore
        return response

    @abc.abstractmethod
    def _calculate_margin_signal(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        pass

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
        estimator = clone(estimator=self.estimator).fit(X=X_train, y=y_train)
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
        :return: predicted values and response dict
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
