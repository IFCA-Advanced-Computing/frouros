"""Supervised DDM based base module."""

import abc
from typing import (  # noqa: TYP001
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, is_classifier  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.base import (
    SupervisedBaseConfig,
    SupervisedBaseEstimatorReFit,
)
from frouros.utils.decorators import check_func_parameters
from frouros.utils.logger import logger


class DDMBaseConfig(SupervisedBaseConfig):
    """Class representing a DDM based configuration class."""


class DDMBasedEstimator(SupervisedBaseEstimatorReFit):
    """Abstract class representing a DDM based estimator."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: SupervisedBaseConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: estimator to be used
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: SupervisedBaseConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(
            estimator=estimator,
            config=config,
            metrics=metrics,
        )
        self.error_scorer = error_scorer  # type: ignore

    @property
    def error_scorer(self) -> Callable:
        """Error scorer property.

        :return: error scorer function
        :rtype: Callable
        """
        return self._error_scorer

    @error_scorer.setter  # type: ignore
    @check_func_parameters
    def error_scorer(self, value: Callable) -> None:
        """Error scorer setter.

        :param value: value to be set
        :type value: Callable
        """
        self._error_scorer = value

    def _drift_case(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        logger.warning("Changing threshold has been exceeded. Drift detected.")
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )
        X_new_context, y_new_context = self._list_to_arrays(  # noqa: N806
            list_=self._fit_method.new_context_samples
        )
        if not is_classifier(self.estimator):
            self._fit_estimator(X=X_new_context, y=y_new_context)
            self._reset()
        self._check_number_classes(
            X_new_context=X_new_context, y_new_context=y_new_context
        )

    def _normal_case(self, *args, **kwargs) -> None:  # noqa: N803
        X, y = kwargs.get("X"), kwargs.get("y")  # noqa: N806
        for _ in range(y.shape[0]):  # type: ignore
            self.ground_truth.popleft()
            self.predictions.popleft()
        self._fit_method.add_fit_context_samples(X=X, y=y)
        X, y = self._list_to_arrays(  # noqa: N806
            list_=self._fit_method.fit_context_samples
        )
        self._fit_estimator(X=X, y=y)
        # Remove warning samples if performance returns to normality
        self._fit_method.post_fit_estimator()

    def _prepare_update(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
        check_is_fitted(self.estimator)
        X, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]

        metrics = self._metrics_func(y_true=y, y_pred=y_pred)
        return X, y_pred, metrics

    def _reset(self, *args, **kwargs) -> None:
        self.num_instances = 0
        self._drift_insufficient_samples = False
        self.sample_weight = None

        map(
            lambda x: x.clear(),  # type: ignore
            [
                self.delayed_predictions,
                self.ground_truth,
                self.predictions,
            ],
        )
        self._fit_method.reset()

    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        logger.warning(
            "Warning threshold has been exceeded. "
            "New concept will be learned until drift is detected."
        )
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )

    @abc.abstractmethod
    def update(
        self, y: np.array
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
