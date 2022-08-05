"""Supervised DDM Based modes module."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.base import SupervisedBaseEstimator
from frouros.supervised.ddm_based.base import DDMBasedEstimator
from frouros.supervised.utils import list_to_arrays
from frouros.supervised.modes import BaseMode, PartialFitEstimator, PartialFitPipeline
from frouros.utils.validation import check_has_partial_fit


class IncrementalLearningMode(BaseMode):
    """Incremental learning mode class."""

    def __init__(
        self,
        detector: Union[Pipeline, SupervisedBaseEstimator],
        value_func: Callable,
        metrics: Optional[List[BaseMetric]] = None,
    ) -> None:
        """Init method.

        :param detector: detector to be used
        :type detector: Union[sklearn.pipeline.Pipeline, SupervisedBaseEstimator]
        :param value_func: function to compute the value passed to the detector
        :type value_func: Callable
        :param metrics: performance metrics to use during iterations
        :type metrics: metrics: Optional[List[BaseMetric]
        """
        super().__init__(detector=detector, value_func=value_func, metrics=metrics)
        self.warning_samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self._drift_insufficient_samples = False

    @BaseMode.detector.setter  # type: ignore
    def detector(self, value: Union[Pipeline, SupervisedBaseEstimator]) -> None:
        """Detector setter.

        :param value: value to be set
        :type value: Union[sklearn.pipeline.Pipeline, SupervisedBaseEstimator]
        :raises NotImplementedError: Not implemented error exception
        """
        self._detector: Union[PartialFitPipeline, PartialFitEstimator]
        if isinstance(value, Pipeline):
            detector = value.steps[-1][-1]
            check_has_partial_fit(estimator=detector.estimator)
            if not isinstance(detector, DDMBasedEstimator):
                raise NotImplementedError("Only DDMBasedEstimator are supported.")
            self._detector = PartialFitPipeline(obj=value)
        else:
            check_has_partial_fit(estimator=value.estimator)
            if not isinstance(value, DDMBasedEstimator):
                raise NotImplementedError("Only DDMBasedEstimator are supported.")
            self._detector = PartialFitEstimator(obj=value)

    def update(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> Dict[str, Union[bool, Dict[str, float]]]:
        """Update method.

        :param y_true: value to be set
        :type y_true: Union[Pipeline, SupervisedBaseEstimator]
        :param y_pred: value to be set
        :type y_pred: Union[Pipeline, SupervisedBaseEstimator]
        :return: response dict
        :rtype: Dict[str, Union[bool, Dict[str, float]]]
        """
        X = kwargs["X"]  # noqa: N806
        metrics = self.metrics_func(y_true=y_true, y_pred=y_pred)
        if not self._drift_insufficient_samples:
            value = self.value_func(y_true=y_true, y_pred=y_pred)
            self.detector.update(value=value)
        status = self.detector.status()
        if status["drift"]:
            self._drift_case(X=X, y=y_true)
        elif status["warning"]:
            self._warning_case(X=X, y=y_true)
        else:
            self._normal_case(X=X, y=y_true)

        return {**status, "metrics": {**metrics}}

    def _drift_case(self, **kwargs) -> None:
        X, y = kwargs["X"], kwargs["y"]  # noqa: N806
        self.warning_samples.append((X, y))
        X_new_context, y_new_context = list_to_arrays(  # noqa: N806
            list_=self.warning_samples
        )
        if len(np.unique(y_new_context)) > 1:  # Check number of classes > 1
            self.detector.fit(X=X_new_context, y=y_new_context)
            self.warning_samples.clear()
            self.detector.reset()
            self._drift_insufficient_samples = False
            if self.metrics:
                self._reset_metrics()
        else:
            self._drift_insufficient_samples = True

    def _normal_case(self, **kwargs) -> None:
        X, y = kwargs["X"], kwargs["y"]  # noqa: N806
        self.warning_samples.clear()
        self.detector.partial_fit(X=X, y=y)

    def _warning_case(self, **kwargs) -> None:
        X, y = kwargs["X"], kwargs["y"]  # noqa: N806
        self.warning_samples.append((X, y))
