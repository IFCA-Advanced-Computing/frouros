"""Supervised modes module."""

import abc
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # type: ignore
from sklearn.base import TransformerMixin  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.base import SupervisedBaseEstimator
from frouros.utils.decorators import check_func_parameters
from frouros.utils.validation import check_has_partial_fit


class BaseFit(abc.ABC):
    """Abstract class representing a fit type."""

    def __init__(self, obj: Union[Pipeline, SupervisedBaseEstimator]) -> None:
        """Init method.

        :param obj: Object that includes the detector
        :type obj: Union[sklearn.pipeline.Pipeline, SupervisedBaseEstimator]
        """
        self.detector = self._get_detector(obj=obj)

    @property
    def detector(self) -> SupervisedBaseEstimator:
        """Detector property.

        :return: detector object
        :rtype: SupervisedBaseEstimator
        """
        return self._detector

    @detector.setter
    def detector(self, value: SupervisedBaseEstimator) -> None:
        """Detector setter.

        :param value: value to be set
        :type value: SupervisedBaseEstimator
        :raises TypeError: Type error exception
        """
        if not isinstance(value, SupervisedBaseEstimator):
            raise TypeError("detector must be of type SupervisedBaseEstimator.")
        self._detector = value

    @abc.abstractmethod
    def _get_detector(
        self, obj: Union[Pipeline, SupervisedBaseEstimator]
    ) -> SupervisedBaseEstimator:
        pass

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # noqa: N803
        """Abstract fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """

    def update(self, value: Union[int, float]) -> None:
        """Update detector with a value.

        :param value: value that is passed to the detector
        :type value: Union[int, float]
        """
        self.detector.update(value=value)

    def status(self) -> Dict[str, bool]:
        """Get detector status.

        :return: detector status
        :rtype: Dict[str, bool]
        """
        return self.detector.status

    def reset(self) -> None:
        """Reset detector."""
        self.detector.reset()


class BasePartialFit(BaseFit):
    """Abstract class representing a partial fit type."""

    @abc.abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray, **kwargs):  # noqa: N803
        """Abstract partial fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """


class NormalFitEstimator(BaseFit):
    """Normal fit estimator class."""

    def _get_detector(self, obj: SupervisedBaseEstimator) -> SupervisedBaseEstimator:
        return obj

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # noqa: N803
        """Fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """
        self.detector.fit(X=X, y=y, **kwargs)


class NormalFitPipeline(NormalFitEstimator):
    """Normal fit pipeline class."""

    def __init__(self, obj: Pipeline) -> None:
        """Init method.

        :param obj: Object that includes the detector
        :type obj: sklearn.pipeline.Pipeline
        """
        super().__init__(obj=obj)
        self.pipeline = obj

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline property.

        :return: pipeline object
        :rtype: sklearn.pipeline.Pipeline
        """
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: Pipeline) -> None:
        """Pipeline setter.

        :param value: value to be set
        :type value: sklearn.pipeline.Pipeline
        :raises TypeError: Type error exception
        """
        if not isinstance(value, Pipeline):
            raise TypeError("pipeline must be of type sklearn.pipeline.Pipeline.")
        self._pipeline = value

    def _get_detector(self, obj: Pipeline) -> SupervisedBaseEstimator:
        return obj.steps[-1][-1]

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # noqa: N803
        """Fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """
        self.pipeline.fit(X=X, y=y, **kwargs)


class PartialFitEstimator(BasePartialFit):
    """Partial fit estimator class."""

    def _get_detector(self, obj: SupervisedBaseEstimator) -> SupervisedBaseEstimator:
        return obj

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # noqa: N803
        """Fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """
        self.detector.fit(X=X, y=y, **kwargs)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # noqa: N803
        """Partial fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """
        self.detector.partial_fit(X=X, y=y, **kwargs)


class PartialFitPipeline(PartialFitEstimator):
    """Partial fit pipeline class."""

    def __init__(self, obj: Pipeline) -> None:
        """Init method.

        :param obj: Object that includes the detector
        :type obj: sklearn.pipeline.Pipeline
        """
        super().__init__(obj=obj)
        self.pipeline = obj
        self.steps = self._get_steps()

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline property.

        :return: pipeline object
        :rtype: sklearn.pipeline.Pipeline
        """
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: Pipeline) -> None:
        """Pipeline setter.

        :param value: value to be set
        :type value: sklearn.pipeline.Pipeline
        :raises TypeError: Type error exception
        """
        if not isinstance(value, Pipeline):
            raise TypeError("value must be of type sklearn.pipeline.Pipeline.")
        self._pipeline = value

    @property
    def steps(self) -> List[Tuple[str, TransformerMixin]]:
        """Steps property.

        :return: list with the steps
        :rtype: List[Tuple[str, sklearn.base.TransformerMixin]]
        """
        return self._steps

    @steps.setter
    def steps(self, value: List[Tuple[str, TransformerMixin]]) -> None:
        """Steps setter.

        :param value: value to be set
        :type value: List[Tuple[str, sklearn.base.TransformerMixin]]
        """
        self._steps = value

    def _get_detector(self, obj: Pipeline) -> SupervisedBaseEstimator:
        return obj.steps[-1][-1]

    def _get_steps(self) -> List[Tuple[str, TransformerMixin]]:
        steps = self.pipeline.steps[:-1]
        for _, step in steps:
            check_has_partial_fit(estimator=step)
        return steps

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # noqa: N803
        """Fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """
        self.pipeline.fit(X=X, y=y, **kwargs)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:  # noqa: N803
        """Partial fit method.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        """
        for _, step in self.steps:
            step.partial_fit(X=X)
            X = step.transform(X)  # noqa: N806
        self.detector.partial_fit(X=X, y=y, **kwargs)


class BaseMode(abc.ABC):
    """Abstract class representing a mode type."""

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
        self.detector = detector  # type: ignore
        self.value_func = value_func  # type: ignore
        self.metrics = metrics  # type: ignore
        self.metrics_func: Callable = (
            (lambda y_true, y_pred: {})
            if not self.metrics
            else lambda y_true, y_pred: {
                metric.name: metric(y_true=y_true, y_pred=y_pred)
                for metric in self.metrics  # type: ignore
            }
        )

    @property
    def detector(self) -> BaseFit:
        """Detector property.

        :return: detector object
        :rtype BaseFit
        """
        return self._detector  # type: ignore # pylint: disable=no-member

    @detector.setter  # type: ignore
    @abc.abstractmethod
    def detector(self, value: Union[Pipeline, SupervisedBaseEstimator]) -> None:
        """Abstract detector setter method.

        :param value: value to be set
        :type value: Union[Pipeline, SupervisedBaseEstimator]
        """

    @property
    def metrics(self) -> List[BaseMetric]:
        """Metrics property.

        :return: performance metrics to use
        :rtype: List[BaseMetric]
        """
        return self._metrics

    @metrics.setter
    def metrics(self, value: Union[BaseMetric, List[BaseMetric]]) -> None:
        """Metrics setter.

        :param value: value to be set
        :type value: Union[BaseMetric, List[BaseMetric]]
        :raises TypeError: Type error exception
        """
        if isinstance(value, List):
            if not all(isinstance(e, BaseMetric) for e in value):
                raise TypeError(
                    "value must be of type BaseMetric or a list of BaseMetric."
                )
            self._metrics = value
        elif value is None:
            self._metrics = []
        elif not isinstance(value, BaseMetric):
            raise TypeError("value must be of type BaseMetric or a list of BaseMetric.")
        else:
            self._metrics = [value]

    @property
    def value_func(self) -> Callable:
        """Calculate value function property.

        :return: error scorer function
        :rtype: Callable
        """
        return self._value_func

    @value_func.setter  # type: ignore
    @check_func_parameters
    def value_func(self, value: Callable) -> None:
        """Calculate value function setter.

        :param value: value to be set
        :type value: Callable
        """
        self._value_func = value

    @abc.abstractmethod
    def _drift_case(self, **kwargs) -> None:
        pass

    def _reset_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset()

    @abc.abstractmethod
    def update(
        self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
    ) -> Dict[str, Union[bool, Dict[str, float]]]:
        """Abstract update method.

        :param y_true: value to be set
        :type y_true: Union[Pipeline, SupervisedBaseEstimator]
        :param y_pred: value to be set
        :type y_pred: Union[Pipeline, SupervisedBaseEstimator]
        :return: response dict
        :rtype: Dict[str, Union[bool, Dict[str, float]]]
        """


class NormalMode(BaseMode):
    """Normal mode class."""

    @BaseMode.detector.setter  # type: ignore
    def detector(self, value: Union[Pipeline, SupervisedBaseEstimator]) -> None:
        """Detector setter.

        :param value: value to be set
        :type value: Union[sklearn.pipeline.Pipeline, SupervisedBaseEstimator]
        :raises NotImplementedError: Not implemented error exception
        """
        self._detector: Union[NormalFitPipeline, NormalFitEstimator]
        if isinstance(value, Pipeline):
            detector = value.steps[-1][-1]
            if not isinstance(detector, SupervisedBaseEstimator):
                raise NotImplementedError("Only SupervisedBaseEstimator are supported.")
            self._detector = NormalFitPipeline(obj=value)
        else:
            if not isinstance(value, SupervisedBaseEstimator):
                raise NotImplementedError("Only SupervisedBaseEstimator are supported.")
            self._detector = NormalFitEstimator(obj=value)

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
        metrics = self.metrics_func(y_true=y_true, y_pred=y_pred)
        value = self.value_func(y_true=y_true, y_pred=y_pred)
        self.detector.update(value=value)
        status = self.detector.status()
        if status["drift"]:
            self._drift_case()
        return {**status, "metrics": {**metrics}}

    def _drift_case(self, **kwargs) -> None:
        self.detector.reset()
        if self.metrics:
            self._reset_metrics()
