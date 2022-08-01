"""Supervised base module."""

import abc
from collections import deque
import copy
from typing import (  # noqa: TYP001
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, clone  # type: ignore
from sklearn.utils.estimator_checks import check_estimator  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.exceptions import (
    NoFitMethodError,
    TrainingEstimatorError,
)
from frouros.utils.logger import logger
from frouros.utils.validation import check_is_one_sample


class BaseFit(abc.ABC):
    """Abstract class representing a fit method."""

    def __init__(self, fit_method: Any) -> None:
        """Init method.

        :param fit_method: fit method to be used
        :type fit_method: Callable
        """
        self.fit_method = fit_method
        self.new_context_samples: List[Tuple[List[float], Union[str, int, float]]] = []

    def __call__(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        sample_weight: Optional[Union[List[int], List[float]]] = None,
        **kwargs,
    ) -> None:
        """__call__ method that invokes the fit method.

        :param X: input data
        :type X: numpy.ndarray
        :param y_true ground truth values
        :type y_true: numpy.ndarray
        """
        self.fit_method(X=X, y=y, sample_weight=sample_weight, **kwargs)

    @property
    def fit_method(self) -> Any:
        """Fit method property.

        :return: fit method
        :rtype: Any
        """
        return self._fit_method

    @fit_method.setter
    def fit_method(self, value: Any) -> None:
        """Fit method setter.

        :param value: value to be set
        :type value: Any
        """
        self._fit_method = value

    @property
    def new_context_samples(self) -> List[Tuple[List[float], Union[str, int, float]]]:
        """New context samples property.

        :return: new context samples
        :rtype: List[Tuple[List[float], Union[str, int, float]]]
        """
        return self._new_context_samples

    @new_context_samples.setter
    def new_context_samples(
        self, value: List[Tuple[List[float], Union[str, int, float]]]
    ) -> None:
        """New context samples setter.

        :param value: value to be set
        :type value: List[Tuple[List[float], Union[str, int, float]]]
        :raises TypeError: Type error exception
        """
        if not isinstance(value, List):
            raise TypeError("value must be of type List.")
        self._new_context_samples = value

    @property
    def fit_context_samples(
        self,
    ) -> List[Tuple[List[float], Union[str, int, float]]]:
        """Fit context samples property.

        :return: fit context samples
        :rtype: List[Tuple[List[float], Union[str, int, float]]]
        """
        return self._fit_context_samples

    @fit_context_samples.setter
    def fit_context_samples(
        self, value: List[Tuple[List[float], Union[str, int, float]]]
    ) -> None:
        """Fit context samples setter.

        :param value: value to be set
        :type value: List[Tuple[List[float], Union[str, int, float]]]
        :raises TypeError: Type error exception
        """
        if not isinstance(value, List):
            raise TypeError("value must be of type List.")
        self._fit_context_samples = value

    def add_fit_context_samples(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> None:
        """Add samples to be used by the next fit method invocation.

        :param X: input data
        :type X: numpy.ndarray
        :param y: ground truth values
        :type y: numpy.ndarray
        """
        self.fit_context_samples.extend([*zip(X.tolist(), y.tolist())])

    def reset(self) -> None:
        """Reset variables and samples lists."""
        self.new_context_samples.clear()

    def pre_fit_initial_estimator(
        self, X: np.ndarray, y: np.ndarray  # noqa: N803
    ) -> None:
        """Method to be executed before the initial fit method."""

    def post_fit_estimator(self) -> None:
        """Method to be executed after the fit method."""
        self.new_context_samples.clear()

    @staticmethod
    def _get_number_classes(y: np.array) -> int:
        return len(np.unique(y))

    @staticmethod
    def list_to_arrays(
        list_: List[Tuple[np.array, Union[str, int, float]]]
    ) -> List[np.ndarray]:
        """Convert list to numpy arrays.

        :param list_: list of samples
        :type list_: List[Tuple[np.array, Union[str, int, float]]]
        :return: list of numpy arrays
        :rtype List[numpy.ndarray]
        """
        return [*map(np.array, zip(*list_))]

    @abc.abstractmethod
    def is_ready(self, y: np.ndarray, **kwargs) -> bool:
        """Abstract method to check if fit method is ready.

        :param y: input data
        :type y: numpy.ndarray
        :return ready flag
        """


class NormalFit(BaseFit):
    """Normal fit method class."""

    def __init__(self, fit_method: Callable) -> None:
        """Init method.

        :param fit_method: fit method to be used
        :type fit_method: Callable
        """
        super().__init__(fit_method=fit_method)
        self.fit_context_samples: List[Tuple[List[float], Union[str, int, float]]] = []

    def __call__(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        sample_weight: Optional[Union[List[int], List[float]]] = None,
        **kwargs,
    ) -> None:
        """__call__ method that invokes the fit method.

        :param X: input data
        :type X: numpy.ndarray
        :param y: ground truth values
        :type y: numpy.ndarray
        :param sample_weight: assigns weights to each sample
        :type sample_weight: Optional[Union[List[int], List[float]]]
        """
        super().__call__(X=X, y=y, sample_weight=sample_weight, **kwargs)

    def reset(self) -> None:
        """Reset variables and samples lists."""
        self.fit_context_samples = copy.deepcopy(self.new_context_samples)
        super().reset()

    def pre_fit_initial_estimator(
        self, X: np.ndarray, y: np.ndarray  # noqa: N803
    ) -> None:
        """Method to be executed before the initial fit method."""
        self.add_fit_context_samples(X=X, y=y)

    def is_ready(self, y: np.ndarray, **kwargs) -> bool:
        """Check if fit method is ready to be used.

        :param y: input data
        :type y: numpy.ndarray
        :return ready flag
        """
        return self._get_number_classes(y=y) > 1


class PartialFit(BaseFit):
    """Partial fit method class."""

    def __init__(self, fit_method: Callable) -> None:
        """Init method.

        :param fit_method: fit method to be used
        :type fit_method: Callable
        """
        super().__init__(fit_method=fit_method)
        self.fit_context_samples: List[Tuple[List[float], Union[str, int, float]]] = []
        self.classes: List[Union[str, int, float, bool]] = []

    @property
    def classes(self) -> List[Union[str, int, float, bool]]:
        """Unique classes property.

        :return: unique classes list
        :rtype: List[Union[str, int, float, bool]]
        """
        return self._classes

    @classes.setter
    def classes(self, value: List[Union[str, int, float, bool]]) -> None:
        """Unique classes setter.

        :param value: value to be set
        :type value: List[Union[str, int, float, bool]]
        """
        self._classes = value

    def __call__(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        sample_weight: Optional[Union[List[int], List[float]]] = None,
        **kwargs,
    ) -> None:
        """__call__ method that invokes the fit method.

        :param X: input data
        :type X: numpy.ndarray
        :param y: ground truth values
        :type y: numpy.ndarray
        :param sample_weight: assigns weights to each sample
        :type sample_weight: Optional[Union[List[int], List[float]]]
        """
        self.classes = [*set(self.classes + np.unique(y).tolist())]
        super().__call__(X=X, y=y, sample_weight=sample_weight, classes=self.classes)

    def post_fit_estimator(self) -> None:
        """Method to be executed after the fit method."""
        super().post_fit_estimator()
        self.fit_context_samples.clear()

    def is_ready(self, y: np.ndarray, **kwargs) -> bool:
        """Check if fit method is ready to be used.

        :param y: input data
        :type y: numpy.ndarray
        :return ready flag
        """
        return True


class SupervisedBaseConfig(abc.ABC):
    """Abstract class representing a supervised configuration class."""

    def __init__(
        self,
        min_num_instances: int,
    ) -> None:
        """Init method.

        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        self.min_num_instances = min_num_instances

    @property
    def min_num_instances(self) -> int:
        """Minimum number of instances property.

        :return: minimum number of instances to start looking for changes
        :rtype: int
        """
        return self._min_num_instances

    @min_num_instances.setter
    def min_num_instances(self, value: int) -> None:
        """Minimum number of instances setter.

        :param value: value to be set
        :type value: int
        """
        self._min_num_instances = value


class SupervisedBaseEstimator(abc.ABC):
    """Abstract class representing a delayed target."""

    def __init__(
        self,
        estimator: BaseEstimator,
        config: SupervisedBaseConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: estimator to be used
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: SupervisedBaseConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        self.estimator = estimator
        self.config = config
        self.metrics: Optional[List[BaseMetric]] = metrics  # type: ignore
        self.delayed_predictions: Deque["Tuple[np.ndarray, np.ndarray]"] = deque()
        self.num_instances = 0
        self.sample_weight: Optional[Union[List[int], List[float]]] = None
        self._drift_insufficient_samples = False
        self._metrics_func: Callable = (
            (lambda y_true, y_pred: None)
            if self.metrics is None
            else lambda y_true, y_pred: {
                metric.name: metric(y_true=y_true, y_pred=y_pred)
                for metric in self.metrics  # type: ignore
            }
        )

    @property
    def config(self) -> SupervisedBaseConfig:
        """Config property.

        :return: configuration parameters of the estimator
        :rtype: SupervisedBaseConfig
        """
        return self._config

    @config.setter
    def config(self, value: SupervisedBaseConfig) -> None:
        """Config setter.

        :param value: value to be set
        :type value: SupervisedBaseConfig
        :raises TypeError: Type error exception
        """
        if not isinstance(value, SupervisedBaseConfig):
            raise TypeError("value must be of type SupervisedBaseConfig.")
        self._config = value

    @property
    def delayed_predictions(self) -> Deque["Tuple[np.ndarray, np.ndarray]"]:
        """Delayed predictions' property.

        :return: delayed predictions' deque
        :rtype: Deque["Tuple[np.ndarray, np.ndarray]"]
        """
        return self._delayed_predictions

    @delayed_predictions.setter
    def delayed_predictions(
        self, value: Deque["Tuple[np.ndarray, np.ndarray]"]
    ) -> None:
        """Delayed predictions' setter.

        :param value: value to be set
        :type value: Deque["Tuple[np.ndarray, np.ndarray]"]
        """
        self._delayed_predictions = value

    @property
    def estimator(self) -> BaseEstimator:
        """Estimator property.

        :return: estimator to use
        :rtype: BaseEstimator
        """
        return self._estimator

    @estimator.setter
    def estimator(self, value: BaseEstimator) -> None:
        """Estimator setter.

        :param value: value to be set
        :type value: BaseEstimator
        """
        check_estimator(value)
        self._estimator = value
        self._fit_method = self._get_fit_method()

    @property
    def metrics(self) -> Optional[List[BaseMetric]]:
        """Metrics property.

        :return: performance metrics to use
        :rtype: Optional[List[BaseMetric]]
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
            self._metrics = value
        elif not isinstance(value, BaseMetric):
            raise TypeError("value must be of type BaseMetric or a list of BaseMetric.")
        else:
            self._metrics = [value]

    @property
    def num_instances(self) -> int:
        """Number of instances counter property.

        :return: Number of instances counter value
        :rtype: int
        """
        return self._num_instances

    @num_instances.setter
    def num_instances(self, value: int) -> None:
        """Number of instances counter setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_instances must be greater or equal than 0.")
        self._num_instances = value

    @property
    def sample_weight(self) -> Optional[Union[List[int], List[float]]]:
        """Sample weight property.

        :return: Sample weight value
        :rtype: Optional[Union[List[int], List[float]]]
        """
        return self._sample_weight

    @sample_weight.setter
    def sample_weight(self, value: Optional[Union[List[int], List[float]]]) -> None:
        """Sample weight setter.

        :param value: value to be set
        :type value: Optional[Union[List[int], List[float]]]
        """
        self._sample_weight = value

    def _get_fit_method(self) -> BaseFit:
        partial_fit = getattr(self.estimator, "partial_fit", None)
        if not callable(partial_fit):
            logger.warning(
                "%s does not have partial_fit method. "
                "Therefore, with each new sample fit method will be used to train "
                "a model from scratch, increasing considerably the computational cost.",
                self.estimator,
            )
            fit = getattr(self.estimator, "fit", None)
            if not fit:
                raise NoFitMethodError(
                    f"{self.estimator} has not partial_fit or fit method."
                )
            return NormalFit(fit_method=fit)
        return PartialFit(fit_method=partial_fit)

    def _fit_extra(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        pass

    def _get_specific_response_attributes(self):
        pass

    @staticmethod
    def _get_update_response(
        drift: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        response = {
            "drift": drift,
        }
        response.update(**kwargs)  # type: ignore
        return response

    def _set_specific_attributes(
        self, specific_attributes: Dict[str, Tuple[Any, Any]]
    ) -> None:
        for attr_name, (_, attr_value) in specific_attributes.items():
            setattr(self, attr_name, attr_value)

    def _reset(self, *args, **kwargs) -> None:
        pass

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
        :return: fitted estimator
        :rtype: self
        """
        self.sample_weight = sample_weight
        self._fit_method.pre_fit_initial_estimator(X=X, y=y)
        try:
            self._fit_method(X=X, y=y, sample_weight=self.sample_weight)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: fit the estimator with more samples."
            ) from e
        self._fit_extra(X=X, y=y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict values.

        :param X: input data
        :type X: numpy.ndarray
        :return: predicted values
        :rtype: numpy.ndarray
        """
        check_is_fitted(self.estimator)
        X = check_array(X)  # noqa: N806
        check_is_one_sample(array=X)
        y_pred = self.estimator.predict(X=X)
        self.delayed_predictions.append((X, y_pred))
        return y_pred

    @abc.abstractmethod
    def update(
        self,
        y: np.ndarray,
        X: np.ndarray = None,  # noqa: N803
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update abstract method."""


class SupervisedBaseEstimatorReFit(SupervisedBaseEstimator):
    """Abstract class representing a re-fit estimator."""

    def _check_drift_sufficient_samples(self) -> bool:
        _, y_new_context = self._fit_method.list_to_arrays(
            list_=self._fit_method.new_context_samples
        )
        if self._fit_method.is_ready(y=y_new_context):
            return True
        return False

    def _check_number_classes(
        self, X_new_context: np.ndarray, y_new_context: np.ndarray  # noqa: N803
    ) -> None:
        if self._fit_method.is_ready(y=y_new_context):
            # Construct a new unfitted estimator with the same parameters
            self.estimator = clone(estimator=self.estimator)
            # Fit new estimator with the next context samples
            self._fit_estimator(X=X_new_context, y=y_new_context)
            self._reset()
        else:
            logger.warning(
                "Classifier estimator needs at least 2 different "
                "classes, but only 1 was found. Samples "
                "will be collected until this is to be fulfilled."
            )
            self._drift_insufficient_samples = True

    def _complete_delayed_drift(self) -> None:
        logger.warning(
            "Delayed drift has been completed because there was only one class."
        )
        self._drift_insufficient_samples = False
        self._reset()

    def _insufficient_samples_case(
        self, X: np.ndarray, y: np.ndarray  # noqa: N803
    ) -> None:
        self._add_context_samples(
            samples_list=self._fit_method.new_context_samples, X=X, y=y
        )

    def _insufficient_samples_response(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        response = self._get_update_response(
            drift=True,
            warning=False,
            **self._get_specific_response_attributes(),
            metrics=metrics,
        )
        return response

    @abc.abstractmethod
    def _drift_case(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _warning_case(self, X: np.array, y: np.array) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _normal_case(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def _reset(self, *args, **kwargs) -> None:
        pass

    def _fit_estimator(self, X: np.array, y: np.array) -> None:  # noqa: N803
        try:
            self._fit_method(X=X, y=y, sample_weight=self.sample_weight)
        except ValueError as e:
            raise TrainingEstimatorError(
                f"{e}\nHint: Increase min_num_instances value."
            ) from e

    def _fit_extra(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        pass

    @staticmethod
    def _add_context_samples(
        samples_list: List[Tuple[List[float], Union[str, int, float]]],
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> None:
        samples_list.extend([*zip(X.tolist(), y.tolist())])
