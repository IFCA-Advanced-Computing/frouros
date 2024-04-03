"""BOCD (Bayesian Online Change Detection) module."""

import abc
import copy
from typing import Any, Optional, Union

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.change_detection.base import (
    BaseChangeDetection,
    BaseChangeDetectionConfig,
)


class BaseBOCDModel(abc.ABC):
    """Abstract class representing a BOCD model."""

    @abc.abstractmethod
    def log_pred_prob(self, idx: int, value: Union[int, float]) -> float:
        """Log predictive probability.

        :param idx: index
        :type idx: int
        :param value: value
        :type value: Union[int, float]
        :return: log predictive probability
        :rtype: float
        """

    @abc.abstractmethod
    def update(self, value: Union[int, float], **kwargs: Any) -> None:
        """Update model.

        :param value: value
        :type value: Union[int, float]
        :param kwargs: additional arguments
        :type kwargs: Any
        """


class GaussianUnknownMean(BaseBOCDModel):
    """Gaussian unknown mean model.

    :Note:
     Adapted from the implementation in https://github.com/gwgundersen/bocd.
    """

    def __init__(
        self,
        prior_mean: float = 0,
        prior_var: float = 1,
        data_var: float = 1,
    ) -> None:
        """Init method.

        :param prior_mean: prior mean
        :type prior_mean: float
        :param prior_var: prior variance
        :type prior_var: float
        :param data_var: data variance
        :type data_var: float
        """
        super().__init__()
        self.mean_params = np.array([prior_mean])
        self.precision_params = np.array([1 / prior_var])
        self.data_var = data_var

    @property
    def data_var(self) -> float:
        """Data variance property.

        :return: data variance
        :rtype: float
        """
        return self._data_var

    @data_var.setter
    def data_var(self, value: float) -> None:
        """Data variance setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0:
            raise ValueError("data_var must be greater than 0.")
        self._data_var = value

    def log_pred_prob(self, idx: int, value: Union[int, float]) -> float:
        """Log predictive probability.

        :param idx: index
        :type idx: int
        :param value: value
        :type value: Union[int, float]
        :return: log predictive probability
        :rtype: float
        """
        post_means = self.mean_params[:idx]
        post_stds = np.sqrt(self.var_params[:idx])
        return norm(post_means, post_stds).logpdf(value)

    def update(self, value: Union[int, float], **kwargs: Any) -> None:
        """Update model.

        :param value: value
        :type value: Union[int, float]
        :param kwargs: additional arguments
        :type kwargs: Any
        """
        new_param_prec = self.precision_params + (1 / self.data_var)
        self.precision_params = np.append([self.precision_params[0]], new_param_prec)
        new_param_mean = (
            self.mean_params * self.precision_params[:-1] + (value / self.data_var)
        ) / new_param_prec
        self.mean_params = np.append([self.mean_params[0]], new_param_mean)

    @property
    def var_params(self) -> np.ndarray:
        """Helper function for computing the posterior variance."""
        return 1 / self.precision_params + self.data_var


class BOCDConfig(BaseChangeDetectionConfig):
    """BOCD (Bayesian Online Change Detection) [adams2007bayesian]_ configuration.

    :param model: BOCD model, defaults to None. If None, :class:`frouros.detectors.concept_drift.streaming.change_detection.bocd.GaussianUnknownMean` is used.
    :type model: Optional[BaseBOCDModel]
    :param hazard: hazard value, defaults to 0.01
    :type hazard: float
    :param min_num_instances: minimum numbers of instances to start looking for changes, defaults to 30
    :type min_num_instances: int

    :References:

    .. [adams2007bayesian] Adams, Ryan Prescott, and David JC MacKay.
        "Bayesian online changepoint detection."
        arXiv preprint arXiv:0710.3742 (2007).
    """  # noqa: E501  pylint: disable=line-too-long

    model_type = GaussianUnknownMean

    def __init__(  # noqa: D107
        self,
        model: Optional[BaseBOCDModel] = None,
        hazard: float = 0.01,
        min_num_instances: int = 30,
    ) -> None:
        super().__init__(
            min_num_instances=min_num_instances,
        )
        self.model = model  # type: ignore
        self.log_hazard = np.log(hazard)
        self.log_1_minus_hazard = np.log(1 - hazard)

    @property
    def model(self) -> BaseBOCDModel:
        """Get model.

        :return: model
        :rtype: BaseBOCDModel
        """
        return self._model

    @model.setter
    def model(self, model: BaseBOCDModel) -> None:
        """Set model.

        :param model: model
        :type model: BaseBOCDModel
        :raises TypeError: if model is not an instance of BaseModel
        """
        if model is not None:
            if not isinstance(model, BaseBOCDModel):
                raise TypeError(
                    f"model must be an instance of BaseModel, not {type(model)}"
                )
            self._model = model
        else:
            self._model = self.model_type()


class BOCD(BaseChangeDetection):
    """BOCD (Bayesian Online Change Detection) [adams2007bayesian]_ detector.

    :param config: configuration object of the detector, defaults to None. If None, the default configuration of :class:`BOCDConfig` is used.
    :type config: Optional[BOCDConfig]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]]

    :Note:
     Adapted from the implementation in https://github.com/gwgundersen/bocd.

    :References:

    .. [adams2007bayesian] Adams, Ryan Prescott, and David JC MacKay.
        "Bayesian online changepoint detection."
        arXiv preprint arXiv:0710.3742 (2007).

    :Example:

    >>> from frouros.detectors.concept_drift import BOCD
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> dist_a = np.random.normal(loc=0.2, scale=0.01, size=1000)
    >>> dist_b = np.random.normal(loc=0.8, scale=0.04, size=1000)
    >>> stream = np.concatenate((dist_a, dist_b))
    >>> detector = BOCD()
    >>> for i, value in enumerate(stream):
    ...     _ = detector.update(value=value)
    ...     if detector.drift:
    ...         print(f"Change detected at step {i}")
    ...         break
    Change detected at step 1031
    """  # noqa: E501  # pylint: disable=line-too-long

    config_type = BOCDConfig

    def __init__(  # noqa: D107
        self,
        config: Optional[BOCDConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "log_r": np.array([[0.0]]),
            "predicted_mean": None,
            "predicted_var": None,
            "log_message": np.array([0.0]),
        }
        self._set_additional_vars_callback()
        self._model = copy.deepcopy(self.config.model)  # type: ignore

    @property
    def log_r(self) -> np.ndarray:
        """Log r getter.

        :return: log r
        :rtype: numpy.ndarray
        """
        return self._additional_vars["log_r"]

    @log_r.setter
    def log_r(self, value: np.ndarray) -> None:
        """Log r setter.

        :param value: value to be set
        :type value: numpy.ndarray
        """
        self._additional_vars["log_r"] = value

    @property
    def predicted_mean(self) -> Optional[float]:
        """Predicted mean getter.

        :return: predicted mean
        :rtype: Optional[float]
        """
        return self._additional_vars["predicted_mean"]

    @predicted_mean.setter
    def predicted_mean(self, value: Optional[float]) -> None:
        """Predicted mean setter.

        :param value: value to be set
        :type value: Optional[float]
        """
        self._additional_vars["predicted_mean"] = value

    @property
    def predicted_var(self) -> Optional[float]:
        """Predicted variance getter.

        :return: predicted variance
        :rtype: Optional[float]
        """
        return self._additional_vars["predicted_var"]

    @predicted_var.setter
    def predicted_var(self, value: Optional[float]) -> None:
        """Predicted variance setter.

        :param value: value to be set
        :type value: Optional[float]
        """
        self._additional_vars["predicted_var"] = value

    @property
    def log_message(self) -> np.ndarray:
        """Log message getter.

        :return: log message
        :rtype: numpy.ndarray
        """
        return self._additional_vars["log_message"]

    @log_message.setter
    def log_message(self, value: np.ndarray) -> None:
        """Log message setter.

        :param value: value to be set
        :type value: numpy.ndarray
        """
        self._additional_vars["log_message"] = value

    def _update(self, value: Union[int, float], **kwargs: Any) -> None:
        self.num_instances += 1
        current_idx = self.num_instances - 1

        # 3. Evaluate predictive probabilities.
        log_pis = self._model.log_pred_prob(
            idx=self.num_instances,
            value=value,
        )

        # 4. Calculate growth probabilities.
        log_pis_plus_message = log_pis + self.log_message
        log_growth_probs = (
            log_pis_plus_message + self.config.log_1_minus_hazard  # type: ignore
        )

        # 5. Calculate changepoint probability.
        log_cp_prob = logsumexp(
            log_pis_plus_message + self.config.log_hazard  # type: ignore
        )

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        # Dynamically expand log_r
        self.log_r = np.concatenate(
            (
                np.pad(
                    array=self.log_r,
                    pad_width=((0, 0), (0, 1)),
                    constant_values=-np.inf,
                ),
                np.expand_dims(new_log_joint - logsumexp(new_log_joint), axis=0),
            ),
            axis=0,
        )

        # 8. Update sufficient statistics.
        self._model.update(
            value=value,
        )

        # Update message.
        self.log_message = new_log_joint

        # 9. Perform prediction.
        probs = np.exp(self.log_r[current_idx, : self.num_instances])
        self.predicted_mean = np.sum(
            probs * self._model.mean_params[: self.num_instances]
        )
        self.predicted_var = np.sum(
            probs * self._model.var_params[: self.num_instances]
        )

        if self.num_instances >= self.config.min_num_instances:
            self.drift = (
                self.log_r[self.num_instances, : self.num_instances + 1].argmax()
                != self.num_instances
            )

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.log_r = np.array([[0.0]])
        self.log_message = np.array([0.0])
        self.predicted_mean = None
        self.predicted_var = None
        self._model = copy.deepcopy(self.config.model)  # type: ignore
