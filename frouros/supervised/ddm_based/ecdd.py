"""ECDD (EWMA for Concept Drift Detection) module."""

from typing import Callable, Dict, List, Optional, Union

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore

from frouros.metrics.base import BaseMetric
from frouros.supervised.ddm_based.base import (
    DDMBasedEstimator,
    ECDDBaseConfig,
)
from frouros.utils.stats import EWMA, Mean


class ECDDWTConfig(ECDDBaseConfig):
    """ECDD-WT (EWMA for Concept Drift Detection with Warning) configuration class."""


class ECDDWT(DDMBasedEstimator):
    """ECDD-WT (EWMA for Concept Drift Detection with Warning) algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: ECDDBaseConfig,
        metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: ECDDBaseConfig
        :param metrics: performance metrics
        :type metrics: Optional[Union[BaseMetric, List[BaseMetric]]]
        """
        super().__init__(
            estimator=estimator,
            error_scorer=error_scorer,
            config=config,
            metrics=metrics,
        )
        self.p = Mean()
        self.z = EWMA(alpha=self.config.lambda_)  # type: ignore
        self._lambda_div_two_minus_lambda = self.config.lambda_ / (  # type: ignore
            2 - self.config.lambda_  # type: ignore
        )

    def _check_threshold(
        self, control_limit: float, z_variance: float, warning_level: float = 1.0
    ) -> bool:
        return self.z.mean > self.p.mean + warning_level * control_limit * z_variance

    def _get_specific_response_attributes(self):
        attributes = {"p_mean": self.p.mean, "z_mean": self.z.mean}
        return attributes

    def _reset(self, *args, **kwargs) -> None:
        super()._reset()
        self.p = Mean()
        self.z = EWMA(alpha=self.config.lambda_)  # type: ignore

    def update(
        self,
        y: np.ndarray,
        X: np.ndarray = None,  # noqa: N803
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :param X: feature data
        :type X: Optional[numpy.ndarray]
        :return: response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
        X, y_pred, metrics = self._prepare_update(y=y)  # noqa: N806

        if self._drift_insufficient_samples:
            self._insufficient_samples_case(X=X, y=y)
            if not self._check_drift_sufficient_samples:
                # Drift has been detected but there are no enough samples
                # to train a new model from scratch
                return self._insufficient_samples_response(metrics=metrics)
            # There are enough samples to train a new model from scratch
            self._complete_delayed_drift()

        error_rate = self.error_scorer(y_true=y, y_pred=y_pred)
        self.p.update(value=error_rate)
        self.z.update(value=error_rate)

        specific_attributes = self._get_specific_response_attributes()

        if self.num_instances >= self.config.min_num_instances:
            error_rate_variance = self.p.mean * (1 - self.p.mean)
            z_variance = np.sqrt(
                self._lambda_div_two_minus_lambda
                * (1 - self.z.one_minus_alpha ** (2 * self.num_instances))
                * error_rate_variance
            )
            control_limit = self.config.control_limit_func(  # type: ignore
                p=self.p.mean
            )

            drift_flag = self._check_threshold(
                control_limit=control_limit, z_variance=z_variance
            )

            if drift_flag:
                # Out-of-Control
                self._drift_case(X=X, y=y)
                self.drift = True
                self.warning = False
            else:
                warning_flag = self._check_threshold(
                    control_limit=control_limit,
                    z_variance=z_variance,
                    warning_level=self.config.warning_level,  # type: ignore
                )
                if warning_flag:
                    # Warning
                    self._warning_case(X=X, y=y)
                    self.warning = True
                else:
                    # In-Control
                    self._normal_case(X=X, y=y)
                    self.warning = False
                self.drift = False
        else:
            self._normal_case(X=X, y=y)
            self.drift, self.warning = False, False

        return self._update_response(
            specific_attributes=specific_attributes, metrics=metrics
        )
