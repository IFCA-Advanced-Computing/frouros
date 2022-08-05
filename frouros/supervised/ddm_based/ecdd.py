"""ECDD (EWMA for Concept Drift Detection) module."""

from typing import Union

import numpy as np  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore

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
        config: ECDDBaseConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param config: configuration parameters
        :type config: ECDDBaseConfig
        """
        super().__init__(
            estimator=estimator,
            config=config,
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

    def reset(self, *args, **kwargs) -> None:
        """Reset method."""
        super().reset()
        self.p = Mean()
        self.z = EWMA(alpha=self.config.lambda_)  # type: ignore

    def update(self, value: Union[int, float]) -> None:
        """Update drift detector.

        :param value: value to update detector
        :type value: Union[int, float]
        """
        self.num_instances += 1

        self.p.update(value=value)
        self.z.update(value=value)

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
                    self.warning = True
                else:
                    # In-Control
                    self.warning = False
                self.drift = False
        else:
            self.drift, self.warning = False, False
