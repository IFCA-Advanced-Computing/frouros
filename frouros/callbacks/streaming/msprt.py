"""mSPRT (Mixing Sequentially Probability Ratio Test) callback module."""

from typing import Union, Tuple, Optional

import numpy as np  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.utils.stats import CircularMean


class mSPRT(BaseCallbackStreaming):  # noqa: N801 # pylint: disable=invalid-name
    """mSPRT (mixing Sequentially Probability Ratio Test) callback class.

    :References:

    .. [johari2022always] Ramesh, Johari, et al.
        "Always valid inference: Continuous monitoring of a/b tests"
        Operations Research 70.3 (2022): 1806-1821.
    """

    def __init__(
        self,
        alpha: float,
        sigma: Union[int, float] = 1.0,
        tau: Union[int, float] = 1.0,
        lambda_: Union[int, float] = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """Init method.

        :param alpha: alpha value
        :type alpha: float
        :param sigma: sigma value
        :type sigma: Union[int, float]
        :param tau: tau value
        :type tau: Union[int, float]
        :param lambda_: lambda value
        :type lambda_: Union[int, float]
        :param name: name value
        :type name: Optional[str]
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.sigma = sigma
        self.sigma_squared = self.sigma**2
        self.two_sigma_squared = 2 * self.sigma_squared
        self.tau = tau
        self.tau_squared = self.tau**2
        self.lambda_ = lambda_
        self.mean = None
        self.theta = None
        self.p_value = 1.0

    @property
    def alpha(self) -> float:
        """Alpha property.

        :return: alpha value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Alpha setter.

        :param value: value to be set
        :type value: float
        """
        if not isinstance(value, float):
            raise TypeError("alpha must be a float")
        if not 0.0 < value < 1.0:
            raise ValueError("alpha must be in the range (0, 1)")
        self._alpha = value

    @property
    def sigma(self) -> Optional[Union[int, float]]:
        """Sigma property.

        :return: sigma value
        :rtype: Optional[Union[int, float]]
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value: Optional[Union[int, float]]) -> None:
        """Sigma setter.

        :param value: value to be set
        :type value: Optional[float]
        """
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("sigma must be int, float or None")
        self._sigma = value

    @property
    def tau(self) -> Optional[Union[int, float]]:
        """Tau property.

        :return: tau squared value
        :rtype: Optional[Union[int, float]]
        """
        return self._tau

    @tau.setter
    def tau(self, value: Union[int, float]) -> None:
        """Tau setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not isinstance(value, (int, float)):
            raise TypeError("tau must be int, float or None")
        self._tau = value

    @property
    def lambda_(self) -> Optional[Union[int, float]]:
        """Lambda property.

        :return: lambda value
        :rtype: Optional[Union[int, float]]
        """
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value: Union[int, float]) -> None:
        """Lambda setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not isinstance(value, (int, float)):
            if value <= 0.0:
                raise ValueError("lambda_ must be greater than 0")
        self._lambda_ = value

    def on_fit_end(self, **kwargs) -> None:
        """On fit end method."""
        self.mean = CircularMean(size=self.detector.window_size)  # type: ignore
        self.theta = self.detector.compare(X=kwargs["X"])[0].distance  # type: ignore

    def on_update_end(self, **kwargs) -> None:
        """On update end method."""
        self.mean.update(value=kwargs["value"])  # type: ignore
        self.p_value, likelihood = self._calculate_p_value()

        self.logs.update(
            {
                "distance_mean": self.mean.get(),  # type: ignore
                "likelihood": likelihood,
                "p_value": self.p_value,
            },
        )

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.mean = None
        self.p_value = 1.0

    def _calculate_p_value(self) -> Tuple[float, float]:
        likelihood = self._likelihood_normal_mixing_distribution(
            mean=self.mean.get(),  # type: ignore
            sigma=self.sigma,  # type: ignore
            sigma_squared=self.sigma_squared,
            tau_squared=self.tau_squared,
            two_sigma_squared=self.two_sigma_squared,
            n=self.detector.window_size,  # type: ignore
            theta=self.theta,  # type: ignore
            lambda_=self.lambda_,  # type: ignore
        )
        p_value = min(
            self.p_value,
            1 / likelihood,
        )
        return p_value, likelihood

    @staticmethod
    def _likelihood_normal_mixing_distribution(  # pylint: disable=too-many-arguments
        mean: float,
        sigma: float,
        sigma_squared: float,
        tau_squared: float,
        two_sigma_squared: float,
        n: int,
        theta: float,
        lambda_: float,
    ) -> float:
        # FIXME: Explore lambda_ influence   # pylint: disable=fixme
        #  and redesign the likelihood formula
        n_tau_squared = n * tau_squared
        sigma_squared_plus_n_tau_squared = sigma_squared + n_tau_squared
        likelihood = (sigma / np.sqrt(sigma_squared_plus_n_tau_squared)) * np.exp(
            n
            * n_tau_squared
            * lambda_  # Not present in mSPRT, added as a hyperparameter to control
            # the influence of the distance difference
            * (mean - theta)
            ** 2  # (mean-theta) ** 2, theta=detector statistic (H_0 value, no distance)
            / (two_sigma_squared * sigma_squared_plus_n_tau_squared)
        )
        return likelihood
