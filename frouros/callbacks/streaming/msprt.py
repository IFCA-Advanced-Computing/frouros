"""mSPRT (Mixing Sequentially Probability Ratio Test) callback module."""

from typing import Union, Tuple, Optional

import numpy as np  # type: ignore
from scipy.stats import norm  # type: ignore

from frouros.callbacks.streaming.base import StreamingCallback
from frouros.utils.stats import Mean


class mSPRT(StreamingCallback):  # noqa: N801 # pylint: disable=invalid-name
    """mSPRT (mixing Sequentially Probability Ratio Test) callback class.

    :References:

    .. [johari2022always] Ramesh, Johari, et al.
        "Always valid inference: Continuous monitoring of a/b tests"
        Operations Research 70.3 (2022): 1806-1821.
    """

    def __init__(
        self,
        alpha: float,
        sigma: float = 1.0,
        tau: Optional[float] = None,
        truncation: int = 1,
        name: Optional[str] = None,
    ) -> None:
        """Init method.

        :param alpha: alpha value
        :type alpha: float
        :param sigma: sigma value
        :type sigma: float
        :param name: name value
        :type name: Optional[str]
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.sigma = sigma
        self.truncation = truncation
        self.sigma_squared = self.sigma**2
        self.two_sigma_squared = 2 * self.sigma_squared
        self.tau = tau
        self.tau_squared = (
            self.tau**2
            if self.tau is not None
            else self._calculate_tau_squared(
                alpha=self.alpha,
                sigma_squared=self.sigma_squared,
                truncation=self.truncation,
            )
        )
        self.incremental_mean = Mean()
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
    def tau(self) -> Optional[float]:
        """Tau property.

        :return: tau squared value
        :rtype: Optional[float]
        """
        return self._tau

    @tau.setter
    def tau(self, value: Optional[float]) -> None:
        """Tau setter.

        :param value: value to be set
        :type value: Optional[float]
        """
        if value is not None and not isinstance(value, float):
            raise TypeError("tau must be a float or None")
        self._tau = value

    def on_fit_end(self, **kwargs) -> None:
        """On fit end method."""
        self.incremental_mean.num_values = len(kwargs["X"])

    def on_update_end(self, value: Union[int, float], **kwargs) -> None:
        """On update end method.

        :param value: value to update detector
        :type value: int
        """
        self.incremental_mean.update(value=value)
        self.p_value, likelihood = self._calculate_p_value(value=value)

        self.logs.update(
            {
                "distance_mean": self.incremental_mean.get(),
                "likelihood": likelihood,
                "p_value": self.p_value,
            },
        )

    def reset(self) -> None:
        """Reset method."""
        self.incremental_mean = Mean()
        self.p_value = 1.0

    @staticmethod
    def _calculate_tau_squared(
        alpha: float,
        sigma_squared: float,
        truncation: int,
    ) -> float:
        b = 2 * np.log(1 / alpha) / ((truncation * sigma_squared) ** 0.5)
        minus_b_cdf = norm.cdf(-b)
        tau_squared = sigma_squared * minus_b_cdf / (1 / b * norm.pdf(b) - minus_b_cdf)
        return tau_squared

    def _calculate_p_value(self, value: float) -> Tuple[float, float]:
        likelihood = self._likelihood_normal_mixing_distribution(
            mean=self.incremental_mean.get(),
            sigma=self.sigma,
            sigma_squared=self.sigma_squared,
            tau_squared=self.tau_squared,
            two_sigma_squared=self.two_sigma_squared,
            value=value,
            n=self.detector.num_instances,  # type: ignore
        )
        p_value = min(
            self.p_value,
            1 / likelihood,
        )
        return p_value, likelihood

    @staticmethod
    def _likelihood_normal_mixing_distribution(
        mean: float,
        sigma: float,
        sigma_squared: float,
        tau_squared: float,
        two_sigma_squared: float,
        value: float,
        n: int,
    ) -> float:
        n_tau_squared = n * tau_squared
        sigma_squared_plus_n_tau_squared = sigma_squared + n_tau_squared
        likelihood = (sigma / np.sqrt(sigma_squared_plus_n_tau_squared)) * np.exp(
            n
            * n_tau_squared
            * (mean - value) ** 2
            / (two_sigma_squared * (sigma_squared_plus_n_tau_squared))
        )
        return likelihood
