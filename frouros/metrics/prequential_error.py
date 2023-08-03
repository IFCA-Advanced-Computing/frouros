"""Prequential error using fading factor metric module."""

from typing import Optional, Union

from frouros.metrics.base import BaseMetric


class PrequentialError(BaseMetric):
    """Prequential error [dawid1984present]_ using fading factor [gama2009issues]_ metric.

    :param alpha: fading factor value, defaults to 1.0
    :type alpha: Union[int, float]
    :param name: name value, defaults to None. If None, the name will be set to `PrequentialError`.
    :type name: Optional[str]

    :References:

    .. [dawid1984present] Dawid, A. Philip.
        "Present position and potential developments:
        Some personal views statistical theory the prequential approach."
        Journal of the Royal Statistical Society:
        Series A (General) 147.2 (1984): 278-290.
    .. [gama2009issues] Gama, Joao, Raquel Sebastiao, and Pedro Pereira Rodrigues.
        "Issues in evaluation of stream learning algorithms."
        Proceedings of the 15th ACM SIGKDD international conference on Knowledge
        discovery and data mining. 2009.

    :Example:

    >>> from frouros.metrics import PrequentialError
    >>> metric = PrequentialError(alpha=0.9)
    >>> X = [1, 1, 0, 1, 0, 0]
    >>> Y = [1, 0, 0, 0, 1, 1]
    >>> for i, (X_sample, Y_sample) in enumerate(zip(X, Y)):
    ...     error_value = 1 - (X_sample == Y_sample)
    ...     prequential_error = metric(error_value=error_value)
    ...     print(f"Metric={prequential_error:.5f} at step {i}")
    Metric=0.00000 at step 0
    Metric=0.52632 at step 1
    Metric=0.33210 at step 2
    Metric=0.52632 at step 3
    Metric=0.64199 at step 4
    Metric=0.71839 at step 5
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        alpha: Union[int, float] = 1.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha
        self.cumulative_error = 0.0
        self.cumulative_instances = 0.0
        self.num_instances = 0

    @property
    def alpha(self) -> Union[int, float]:
        """Fading factor property.

        :return: fading factor value
        :rtype: Union[int, float]
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: Union[int, float]) -> None:
        """Fading factor setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        if not 0.0 < value <= 1.0:
            raise ValueError("value must be in the range (0, 1].")
        self._alpha = value

    @property
    def cumulative_instances(self) -> Union[int, float]:
        """Cumulative instances' property.

        :return: fading factor value
        :rtype: Union[int, float]
        """
        return self._cumulative_instances

    @cumulative_instances.setter
    def cumulative_instances(self, value: Union[int, float]) -> None:
        """Cumulative instances' setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self._cumulative_instances = value

    @property
    def cumulative_fading_error(self) -> Union[int, float]:
        """Cumulative fading error property.

        :return: cumulative facing error value
        :rtype: Union[int, float]
        """
        return self.cumulative_error / self.cumulative_instances

    def __call__(
        self,
        error_value: float,
    ) -> Union[int, float]:
        """__call__ method that updates the prequential error using fading factor.

        :param error_value error value
        :type error_value: float
        :return: cumulative facing error
        :rtype: Union[int, float]
        """
        self.cumulative_error = self.cumulative_error * self.alpha + error_value
        self.cumulative_instances = self.cumulative_instances * self.alpha + 1
        return self.cumulative_fading_error

    def reset(self) -> None:
        """Reset method."""
        self.cumulative_error = 0.0
        self.cumulative_instances = 0.0
        self.num_instances = 0

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return f"{super().__repr__()[:-1]}, alpha={self.alpha})"
