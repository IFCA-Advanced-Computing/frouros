"""Reset batch callback module."""

from typing import Any, Optional

import numpy as np

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.utils.logger import logger


class ResetStatisticalTest(BaseCallbackBatch):
    """Reset callback class that can be applied to :mod:`data_drift.batch.statistical_test <frouros.detectors.data_drift.batch.statistical_test>` detectors.

    :param alpha: significance value
    :type alpha: float
    :param name: name value, defaults to None. If None, the name will be set to `ResetStatisticalTest`.
    :type name: Optional[str]

    :Example:

    >>> from frouros.callbacks import ResetStatisticalTest
    >>> from frouros.detectors.data_drift import KSTest
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = KSTest(callbacks=ResetStatisticalTest(alpha=0.01))
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    INFO:frouros:Drift detected. Resetting detector...
    StatisticalResult(statistic=0.55, p_value=3.0406585087050305e-14)
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        alpha: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        """Alpha property.

        :return: significance value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Alpha setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("value must be greater than 0.")
        self._alpha = value

    def on_compare_end(
        self,
        result: Any,
        X_ref: np.ndarray,  # noqa: N803
        X_test: np.ndarray,
    ) -> None:
        """On compare end method.

        :param result: result obtained from the `compare` method
        :type result: Any
        :param X_ref: reference data
        :type X_ref: numpy.ndarray
        :param X_test: test data
        :type X_test: numpy.ndarray
        """
        p_value = result.p_value
        if p_value <= self.alpha:
            logger.info("Drift detected. Resetting detector...")
            self.detector.reset()  # type: ignore

    # FIXME: set_detector method as a workaround to  # pylint: disable=fixme
    #  avoid circular import problem. Make it an abstract method and
    #  uncomment commented code when it is solved

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, BaseDataDriftBatch):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type BaseDataDriftBatch."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""
