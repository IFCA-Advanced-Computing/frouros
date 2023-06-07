"""Reset batch callback module."""

from typing import Optional

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.utils.logger import logger


class ResetStatisticalTest(BaseCallbackBatch):
    """Reset on statistical test batch callback class."""

    def __init__(self, alpha: float, name: Optional[str] = None) -> None:
        """Init method.

        :param alpha: significance value
        :type alpha: float
        :param name: name to be use
        :type name: Optional[str]
        """
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

    def on_compare_end(self, **kwargs) -> None:
        """On compare end method."""
        p_value = kwargs["result"].p_value
        if p_value < self.alpha:
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
