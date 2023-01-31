"""Callback module."""


class Callback:
    """Abstract class representing a callback."""

    def __init__(self) -> None:
        """Init method."""
        self.detector = None

    # FIXME: Workaround to avoid circular import problem  # pylint: disable=fixme
    def set_detector(self, detector) -> None:
        """Set detector method."""
        self.detector = detector

    # @property
    # def detector(self) -> Optional[ConceptDriftBase, DataDriftBatchBase]:
    #     return self._detector
    #
    # @detector.setter
    # def detector(self, value: Optional[ConceptDriftBase, DataDriftBatchBase]) -> None:
    #     if not isinstance(
    #             value, (ConceptDriftBase, DataDriftBatchBase)):
    #         raise TypeError(
    #             "value must be of type ConceptDriftBase or DataDriftBatchBase."
    #         )
    #     self._detector = value

    def on_fit_start(self) -> None:
        """On fit start method."""

    def on_fit_end(self) -> None:
        """On fit end method."""

    def on_drift_detected(self) -> None:
        """On drift detected method."""


class StreamingCallback(Callback):
    """Streaming callback class."""

    def on_update_start(self) -> None:
        """On update start method."""

    def on_update_end(self) -> None:
        """On update end method."""


class BatchCallback(Callback):
    """Batch callback class."""

    def on_compare_start(self) -> None:
        """On compare start method."""

    def on_compare_end(self, **kwargs) -> None:
        """On compare end method."""


class ResetOnBatchDataDrift(BatchCallback):
    """Reset on batch data drift callback class."""

    def __init__(self, alpha: float) -> None:
        """Init method.

        :param alpha: significance value
        :type alpha: float
        """
        super().__init__()
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
        """
        if value <= 0.0:
            raise ValueError("value must be greater than 0.")
        self._alpha = value

    def on_compare_end(self, **kwargs) -> None:
        """On compare end method."""
        p_value = kwargs["result"].p_value
        if p_value < self.alpha:
            self.detector.reset()  # type: ignore
