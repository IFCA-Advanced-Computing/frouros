"""History callback module."""

from typing import Any, Dict, List, Optional

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.utils.stats import BaseStat


class HistoryConceptDrift(BaseCallbackStreaming):
    """HistoryConceptDrift callback class."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Init method.

        :param name: name value
        :type name: Optional[str]
        """
        super().__init__(name=name)
        self.additional_vars: List[str] = []
        self.history: Dict[str, List[Any]] = {
            "value": [],
            "num_instances": [],
            "drift": [],
        }

    def add_additional_vars(self, vars_: List[str]) -> None:
        """Add addtional variables to track.

        :param vars_: list of variables
        :type vars_: List[str]
        """
        self.additional_vars.extend(vars_)
        self.history = {**self.history, **{var: [] for var in self.additional_vars}}

    def on_update_end(self, **kwargs) -> None:
        """On update end method."""
        self.history["value"].append(kwargs["value"])
        self.history["num_instances"].append(
            self.detector.num_instances  # type: ignore
        )
        self.history["drift"].append(self.detector.drift)  # type: ignore
        for var in self.additional_vars:
            additional_var = self.detector.additional_vars[var]  # type: ignore
            # FIXME: Extract isinstance check to be done when  # pylint: disable=fixme
            #  add_addtional_vars is called (avoid the same computation)
            self.history[var].append(
                additional_var.get()
                if isinstance(additional_var, BaseStat)
                else additional_var
            )

        self.logs.update(**self.history)

    # FIXME: set_detector method as a workaround to  # pylint: disable=fixme
    #  avoid circular import problem. Make it an abstract method and
    #  uncomment commented code when it is solved

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, BaseConceptDrift):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type BaseConceptDrift."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""
        for key in self.history.keys():
            self.history[key].clear()
