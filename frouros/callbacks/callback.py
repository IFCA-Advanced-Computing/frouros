"""Callback module."""

import abc
import copy
import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore

# FIXME: set_detector method as a workaround to  # pylint: disable=fixme
#  avoid circular import problem. Make it an abstract method and
#  uncomment commented code when it is solved

# from frouros.detectors.concept_drift.base import ConceptDriftBase
# from frouros.detectors.concept_drift.ddm_based.base import DDMBased
# from frouros.detectors.data_drift.batch.base import DataDriftBatchBase
from frouros.utils.stats import permutation_test, Stat


class Callback:
    """Abstract class representing a callback."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Init method.

        :param name: name value
        :type name: Optional[str]
        """
        self.name = name  # type: ignore
        self.detector = None
        self.logs = {}  # type: ignore

    @property
    def name(self) -> str:
        """Name property.

        :return: name value
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        """Name method setter.

        :param value: value to be set
        :type value: Optional[str]
        :raises TypeError: Type error exception
        """
        if not isinstance(value, str) and value is not None:
            raise TypeError("name must be of type str or None.")
        self._name = self.__class__.__name__ if value is None else value

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

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""


class StreamingCallback(Callback):
    """Streaming callback class."""

    def on_update_start(self, value: Union[int, float], **kwargs) -> None:
        """On update start method."""

    def on_update_end(self, value: Union[int, float], **kwargs) -> None:
        """On update end method."""

    # @abc.abstractmethod
    # def set_detector(self, detector) -> None:
    #     """Set detector method."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""


class BatchCallback(Callback):
    """Batch callback class."""

    def on_compare_start(self) -> None:
        """On compare start method."""

    def on_compare_end(self, **kwargs) -> None:
        """On compare end method."""

    # @abc.abstractmethod
    # def set_detector(self, detector) -> None:
    #     """Set detector method."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""


class History(StreamingCallback):
    """History callback class."""

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

    def on_update_end(self, value: Union[int, float], **kwargs) -> None:
        """On update end method.

        :param value: value to update detector
        :type value: int
        """
        self.history["value"].append(value)
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
                if isinstance(additional_var, Stat)
                else additional_var
            )

        self.logs.update(**self.history)

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, ConceptDriftBase):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type ConceptDriftBase."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""
        for key in self.history.keys():
            self.history[key].clear()


class PermutationTestOnBatchData(BatchCallback):
    """Permutation test on batch data callback class."""

    def __init__(
        self,
        num_permutations: int,
        num_jobs: int = -1,
        name: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Init method.

        :param num_permutations: number of permutations
        :type num_permutations: int
        :param num_jobs: number of jobs
        :type num_jobs: int
        :param name: name to be use
        :type name: Optional[str]
        """
        super().__init__(name=name)
        self.num_permutations = num_permutations
        self.num_jobs = num_jobs
        self.verbose = verbose
        self.permutation_kwargs = kwargs

    @property
    def num_permutations(self) -> int:
        """Number of permutations property.

        :return: number of permutation to obtain the p-value
        :rtype: int
        """
        return self._num_permutations

    @num_permutations.setter
    def num_permutations(self, value: int) -> None:
        """Number of permutations method setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("value must be greater of equal than 1.")
        self._num_permutations = value

    @property
    def num_jobs(self) -> int:
        """Number of jobs property.

        :return: number of jobs to use
        :rtype: int
        """
        return self._num_jobs

    @num_jobs.setter
    def num_jobs(self, value: int) -> None:
        """Number of jobs method setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value == 0 or value < -1:
            raise ValueError("value must be greater than 0 or -1.")
        self._num_jobs = multiprocessing.cpu_count() if value == -1 else value

    @property
    def verbose(self) -> bool:
        """Verbose flag property.

        :return: verbose flag
        :rtype: bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Verbose flag setter.

        :param value: value to be set
        :type value: bool
        :raises TypeError: Type error exception
        """
        if not isinstance(value, bool):
            raise TypeError("value must of type bool.")
        self._verbose = value

    @staticmethod
    def _calculate_p_value(  # pylint: disable=too-many-arguments
        X_ref: np.ndarray,  # noqa: N803
        X_test: np.ndarray,
        statistic: Callable,
        statistic_args: Dict[str, Any],
        observed_statistic: float,
        num_permutations: int,
        num_jobs: int,
        random_state: int,
        verbose: bool,
    ) -> Tuple[List[float], float]:
        permuted_statistic = permutation_test(
            X=X_ref,
            Y=X_test,
            statistic=statistic,
            statistical_args=statistic_args,
            num_permutations=num_permutations,
            num_jobs=num_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        p_value = (observed_statistic < permuted_statistic).mean()  # type: ignore
        return permuted_statistic, p_value

    def on_compare_end(self, **kwargs) -> None:
        """On compare end method."""
        X_ref, X_test = kwargs["X_ref"], kwargs["X_test"]  # noqa: N806
        observed_statistic = kwargs["result"][0]
        permuted_statistics, p_value = self._calculate_p_value(
            X_ref=X_ref,
            X_test=X_test,
            statistic=self.detector.statistical_method,  # type: ignore
            statistic_args=self.detector.statistical_kwargs,  # type: ignore
            observed_statistic=observed_statistic,
            num_permutations=self.num_permutations,
            num_jobs=self.num_jobs,
            verbose=self.verbose,
            **self.permutation_kwargs,
        )
        self.logs.update(
            {
                "observed_statistic": observed_statistic,
                "permuted_statistics": permuted_statistics,
                "p_value": p_value,
            },
        )

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, DataDriftBatchBase):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type DataDriftBatchBase."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""


class ResetOnBatchDataDrift(BatchCallback):
    """Reset on batch data drift callback class."""

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
            print("Drift detected. Resetting detector.")
            self.detector.reset()  # type: ignore

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, DataDriftBatchBase):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type DataDriftBatchBase."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""


class WarningSamplesBuffer(StreamingCallback):
    """Store warning samples as a buffer callback class."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Init method.

        :param name: name to be use
        :type name: Optional[str]
        """
        super().__init__(name=name)
        self.X: List[Any] = []
        self.y: List[Any] = []
        self._start_warning = False

    def on_update_start(self, value: Union[int, float], **kwargs) -> None:
        """On update start method."""
        self._start_warning = not self.detector.warning  # type: ignore

    def on_update_end(self, value: Union[int, float], **kwargs) -> None:
        """On update end method.

        :param value: value to update detector
        :type value: int
        """
        self.logs = {
            "X": copy.deepcopy(self.X),
            "y": copy.deepcopy(self.y),
        }

    def on_warning_detected(self, **kwargs) -> None:
        """On warning detected method."""
        if self._start_warning:
            map(lambda x: x.clear(), [self.X, self.y])
        self.X.append(kwargs["X"])
        self.y.append(kwargs["y"])

    # def set_detector(self, detector) -> None:
    #     """Set detector method.
    #
    #     :raises TypeError: Type error exception
    #     """
    #     if not isinstance(detector, DDMBased):
    #         raise TypeError(
    #             f"callback {self.__class__.name} cannot be used with detector"
    #             f" {detector.__class__name}. Must be used with a detector of "
    #             f"type DDMBased."
    #         )
    #     self.detector = detector

    def reset(self) -> None:
        """Reset method."""
        self.X.clear()
        self.y.clear()
        self._start_warning = False
