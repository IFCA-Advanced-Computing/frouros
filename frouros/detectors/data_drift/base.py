"""Data drift base module."""


import abc
from collections import namedtuple
import operator
from typing import List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.base import DetectorBase

TestResult = namedtuple("TestResult", ["statistic", "p_value"])


class DataTypeBase(abc.ABC):
    """Abstract class representing a data type."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Init method."""


class CategoricalData(DataTypeBase):
    """Class representing categorical data."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.output_type = None


class NumericalData(DataTypeBase):
    """Class representing numerical data."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.output_type = np.float32


class StatisticalTypeBase(abc.ABC):
    """Abstract class representing a statistical data type."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Init method."""


class UnivariateData(StatisticalTypeBase):
    """Class representing a univariate data type."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.dim_check = operator.eq


class MultivariateData(StatisticalTypeBase):
    """Class representing a multivariate data type."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.dim_check = operator.ge


class DataDriftBase(DetectorBase):
    """Abstract class representing a data drift detector."""

    def __init__(
        self,
        data_type: DataTypeBase,
        statistical_type: StatisticalTypeBase,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: DataTypeBase
        :param statistical_type: statistical type
        :type statistical_type: StatisticalTypeBase
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(callbacks=callbacks)
        self.data_type = data_type
        self.statistical_type = statistical_type
        self.X_ref = None  # type: ignore

    @property
    def data_type(self) -> DataTypeBase:
        """Data type property.

        :return: data type
        :rtype: DataTypeBase
        """
        return self._data_type

    @data_type.setter
    def data_type(self, value: DataTypeBase) -> None:
        """Data type setter.

        :param value: value to be set
        :type value: DataTypeBase
        :raises TypeError: Type error exception
        """
        if not isinstance(value, DataTypeBase):
            raise TypeError("value must be of type DataTypeBase.")
        self._data_type = value

    @property
    def statistical_type(self) -> StatisticalTypeBase:
        """Statistical type property.

        :return: statistical type
        :rtype: StatisticalTypeBase
        """
        return self._statistical_type

    @statistical_type.setter
    def statistical_type(self, value: StatisticalTypeBase) -> None:
        """Statistical type setter.

        :param value: value to be set
        :type value: StatisticalTypeBase
        :raises TypeError: Type error exception
        """
        if not isinstance(value, StatisticalTypeBase):
            raise TypeError("value must be of type StatisticalTypeBase.")
        self._statistical_type = value

    @property
    def X_ref(self) -> Optional[np.ndarray]:  # noqa: N802
        """Reference data property.

        :return: reference data
        :rtype: Optional[numpy.ndarray]
        """
        return self._X_ref  # type: ignore # pylint: disable=E1101

    @X_ref.setter  # type: ignore
    def X_ref(self, value: Optional[np.ndarray]) -> None:  # noqa: N802
        """Reference data setter.

        :param value: value to be set
        :type value: Optional[numpy.ndarray]
        """
        if value is not None:
            self._check_array(X=value)
        self._X_ref = value

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""

    @abc.abstractmethod
    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> np.ndarray:
        pass
