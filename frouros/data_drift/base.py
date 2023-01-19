"""Data drift base module."""


import abc
from collections import namedtuple
import operator

import numpy as np  # type: ignore

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
        self.output_type = None


class NumericalData(DataTypeBase):
    """Class representing numerical data."""

    def __init__(self) -> None:
        """Init method."""
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
        self.dim_check = operator.eq


class MultivariateData(StatisticalTypeBase):
    """Class representing a multivariate data type."""

    def __init__(self) -> None:
        """Init method."""
        self.dim_check = operator.gt
