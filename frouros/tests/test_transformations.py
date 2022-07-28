"""Test transformations module."""

import numpy as np  # type: ignore

from frouros.transformations import CategoricalDetectors, NumericalDetectors
from frouros.unsupervised.statistical_test import ChiSquareTest, KSTest


def test_categorical_transformation(categorical_dataset: np.ndarray) -> None:
    """Test categorical detectors transformation.

    :param categorical_dataset: categorical dataset
    :type categorical_dataset: Tuple[numpy.array, numpy.array]
    """
    X_ref, X_test = categorical_dataset  # noqa: N806

    detectors = [ChiSquareTest()]

    categorical_detectors = CategoricalDetectors(detectors=detectors, columns=[0, 1])
    categorical_detectors.fit(X=X_ref)
    categorical_detectors.transform(X=X_test)

    test = categorical_detectors.detectors[0].test
    for (statistic, p_value), (expected_statistic, expected_p_value) in zip(
        test, [(7.19999, 0.0273237), (0.53333, 0.7659283)]  # type: ignore
    ):
        assert np.isclose(statistic, expected_statistic)
        assert np.isclose(p_value, expected_p_value)


def test_numeric_transformation(numerical_dataset: np.ndarray) -> None:
    """Test numeric detectors transformation.

    :param numerical_dataset: numerical dataset
    :type numerical_dataset: Tuple[numpy.array, numpy.array]
    """
    X_ref, X_test = numerical_dataset  # noqa: N806

    detectors = [KSTest()]

    numerical_detectors = NumericalDetectors(detectors=detectors)
    numerical_detectors.fit(X=X_ref)
    numerical_detectors.transform(X=X_test)

    test = numerical_detectors.detectors[0].test
    for (statistic, p_value), (expected_statistic, expected_p_value) in zip(
        test, [(0.3, 0.78692978), (1.0, 1.08250882e-05)]  # type: ignore
    ):
        assert np.isclose(statistic, expected_statistic)
        assert np.isclose(p_value, expected_p_value)
