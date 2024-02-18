"""Test PermutationTestDistanceBased module."""

import multiprocessing

import numpy as np  # type: ignore
import pytest  # type: ignore

from frouros.callbacks.batch.permutation_test import PermutationTestDistanceBased


def test_initialization_with_valid_input_parameters():
    """Test the initialization with valid input parameters."""
    num_permutations = 1000
    total_num_permutations = None
    num_jobs = -1
    method = "auto"
    random_state = 31
    verbose = False
    name = "permutation_test"

    callback = PermutationTestDistanceBased(
        num_permutations=num_permutations,
        total_num_permutations=total_num_permutations,
        num_jobs=num_jobs,
        method=method,
        random_state=random_state,
        verbose=verbose,
        name=name,
    )

    assert callback.num_permutations == num_permutations
    assert callback.total_num_permutations == total_num_permutations
    assert callback.num_jobs == multiprocessing.cpu_count()
    assert callback.method == method
    assert callback.random_state == random_state
    assert callback.verbose == verbose
    assert callback.name == name


def test_calculate_p_value_with_valid_input_parameters(mocker):
    """Test the calculation of p-value with valid input parameters."""

    def statistic(x, y):
        return np.abs(np.mean(x) - np.mean(y))

    X_ref = np.array([1, 2, 3, 4, 5])  # noqa: N806
    X_test = np.array([6, 7, 8, 9, 10])  # noqa: N806
    statistic_args = {}
    observed_statistic = 1.0
    num_permutations = 1000
    total_num_permutations = None
    num_jobs = 1
    method = "auto"
    random_state = 31
    verbose = False

    permuted_statistic = [0.5, 0.6, 0.7, 0.8, 0.9]
    p_value = 0.166666167

    mocker.patch(
        "frouros.callbacks.batch.permutation_test.permutation",
        return_value=(permuted_statistic, 1000000),
    )
    result = PermutationTestDistanceBased._calculate_p_value(
        X_ref=X_ref,
        X_test=X_test,
        statistic=statistic,
        statistic_args=statistic_args,
        observed_statistic=observed_statistic,
        num_permutations=num_permutations,
        total_num_permutations=total_num_permutations,
        num_jobs=num_jobs,
        method=method,
        random_state=random_state,
        verbose=verbose,
    )

    np.testing.assert_array_equal(result[0], np.array(permuted_statistic), strict=True)
    np.testing.assert_allclose(result[1], p_value, rtol=1e-6)


def test_compute_estimate_with_valid_input_parameters():
    """Test the computation of estimate with valid input parameters."""
    extreme_statistic = np.array([True, False, True, True, False])

    p_value = PermutationTestDistanceBased._compute_estimate(extreme_statistic)

    np.testing.assert_allclose(p_value, 0.6, rtol=1e-6)


def test_initialization_with_invalid_num_permutations():
    """Test the initialization with invalid number of permutations."""
    num_permutations = 0
    total_num_permutations = None
    num_jobs = -1
    method = "auto"
    random_state = 31
    verbose = False
    name = "permutation_test"

    with pytest.raises(ValueError):
        PermutationTestDistanceBased(
            num_permutations=num_permutations,
            total_num_permutations=total_num_permutations,
            num_jobs=num_jobs,
            method=method,
            random_state=random_state,
            verbose=verbose,
            name=name,
        )


def test_initialization_with_invalid_num_permutations_exceeding_max():
    """Test the initialization with number of permutations exceeding the maximum."""
    num_permutations = 1000001
    total_num_permutations = None
    num_jobs = -1
    method = "auto"
    random_state = 31
    verbose = False
    name = "permutation_test"

    with pytest.raises(ValueError):
        PermutationTestDistanceBased(
            num_permutations=num_permutations,
            total_num_permutations=total_num_permutations,
            num_jobs=num_jobs,
            method=method,
            random_state=random_state,
            verbose=verbose,
            name=name,
        )


def test_compute_approximate_valid_input():
    """Test the computation of approximate value with valid input parameters."""
    extreme_statistic = np.array([True, False, True, True, False])
    total_num_permutations = 1000
    permuted_statistic = np.array([True, False, True, True, False])

    p_value = PermutationTestDistanceBased._compute_approximate(
        extreme_statistic, total_num_permutations, permuted_statistic
    )

    np.testing.assert_allclose(p_value, 0.6666664166666666, rtol=1e-9)


def test_compute_exact_valid_input():
    """Test the computation of exact value with valid input parameters."""
    extreme_statistic = np.array([True, False, True, True, False])
    total_num_permutations = 1000
    permuted_statistic = np.array([True, False, True, True, False])

    p_value = PermutationTestDistanceBased._compute_exact(
        extreme_statistic, total_num_permutations, permuted_statistic
    )

    np.testing.assert_allclose(p_value, 0.6661666666665, rtol=1e-9)


def test_compute_conservative_valid_input():
    """Test the computation of conservative value with valid input parameters."""
    num_permutations = 1000
    observed_statistic = 0.5
    permuted_statistic = np.array([0.4, 0.6, 0.7, 0.3, 0.2])

    p_value = PermutationTestDistanceBased._compute_conservative(
        num_permutations, observed_statistic, permuted_statistic
    )

    np.testing.assert_allclose(p_value, 0.002997002997002997, rtol=1e-9)


def test_compute_estimate_valid_input():
    """Test the computation of estimate with valid input parameters."""
    extreme_statistic = np.array([True, False, True, True, False])

    p_value = PermutationTestDistanceBased._compute_estimate(extreme_statistic)

    np.testing.assert_allclose(p_value, 0.6, rtol=1e-9)
