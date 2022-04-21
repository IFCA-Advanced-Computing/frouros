"""Configuration file for the tests."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore

from frouros.datasets.real import Elec2


@pytest.fixture(scope="module")
def dataset() -> Tuple[np.array, np.array, np.array]:
    """Dataset using Elec2.

    :return: dataset
    :rtype: Tuple[np.array, np.array, np.array]
    """
    elec2 = Elec2()
    elec2.download()
    dataset_ = elec2.load()

    # Two comprehensions lists are faster than iterating over one
    # (Python doing Python´s things).
    X = np.array(  # noqa: N806
        [sample.tolist()[:-1] for sample in dataset_], dtype=np.float16
    )
    y = np.array([sample[-1] for sample in dataset_], dtype="str")

    X_ref = X[:-1]  # noqa: N806
    y_ref = y[:-1]

    X_test = X[-2:, :]  # noqa: N806

    return X_ref, y_ref, X_test
