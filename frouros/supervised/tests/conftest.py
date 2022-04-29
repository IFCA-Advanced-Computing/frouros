"""Configuration file for the tests."""

from typing import Tuple

import pytest  # type: ignore
import numpy as np  # type: ignore

from frouros.datasets.real import Elec2


@pytest.fixture(scope="module")
def classification_dataset() -> Tuple[np.array, np.array, np.array, np.array]:
    """Classification dataset using Elec2.

    :return: classification dataset
    :rtype: Tuple[np.array, np.array, np.array, np.array]
    """
    elec2 = Elec2()
    elec2.download()
    dataset = elec2.load()

    # Two comprehensions lists are faster than iterating over one
    # (Python doing Python´s things).
    X = np.array(  # noqa: N806
        [sample.tolist()[:-1] for sample in dataset], dtype=np.float16
    )
    y = np.array([sample[-1] for sample in dataset], dtype="str")

    split_idx = 43500

    X_ref = X[:split_idx]  # noqa: N806
    y_ref = y[:split_idx]

    X_test = X[split_idx:]  # noqa: N806
    y_test = y[split_idx:]

    return X_ref, y_ref, X_test, y_test
