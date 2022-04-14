"""Configuration file for the tests."""

from typing import Tuple

import pytest  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from frouros.datasets.real import Elec2


@pytest.fixture(scope="function")
def classification_dataset() -> Tuple[np.array, np.array, np.array, np.array]:
    """Classification dataset using Elec2.

    :return: classification dataset
    :rtype: Tuple[np.array, np.array, np.array, np.array]
    """
    elec2 = Elec2()
    elec2.download()
    dataset = elec2.load()

    df = pd.DataFrame(dataset)

    target = "class"

    X = (  # noqa: N806
        df.loc[:, df.columns != target].apply(pd.to_numeric, errors="coerce").to_numpy()
    )
    y = df["class"].to_numpy(dtype="str")

    split_idx = 43500

    X_ref = X[:split_idx]  # noqa: N806
    y_ref = y[:split_idx]

    X_test = X[split_idx:]  # noqa: N806
    y_test = y[split_idx:]

    return X_ref, y_ref, X_test, y_test
