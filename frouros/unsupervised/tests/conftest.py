"""Configuration file for the tests."""

from typing import Tuple

import pytest  # type: ignore
import pandas as pd  # type: ignore
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

    df = pd.DataFrame(dataset_)

    target = "class"

    X = (  # noqa: N806
        df.loc[:, df.columns != target].apply(pd.to_numeric, errors="coerce").to_numpy()
    )
    y = df["class"].to_numpy(dtype="str")

    X_ref = X[:-1]  # noqa: N806
    y_ref = y[:-1]

    X_test = X[-2:, :]  # noqa: N806

    return X_ref, y_ref, X_test
