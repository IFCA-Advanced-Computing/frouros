"""Test checks module."""

from typing import Any

import pytest  # type: ignore

from frouros.callbacks.base import BaseCallback
from frouros.callbacks.batch import PermutationTestDistanceBased, ResetStatisticalTest
from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.callbacks.streaming import HistoryConceptDrift, WarningSamplesBuffer
from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.utils.checks import check_callbacks


@pytest.mark.parametrize(
    "callbacks, expected_cls",
    [
        (
            None,
            BaseCallbackBatch,
        ),
        (
            PermutationTestDistanceBased(
                num_permutations=10,
            ),
            BaseCallbackBatch,
        ),
        (
            None,
            BaseCallbackStreaming,
        ),
        (
            [
                PermutationTestDistanceBased(
                    num_permutations=10,
                ),
                ResetStatisticalTest(
                    alpha=0.05,
                ),
            ],
            BaseCallbackBatch,
        ),
        (
            HistoryConceptDrift(),
            BaseCallbackStreaming,
        ),
        (
            [HistoryConceptDrift(), WarningSamplesBuffer()],
            BaseCallbackStreaming,
        ),
    ],
)
def test_check_callbacks(
    callbacks: Any,
    expected_cls: BaseCallback,
) -> None:
    """Test check_callbacks function.

    :param callbacks: callbacks
    :type callbacks: Any
    :param expected_cls: expected callback class
    :type expected_cls: BaseCallback
    """
    check_callbacks(
        callbacks=callbacks,
        expected_cls=expected_cls,
    )


@pytest.mark.parametrize(
    "callbacks, expected_cls",
    [
        (
            PermutationTestDistanceBased(
                num_permutations=10,
            ),
            BaseCallbackStreaming,
        ),
        (
            [
                PermutationTestDistanceBased(
                    num_permutations=10,
                ),
                ResetStatisticalTest(
                    alpha=0.05,
                ),
            ],
            BaseCallbackStreaming,
        ),
        (
            HistoryConceptDrift(),
            BaseCallbackBatch,
        ),
        (
            [HistoryConceptDrift(), WarningSamplesBuffer()],
            BaseCallbackBatch,
        ),
    ],
)
def test_check_callbacks_exceptions(
    callbacks: Any,
    expected_cls: BaseCallback,
) -> None:
    """Test check_callbacks function exceptions.

    :param callbacks: callbacks
    :type callbacks: Any
    :param expected_cls: expected callback class
    :type expected_cls: BaseCallback
    """
    with pytest.raises(TypeError):
        check_callbacks(
            callbacks=callbacks,
            expected_cls=expected_cls,
        )
