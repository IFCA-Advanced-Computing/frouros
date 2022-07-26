"""Data structures module."""

from typing import Optional, List, Union  # noqa: TYP001


import numpy as np  # type: ignore


class EmptyQueueError(Exception):
    """Empty queue exception."""

    def __init__(self, *args, msg="Queue is empty.", **kwargs) -> None:
        """Init method.

        :param msg: exception message
        :type msg: str
        """
        super().__init__(msg, *args, **kwargs)


class CircularQueue:
    """Class representing a circular queue."""

    def __init__(self, max_len: int) -> None:
        """Init method.

        :param max_len: maximum capacity
        :type max_len: int
        """
        self.count = 0
        self.first = 0
        self.last = -1
        self.max_len = max_len
        self.queue = [None] * self.max_len

    @property
    def count(self) -> int:
        """Number of total elements property.

        :return: Number of total elements
        :rtype: int
        """
        return self._count

    @count.setter
    def count(self, value: int) -> None:
        """Number of total elements setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("count must be greater or equal than 0.")
        self._count = value

    @property
    def first(self) -> int:
        """First queue index property.

        :return: first queue index
        :rtype: int
        """
        return self._first

    @first.setter
    def first(self, value: int) -> None:
        """First queue index setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("first must be greater or equal than 0.")
        self._first = value

    @property
    def last(self) -> int:
        """Last queue index property.

        :return: last queue index
        :rtype: int
        """
        return self._last

    @last.setter
    def last(self, value: int) -> None:
        """Last queue index setter.

        :param value: value to be set
        :type value: int
        """
        self._last = value

    @property
    def max_len(self) -> int:
        """Maximum number of allowed elements property.

        :return: maximum number of allowed elements
        :rtype: int
        """
        return self._max_len

    @max_len.setter
    def max_len(self, value: int) -> None:
        """Maximum number of allowed elements setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("max_len must be greater or equal than 0.")
        self._max_len = value

    @property
    def queue(self) -> List[Optional[bool]]:
        """Queue property.

        :return: queue
        :rtype: List[Optional[bool]]
        """
        return self._queue

    @queue.setter
    def queue(self, value: List[Optional[bool]]) -> None:
        """Queue setter.

        :param value: value to be set
        :type value: List[Optional[bool]]
        :raises ValueError: Value error exception
        """
        if not isinstance(value, list):
            raise TypeError("queue must be of type list.")
        self._queue = value

    @property
    def size(self) -> int:
        """Number of current elements property.

        :return: Number of current elements
        :rtype: int
        """
        return self.count

    def clear(self) -> None:
        """Clear queue."""
        self.count = 0
        self.first = 0
        self.last = -1
        self.queue = [None] * self.max_len

    def dequeue(self) -> bool:
        """Dequeue oldest element.

        :rtype: bool
        :raises EmptyQueue: Empty queue error exception
        """
        if self.is_empty():
            raise EmptyQueueError()
        element = self.queue[self.first]
        self.first = (self.first + 1) % self.max_len
        self.count -= 1
        return element  # type: ignore

    def enqueue(self, value: Union[np.ndarray, float]) -> None:
        """Enqueue element/s.

        :param value: value to be enqueued
        :type value: Union[np.ndarray, float]
        """
        if self.is_full():
            _ = self.dequeue()
        self.last = (self.last + 1) % self.max_len
        self.queue[self.last] = value  # type: ignore
        self.count += 1

    def is_empty(self) -> bool:
        """Check if queue is empty.

        :return: check if queue is empty
        :rtype: bool
        """
        return self.size == 0

    def is_full(self) -> bool:
        """Check if queue is full.

        :return: check if queue is full
        :rtype: bool
        """
        return self.size == self.max_len

    def __len__(self) -> int:
        """Queue size.

        :return: queue size
        :rtype: int
        """
        return self.size

    def maintain_last_element(self) -> None:
        """Clear all elements except the last one."""
        self.first = self.last
        self.count = 1

    def __getitem__(self, idx: int) -> float:
        """Get queue item by position.

        :param idx: position index
        :type idx: int
        :return: queue item
        :rtype: float
        """
        return self.queue[idx]  # type: ignore


class AccuracyQueue(CircularQueue):
    """Class representing an accuracy queue."""

    def __init__(self, max_len: int) -> None:
        """Init method.

        :param max_len: maximum capacity
        :type max_len: int
        """
        super().__init__(max_len=max_len)
        self.num_true = 0

    @property
    def num_false(self):
        """Number of false label property.

        :return: number of false labels
        :rtype: int
        """
        return self.count - self.num_true

    @property
    def num_true(self) -> int:
        """Number of true label property.

        :return: number of true labels
        :rtype: int
        """
        return self._num_true

    @num_true.setter
    def num_true(self, value: int) -> None:
        """Number of true labels setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_true value must be greater or equal than 0.")
        self._num_true = value

    def clear(self) -> None:
        """Clear queue."""
        super().clear()
        self.num_true = 0

    def dequeue(self) -> bool:
        """Dequeue oldest element.

        :return oldest element
        :rtype: bool
        :raises EmptyQueue: Empty queue error exception
        """
        element = super().dequeue()
        self.num_true -= 1 if element else 0
        return element

    def enqueue(self, value: Union[np.ndarray, float]) -> None:
        """Enqueue element/s.

        :param value: value to be enqueued
        :type value: Union[np.ndarray, float]
        """
        super().enqueue(value=value)
        self.num_true += np.count_nonzero(value)
