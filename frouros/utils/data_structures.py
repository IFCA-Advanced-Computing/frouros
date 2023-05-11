"""Data structures module."""

from typing import Any, Optional, List, Union

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
    def queue(self) -> List[Optional[Any]]:
        """Queue property.

        :return: queue
        :rtype: List[Optional[Any]]
        """
        return self._queue

    @queue.setter
    def queue(self, value: List[Optional[Any]]) -> None:
        """Queue setter.

        :param value: value to be set
        :type value: List[Optional[Any]]
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

    def dequeue(self) -> Any:
        """Dequeue oldest element.

        :rtype: value: Any
        :raises EmptyQueue: Empty queue error exception
        """
        if self.is_empty():
            raise EmptyQueueError()
        element = self.queue[self.first]
        self.first = (self.first + 1) % self.max_len
        self.count -= 1
        return element  # type: ignore

    def enqueue(self, value: Union[np.ndarray, int, float]) -> Optional[Any]:
        """Enqueue element/s.

        :param value: value to be enqueued
        :type value: Union[np.ndarray, float]
        :return element: dequeued element
        :rtype: Optional[Any]
        """
        element = self.dequeue() if self.is_full() else None
        self.last = (self.last + 1) % self.max_len
        self.queue[self.last] = value  # type: ignore
        self.count += 1
        return element

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

    def __getitem__(self, idx: int) -> Any:
        """Get queue item by position.

        :param idx: position index
        :type idx: int
        :return: queue item
        :rtype: Any
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


# FIXME: There seem to be a bug on the treap DS. Uncomment all  # pylint: disable=fixme
#  the commented code lines when that is solved.
# class Node:
#     """Class representing a node of a treap."""
#
#     def __init__(
#         self,
#         key: Tuple[float, Union[int, float]],
#         value: Union[int, float],
#     ) -> None:
#         """Init method.
#
#         :param key: key
#         :type key: Tuple[float, Union[int, float]]
#         :param value: value
#         :type value: Union[int, float]]
#         """
#         self.key = key
#         self.value_ = value
#         self.priority = np.random.rand()
#         self.size = 1
#         self.height = 1
#         self.lazy = 0
#         self.max_value = value
#         self.min_value = value
#         self.left = None
#         self.right = None
#
#     @property
#     def key(self) -> Tuple[float, Union[int, float]]:
#         """Key property.
#
#         :return: key value
#         :rtype: Tuple[float, Union[int, float]]
#         """
#         return self._key
#
#     @key.setter
#     def key(self, value: Tuple[float, Union[int, float]]) -> None:
#         """Key setter.
#
#         :param value: value to be set
#         :type value: int
#         :raises TypeError: Type error exception
#         """
#         if not (isinstance(value[0], float) and isinstance(value[1], (int, float))):
#             raise TypeError("value must be a tuple of float and int or float.")
#         self._key = value
#
#     @property
#     def value_(self) -> Union[int, float]:
#         """Value property.
#
#         :return: value
#         :rtype: Union[int, float]
#         """
#         return self._value
#
#     @value_.setter
#     def value_(self, value: Union[int, float]) -> None:
#         """Value setter.
#
#         :param value: value to be set
#         :type value: int
#         :raises TypeError: Type error exception
#         """
#         if not isinstance(value, (int, float)):
#             raise TypeError("value must be of type int or float.")
#         self._value = value
#
#     @property
#     def priority(self) -> float:
#         """Priority property.
#
#         :return: priority value
#         :rtype: float
#         """
#         return self._priority
#
#     @priority.setter
#     def priority(self, value: float) -> None:
#         """Value setter.
#
#         :param value: value to be set
#         :type value: float
#         :raises ValueError: Value error exception
#         """
#         if not 0.0 <= value <= 1.0:
#             raise ValueError("value must be in the range [0, 1].")
#         self._priority = value
#
#     @property
#     def size(self) -> int:
#         """Size property.
#
#         :return: size value
#         :rtype: int
#         """
#         return self._size
#
#     @size.setter
#     def size(self, value: int) -> None:
#         """Size setter.
#
#         :param value: value to be set
#         :type value: int
#         :raises ValueError: Value error exception
#         """
#         if value < 0:
#             raise ValueError("value must be a positive number.")
#         self._size = value
#
#     @property
#     def height(self) -> int:
#         """Height property.
#
#         :return: height value
#         :rtype: int
#         """
#         return self._height
#
#     @height.setter
#     def height(self, value: int) -> None:
#         """Height setter.
#
#         :param value: value to be set
#         :type value: int
#         :raises ValueError: Value error exception
#         """
#         if value < 0:
#             raise ValueError("value must be a positive number.")
#         self._height = value
#
#     @property
#     def lazy(self) -> Union[int, float]:
#         """Lazy property.
#
#         :return: lazy value
#         :rtype: Union[int, float]
#         """
#         return self._lazy
#
#     @lazy.setter
#     def lazy(self, value: Union[int, float]) -> None:
#         """Lazy setter.
#
#         :param value: value to be set
#         :type value: Union[int, float]
#         """
#         self._lazy = value
#
#     @property
#     def max_value(self) -> Union[int, float]:
#         """Maximum value property.
#
#         :return: maximum value
#         :rtype: Union[int, float]
#         """
#         return self._max_value
#
#     @max_value.setter
#     def max_value(self, value: Union[int, float]) -> None:
#         """Maximum setter.
#
#         :param value: value to be set
#         :type value: Union[int, float]
#         """
#         self._max_value = value
#
#     @property
#     def min_value(self) -> Union[int, float]:
#         """Minimum value property.
#
#         :return: minimum value
#         :rtype: Union[int, float]
#         """
#         return self._min_value
#
#     @min_value.setter
#     def min_value(self, value: Union[int, float]) -> None:
#         """Minimum setter.
#
#         :param value: value to be set
#         :type value: Union[int, float]
#         """
#         self._min_value = value
#
#     @property
#     def left(self):
#         """Left node property.
#
#         :return: left node
#         :rtype: Optional[Node]
#         """
#         return self._left
#
#     @left.setter
#     def left(self, value) -> None:
#         """Left node setter.
#
#         :param value: value to be set
#         :type value: Optional[Node]
#         """
#         self._left = value
#
#     @property
#     def right(self):
#         """Right node property.
#
#         :return: right node
#         :rtype: Optional[Node]
#         """
#         return self._right
#
#     @right.setter
#     def right(self, value) -> None:
#         """Right node setter.
#
#         :param value: value to be set
#         :type value: Optional[Node]
#         """
#         self._right = value
#
#     def increase(self, value: Union[int, float]) -> None:
#         """Increase node by a value.
#
#         :param value: value to use
#         :type value: Union[int, float]
#         """
#         self.value_ += value
#         self.max_value += value
#         self.min_value += value
#         self.lazy += value
#
#     def unlazy(self) -> None:
#         """Unlazy node."""
#         if self.left is not None:
#             self.left.increase(value=self.lazy)
#         if self.right is not None:
#             self.right.increase(value=self.lazy)
#         self.lazy = 0
#
#     def split_first(self):
#         """Split first element."""
#         self.unlazy()
#
#         if self.left is not None:
#             left, self.left = self.left.split_first()
#             right = self
#         else:
#             right = self.right
#             self.right = None
#             left = self
#
#         if self.left is not None:
#             self.left.update()
#         if self.right is not None:
#             self.right.update()
#         return left, right
#
#     def split_last(self):
#         """Split last element."""
#         self.unlazy()
#
#         if self.right is not None:
#             self.right, right = self.right.split_last()
#             left = self
#         else:
#             left = self.left
#             self.left = None
#             right = self
#
#         if self.left is not None:
#             self.left.update()
#         if self.right is not None:
#             self.right.update()
#         return left, right
#
#     def update(self):
#         """Update node values."""
#         self.unlazy()
#
#         self.size = 1
#         self.height = 0
#         self.max_value = self.value_
#         self.min_value = self.value_
#
#         if self.left is not None:
#             self.size += self.left.size
#             self.height = self.left.height
#             self.max_value = max(self.max_value, self.left.max_value)
#             self.min_value = min(self.min_value, self.left.min_value)
#
#         if self.right is not None:
#             self.size += self.right.size
#             self.height = max(self.height, self.right.height)
#             self.max_value = max(self.max_value, self.right.max_value)
#             self.min_value = min(self.min_value, self.right.min_value)
#
#         self.height += 1
#
#     def split(self, key: Tuple[float, Union[int, float]]):
#         """Split."""
#         self.unlazy()
#
#         if key <= self.key:
#             left, self.left = (
#                 self.left.split(key=key) if self.left is not None else (None, None)
#             )
#             right = self
#         else:
#             self.right, right = (
#                 self.right.split(key=key) if self.right is not None else (None, None)
#             )
#             left = self
#         if left is not None:
#             left.update()
#         if right is not None:
#             right.update()
#         return left, right
#
#
# class Treap:
#     """Class representing a treap."""
#
#     def __init__(self, r: float = 1.0) -> None:
#         """Init method.
#
#         :param r: constant that satisfies |A| = r|B|
#         :type r: float
#         """
#         self.root = None
#         self.r = r
#         self.num_samples = [0, 0]
#
#     @property
#     def root(self) -> Optional[Node]:
#         """Root node property.
#
#         :return: root node
#         :rtype: Optional[Node]
#         """
#         return self._root
#
#     @root.setter
#     def root(self, value: Optional[Node]) -> None:
#         """Root node setter.
#
#         :param value: value to be set
#         :type value: int
#         :raises TypeError: Type error exception
#         """
#         if value is not None and not isinstance(value, Node):
#             raise TypeError("value must be of type Node or None.")
#         self._root = value
#
#     @property
#     def r(self) -> Union[int, float]:
#         """R value property.
#
#         :return: r value
#         :rtype: Union[int, float]
#         """
#         return self._r
#
#     @r.setter
#     def r(self, value: Union[int, float]) -> None:
#         """R value setter.
#
#         :param value: value to be set
#         :type value: Union[int, float]
#         :raises TypeError: Type error exception
#         """
#         if not isinstance(value, (int, float)):
#             raise TypeError("value must be of type int or float.")
#         self._r = value
#
#     @property
#     def num_samples(self) -> List[int]:
#         """Number of samples property.
#
#         :return: number of samples
#         :rtype: List[int]
#         """
#         return self._num_samples
#
#     @num_samples.setter
#     def num_samples(self, value: List[int]) -> None:
#         """Number of samples value setter.
#
#         :param value: value to be set
#         :type value: List[int]
#         :raises ValueError: Value error exception
#         """
#         if value[0] < 0 and value[1] < 0:
#             raise ValueError("value must be a positive number in both elements.")
#         self._num_samples = value
#
#     @property
#     def max(self) -> Union[int, float]:
#         """Maximum value property.
#
#         :return: maximum value
#         :rtype: Union[int, float]
#         """
#         return self.root.max_value  # type: ignore
#
#     @property
#     def min(self) -> Union[int, float]:
#         """Minimum value property.
#
#         :return: minimum value
#         :rtype: Union[int, float]
#         """
#         return self.root.min_value  # type: ignore
#
#     def insert(self, obs: float, group: int) -> None:
#         """Insert an observation by group.
#
#         :param obs: observation to insert
#         :type obs: float
#         :param group: group to which the observations belongs
#         :type group: int
#         """
#         key = (obs, group)
#         r = 1 if group == 0 else -self.r
#
#         self.num_samples[group] += 1
#
#         left, right = self.root.split(key) if self.root is not None else (None, None)
#
#         left, temp = left.split_last() if left is not None else (None, None)
#         initial_value = 0 if temp is None else temp.value_
#
#         left = self.merge(left=left, right=temp)
#
#         right = self.merge(left=Node(key=key, value=initial_value), right=right)
#         right.increase(value=r)
#
#         self.root = self.merge(left=left, right=right)
#
#     def remove(self, obs: float, group: int) -> None:
#         """Remove an observation by group.
#
#         :param obs: observation to remove
#         :type obs: float
#         :param group: group to which the observations belongs
#         :type group: int
#         """
#         key = (obs, group)
#         r = -1 if group == 0 else self.r
#
#         self.num_samples[group] -= 1
#
#         left, right = self.root.split(key) if self.root is not None else (None, None)
#         temp, right = right.split_first() if right is not None else (None, None)
#
#         if right is not None and temp is not None and temp.key == key:
#             right.increase(value=r)
#         else:
#             right = self.merge(left=temp, right=right)
#
#         self.root = self.merge(left=left, right=right)
#
#     def merge(self, left: Optional[Node], right: Optional[Node]) -> Optional[Node]:
#         """Merge two given nodes.
#
#         :param left: left node
#         :type left: Optional[Node]
#         :param right: right node
#         :type right: Optional[Node]
#         :return result of merging the two nodes
#         :rtype Optional[Node]
#         """
#         if left is None or right is None:
#             return left or right
#
#         if left.priority > right.priority:
#             left.unlazy()
#             left.right = self.merge(left=left.right, right=right)
#             node = left
#         else:
#             right.unlazy()
#             right.left = self.merge(left=left, right=right.left)
#             node = right
#         node.update()
#         return node
