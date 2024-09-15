from typing import TypeVar, Generic
from queue import Queue

T = TypeVar('T')


class QueueIterator(Generic[T]):
    def __init__(self, queue: Queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self) -> T:
        data: T = self.queue.get()
        if data is None:
            raise StopIteration from None
        return data
