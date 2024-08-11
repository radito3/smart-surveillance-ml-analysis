from typing import TypeVar, Generic

import multiprocessing.connection as cn


T = TypeVar('T')


class ConnIterator(Generic[T]):
    def __init__(self, conn: cn.Connection):
        self.conn = conn

    def __iter__(self):
        return self

    def __next__(self) -> T:
        # should there be a timeout when reading?
        data: T = self.conn.recv()
        if data is None:
            raise StopIteration from None
        return data
