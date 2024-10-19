from queue import Queue
from threading import Condition


class PeekableQueue(Queue):
    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        self.condition = Condition()

    def peek(self) -> any:
        with self.condition:
            while self.empty():
                self.condition.wait()

            item, read_count, max_reads = self.queue[0]

            read_count += 1

            if read_count >= max_reads:  # this is the last subscriber
                self.get_nowait()  # remove element
                self.condition.notify_all()
            else:
                self.queue[0] = (item, read_count, max_reads)

            return item

    def put_with_max_reads(self, item, max_reads):
        with self.condition:
            while self.full():
                self.condition.wait()

            super().put((item, 0, max_reads))
            self.condition.notify_all()
