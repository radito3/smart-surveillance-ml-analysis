from queue import Queue
from threading import Condition


class TopicQueue(Queue):
    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        self.condition: Condition = Condition()
        self.interrupted: bool = False

    def peek(self) -> any:
        with self.condition:
            while self.qsize() == 0:
                if self.interrupted:
                    return None
                self.condition.wait()

            if self.interrupted:
                return None

            item, read_count, max_reads = self.queue[0]

            read_count += 1

            if read_count >= max_reads:  # this is the last subscriber
                self.get_nowait()  # remove element
                self.condition.notify_all()
            else:
                self.queue[0] = (item, read_count, max_reads)

            return item

    def interrupt(self):
        self.interrupted = True
        with self.condition:
            self.condition.notify_all()

    def put_with_max_reads(self, item, max_reads) -> bool:
        with self.condition:
            while self.qsize() >= self.maxsize:
                if self.interrupted:
                    return False
                self.condition.wait()

            if self.interrupted:
                return False

            super().put((item, 0, max_reads))
            self.condition.notify_all()
        return True
