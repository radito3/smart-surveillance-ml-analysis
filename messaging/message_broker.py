from collections.abc import Callable
from queue import Queue
from threading import Lock, Thread, Condition

from .broker_interface import Broker
from .consumer import Consumer


class PeekableQueue(Queue):
    def __init__(self, maxsize=0):
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
                self.task_done()
            else:
                self.queue[0] = (item, read_count, max_reads)

            return item

    def put_with_max_reads(self, item, max_reads):
        with self.condition:
            super().put((item, 0, max_reads))
            self.condition.notify_all()


class MessageBroker(Broker):

    def __init__(self):
        self.topics_mappings: dict[str, tuple[PeekableQueue[any], int]] = {}
        self.topics_lock: Lock = Lock()
        self.consumer_threads: list[Thread] = []

    def read_from(self, topic: str) -> any:
        if topic not in self.topics_mappings:
            raise ValueError(f"Topic {topic} does not exist")

        return self.topics_mappings[topic][0].peek()

    def write_to(self, topic: str, message: any):
        if topic not in self.topics_mappings:
            raise ValueError(f"Topic {topic} does not exist")

        mapping = self.topics_mappings[topic]
        mapping[0].put_with_max_reads(message, mapping[1])

    def create_topic(self, topic: str):
        with self.topics_lock:
            if topic not in self.topics_mappings:
                self.topics_mappings[topic] = (PeekableQueue(10), 0)

    def subscribe_to(self, topic: str, consumer: Consumer):
        if topic not in self.topics_mappings:
            raise ValueError(f"Topic {topic} does not exist")

        self.topics_mappings[topic][1] += 1
        self.consumer_threads.append(Thread(target=self.__consumer_thread_wrapper, args=(topic, consumer)))

    def __consumer_thread_wrapper(self, topic: str, consumer: Consumer):
        if not consumer.init():
            self.write_to(topic, None)
            return
        consumer.run()

    def delete_topic(self, topic: str):
        with self.topics_lock:
            if topic in self.topics_mappings:
                self.topics_mappings[topic][0].join()  # wait for all messages to be consumed, including the tombstone
                del self.topics_mappings[topic]

    def start(self):
        # try init the producers
        # if failure, send a tombstone message and exit; the consumer threads will exit soon after
        [thread.start() for thread in self.consumer_threads]

    def wait(self):
        [thread.join() for thread in self.consumer_threads]
        # delete topics for cleanup?
