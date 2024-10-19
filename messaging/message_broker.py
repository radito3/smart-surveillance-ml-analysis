import logging
import traceback
from threading import Thread, Lock

from util.peekable_queue import PeekableQueue
from .broker_interface import Broker
from .consumer import Consumer


class MessageBroker(Broker):

    def __init__(self):
        self.topics_mappings: dict[str, tuple[PeekableQueue[any], int]] = {}
        self.topics_lock: Lock = Lock()
        self.consumer_threads: dict[str, Thread] = {}

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

    def add_subscriber_for(self, topic: str, consumer: Consumer):
        if topic not in self.topics_mappings:
            raise ValueError(f"Topic {topic} does not exist")

        with self.topics_lock:
            tmp = self.topics_mappings[topic]
            self.topics_mappings[topic] = (tmp[0], tmp[1] + 1)
        if consumer.get_name() not in self.consumer_threads:
            self.consumer_threads[consumer.get_name()] = Thread(name=consumer.get_name() + '-thread',
                                                                target=self.__consumer_thread_wrapper,
                                                                args=(topic, consumer,))

    def __consumer_thread_wrapper(self, topic: str, consumer: Consumer):
        try:
            consumer.init()
            consumer.run()
        except Exception as e:
            logging.error(f"Exception occurred in {consumer.get_name()}: {e}")
            traceback.print_exception(e)

        with self.topics_lock:
            if self.topics_mappings[topic][1] > 0:
                tmp = self.topics_mappings[topic]
                self.topics_mappings[topic] = (tmp[0], tmp[1] - 1)

    def start_streams(self):
        [thread.start() for thread in self.consumer_threads.values()]

    def wait(self):
        [thread.join() for thread in self.consumer_threads.values()]
        # delete topics for cleanup?

    def interrupt(self):
        for topic in self.topics_mappings.values():
            topic[0].put_with_max_reads(None, topic[1])
