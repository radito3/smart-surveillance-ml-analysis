import logging
import traceback
from threading import Thread, Lock, Event
from itertools import cycle

from .aggregate_consumer import AggregateConsumer
from .topic import Topic
from .broker_interface import Broker
from .consumer import Consumer


class MessageBroker(Broker):

    def __init__(self):
        self.topics: dict[str, Topic] = {}
        self.topics_lock: Lock = Lock()
        self.consumer_threads: dict[str, Thread] = {}
        self.shutdown: Event = Event()

    def read_from(self, topic: str, consumer_name: str) -> any:
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        return self.topics[topic].consume(consumer_name)

    def write_to(self, topic: str, message: any) -> bool:
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        return self.topics[topic].publish(message)

    def create_topic(self, topic: str):
        with self.topics_lock:
            if topic not in self.topics:
                self.topics[topic] = Topic(topic)

    def add_subscriber_for(self, topic: str, consumer: Consumer):
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        with self.topics_lock:
            self.topics[topic].subscribe(consumer.get_name())

        if consumer.get_name() not in self.consumer_threads:
            self.consumer_threads[consumer.get_name()] = Thread(name=consumer.get_name() + '-thread',
                                                                target=self.__consumer_thread_wrapper,
                                                                args=(topic, consumer,))

    def __consumer_thread_wrapper(self, topic: str, consumer: Consumer):
        subscribed_topics = cycle([topic for topic in self.topics.keys()
                                   if consumer.get_name() in self.topics[topic].subscribers])
        exhausted_topic: bool = False
        exhausted_topic_names: set[str] = set()

        try:
            consumer.init()
            while not self.shutdown.is_set():
                for _topic in subscribed_topics:
                    message = self.topics[_topic].consume(consumer.get_name())
                    if message is None:  # read until a tombstone message
                        exhausted_topic = True
                        exhausted_topic_names.add(topic)
                        break
                    consumer.process_message(message if not isinstance(consumer, AggregateConsumer) else (_topic, message))

                if exhausted_topic:
                    exhausted_topic = False
                    subscribed_topics = filter(lambda t: t not in exhausted_topic_names, subscribed_topics)

            consumer.cleanup()
        except Exception as e:
            logging.error(f"Exception occurred in {consumer.get_name()}: {e}")
            traceback.print_exception(e)
            # we will rely on k8s to restart the whole Pod in case of failure
            # this is much simpler than adding Dead-letter Queues and retries for consumers
            self.interrupt()

        with self.topics_lock:
            self.topics[topic].unsubscribe(consumer.get_name())

    # the broker should not be the one facilitating these 2 methods
    def start_streams(self):
        [thread.start() for thread in self.consumer_threads.values()]

    def wait(self):
        [thread.join() for thread in self.consumer_threads.values()]

    def interrupt(self):
        if not self.shutdown.is_set():
            self.shutdown.set()
            for topic in self.topics.values():
                topic.stop_processing_messages()
