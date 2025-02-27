from threading import Lock, Event

from .topic import Topic


class MessageBroker:

    def __init__(self):
        self.topics: dict[str, Topic] = {}
        self.topics_lock: Lock = Lock()
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

    def subscribe_to(self, topic: str, consumer_name: str):
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        with self.topics_lock:
            self.topics[topic].subscribe(consumer_name)

    def unsubscribe_from(self, topic: str, consumer_name: str):
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        with self.topics_lock:
            self.topics[topic].unsubscribe(consumer_name)

    def interrupt(self):
        if not self.shutdown.is_set():
            self.shutdown.set()
            for topic in self.topics.values():
                topic.stop_processing_messages()
