from threading import Event

from .topic import Topic


class MessageBroker:

    def __init__(self):
        self.topics: dict[str, Topic] = {}
        self.shutdown: Event = Event()

    def read_from(self, topic: str) -> any:
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        return self.topics[topic].consume()

    def write_to(self, topic: str, message: any):
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        self.topics[topic].publish(message)

    def create_topic(self, topic: str):
        if topic not in self.topics:
            self.topics[topic] = Topic(topic)

    def subscribe_to(self, topic: str):
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        self.topics[topic].subscribe()

    def unsubscribe_from(self, topic: str):
        if topic not in self.topics:
            raise ValueError(f"Topic {topic} does not exist")

        self.topics[topic].unsubscribe()

    def is_running(self) -> bool:
        return not self.shutdown.is_set()

    def interrupt(self):
        if not self.shutdown.is_set():
            for topic in self.topics.values():
                topic.stop_processing_messages()
            self.shutdown.set()
