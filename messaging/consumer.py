import logging

from messaging.message_broker import MessageBroker


class Consumer:
    def __init__(self, broker: MessageBroker):
        self.broker = broker

    def run(self, topic: str):
        self.broker.subscribe_to(topic)
        while self.broker.is_running():
            message = self.broker.read_from(topic)
            if message is None:
                break
            self.accept(topic, message)
        self.broker.unsubscribe_from(topic)

    def accept(self, topic: str, message: any):
        logging.warning(f"Discarding message from topic {topic}...")
