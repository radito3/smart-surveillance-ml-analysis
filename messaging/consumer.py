from .broker_interface import Broker


class Consumer:

    def __init__(self, broker: Broker, topic: str):
        self.broker: Broker = broker
        self.topic: str = topic

    def get_name(self) -> str:
        return 'base-consumer'

    def init(self) -> bool:
        return True

    def __iter__(self):
        return self

    def __next__(self) -> any:
        message = self.broker.read_from(self.topic)
        if message is None:  # read until a tombstone message
            raise StopIteration
        return message

    def run(self):
        for msg in self:
            self.consume_message(msg)
        self.cleanup()

    def cleanup(self):
        pass

    def consume_message(self, message: any):
        # we don't need an abstract method to allow instantiation of this base class
        pass
