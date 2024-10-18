from .broker_interface import Broker


class Producer:

    def __init__(self, broker: Broker):
        self.broker: Broker = broker

    def init(self):
        pass

    def get_name(self) -> str:
        return 'base-producer'

    def produce_value(self, topic: str, value: any):
        self.broker.write_to(topic, value)
