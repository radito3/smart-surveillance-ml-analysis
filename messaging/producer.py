from messaging.message_broker import MessageBroker


class Producer:
    def __init__(self, broker: MessageBroker):
        self.broker = broker

    def name(self) -> str:
        return "base-producer"

    def run(self):
        pass
