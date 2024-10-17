from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.windowed_consumer import WindowedConsumer


class AggregateConsumer(Consumer):

    def __init__(self, broker: Broker, topics: list[str], **kwargs):
        super().__init__(broker, '-')
        self.delegates = [self.__create_delegate(broker, topic, **kwargs) for topic in topics]

    @staticmethod
    def __create_delegate(broker: Broker, topic: str, **kwargs) -> Consumer:
        window_size = int(kwargs[topic]) if topic in kwargs else 1
        if window_size == 1:
            return Consumer(broker, topic)
        if 'step' in kwargs:
            return WindowedConsumer(broker, topic, window_size, int(kwargs['step']))
        return WindowedConsumer(broker, topic, window_size)

    def __next__(self) -> any:
        aggregate_msg: dict[str, any] = {}
        for delegate in self.delegates:
            message = delegate.__next__()
            aggregate_msg[delegate.topic] = message
        return aggregate_msg
