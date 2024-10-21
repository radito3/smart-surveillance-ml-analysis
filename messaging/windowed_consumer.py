from messaging.broker_interface import Broker
from messaging.consumer import Consumer


class WindowedConsumer(Consumer):

    def __init__(self, broker: Broker, topic: str, window_size: int, window_step: int = -1):
        super().__init__(broker, topic)
        self.window_size: int = window_size  # in number of messages
        self.window_step: int = window_size if window_step == -1 else window_step
        self.buffer: list[any] = []

    def __next__(self) -> any:
        while len(self.buffer) < self.window_size:
            self.buffer.append(super().__next__())
        result = self.buffer.copy()
        self.buffer = self.buffer[self.window_step:]
        return result
