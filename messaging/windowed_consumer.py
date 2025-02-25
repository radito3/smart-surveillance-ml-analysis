from messaging.broker_interface import Broker
from messaging.consumer import Consumer
from messaging.producer import Producer


class BatchingProcessor(Producer, Consumer):

    def __init__(self, broker: Broker, window_size: int, window_step: int = -1):
        Producer.__init__(self, broker)
        self.window_size: int = window_size  # in number of messages
        self.window_step: int = window_size if window_step == -1 else window_step
        self.buffer: list[any] = []

    def process_message(self, message: any):
        self.buffer.append(message)
        if len(self.buffer) == self.window_size:
            self.process_windowed_message(self.buffer)
            self.buffer = self.buffer[self.window_step:]

    # def buffer_add_hook(self, message: any) -> any:
    #     return message

    def process_windowed_message(self, message: list[any]):
        pass
