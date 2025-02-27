from collections.abc import Callable
from typing import Self, final


class MessageProcessor:

    def __init__(self, simple_func: Callable[[any], None] | None = None):
        self.next_processor: Self | None = None
        self.simple_func = simple_func

    def init(self):
        pass

    @final
    def init_chain(self):
        self.init()
        if self.next_processor is not None:
            self.next_processor.init_chain()

    def process(self, message: any):
        if self.simple_func is not None:
            self.simple_func(message)

    @final
    def next(self, message: any):
        if self.next_processor is not None:
            self.next_processor.process(message)

    @final
    def set_next(self, next_proc: Self):
        self.next_processor = next_proc

    def cleanup(self):
        pass

    @final
    def cleanup_chain(self):
        self.cleanup()
        if self.next_processor is not None:
            self.next_processor.cleanup_chain()


class BatchingProcessor(MessageProcessor):

    def __init__(self, window_size: int, window_step: int = -1):
        super().__init__()
        self.window_size: int = window_size  # in number of messages
        self.window_step: int = window_size if window_step == -1 else window_step
        self.buffer: list[any] = []

    def process(self, message: any):
        self.buffer.append(message)
        if len(self.buffer) == self.window_size:
            self.next(self.buffer)
            self.buffer = self.buffer[self.window_step:]


class FilteringProcessor(MessageProcessor):

    def __init__(self, predicate: Callable[[any], bool]):
        super().__init__()
        self.predicate = predicate

    def process(self, message: any):
        if self.predicate(message):
            self.next(message)
