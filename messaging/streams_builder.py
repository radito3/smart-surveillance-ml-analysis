from typing import Self, Callable

from messaging.message_broker import MessageBroker
from messaging.stream import Stream
from messaging.processor import (
    MessageProcessor,
    BatchingProcessor,
    FilteringProcessor,
    CyclicBarrierFilter,
    StreamJoiner,
)


class StreamConfig:
    def __init__(self):
        self.name: str = "-"
        self.source_topic: str = ""
        self.pipeline: list[MessageProcessor] = []
        self.output_topic: str | None = None
        self.sink = None


class StreamsBuilder:
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.configs: list[StreamConfig] = []
        self.current_config: StreamConfig = StreamConfig()

    def stream(self, topic: str) -> Self:
        self.current_config.source_topic = topic
        return self

    def join(self, other_topic: str, joiner: Callable[[any, any], any]) -> Self:
        self.current_config.pipeline.append(StreamJoiner(self.broker, other_topic, joiner))
        return self

    def named(self, name: str) -> Self:
        self.current_config.name = name
        return self

    def window(self, size, step=-1) -> Self:
        self.current_config.pipeline.append(BatchingProcessor(size, step))
        return self

    def barrier(self, threshold: int, predicate: Callable[[any], bool]) -> Self:
        self.current_config.pipeline.append(CyclicBarrierFilter(threshold, predicate))
        return self

    def filter(self, predicate: Callable[[any], bool]) -> Self:
        self.current_config.pipeline.append(FilteringProcessor(predicate))
        return self

    def process(self, transform_func: MessageProcessor) -> Self:
        self.current_config.pipeline.append(transform_func)
        return self

    def through(self, topic: str) -> Self:
        name = self.current_config.name
        self.to(topic)
        return self.stream(topic).named(name+"-passthrough")

    def to(self, topic: str):
        self.current_config.output_topic = topic
        self.current_config.sink = lambda msg: self.broker.write_to(topic, msg)
        self.configs.append(self.current_config)
        self.current_config = StreamConfig()

    def for_each(self, action: Callable[[any], None]):
        self.current_config.sink = action
        self.configs.append(self.current_config)
        self.current_config = StreamConfig()

    def build(self) -> list[Stream]:
        return [self.__build_single(config, self.broker) for config in self.configs]

    @staticmethod
    def __build_single(config: StreamConfig, broker: MessageBroker) -> Stream:
        num_stages = len(config.pipeline)

        if num_stages > 1:
            for i in range(num_stages - 1):
                config.pipeline[i].set_next(config.pipeline[i + 1])

        if num_stages == 0:
            config.pipeline.append(MessageProcessor(lambda msg: config.sink(msg)))
        else:
            config.pipeline[-1].set_next(MessageProcessor(lambda msg: config.sink(msg)))

        return Stream(
            broker,
            config.name,
            config.source_topic,
            config.pipeline[0],
            config.output_topic,
        )
