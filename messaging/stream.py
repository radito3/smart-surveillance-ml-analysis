import logging
import traceback
from typing import Self, Callable

from messaging.message_broker import MessageBroker
from messaging.processor import (
    MessageProcessor,
    BatchingProcessor,
    FilteringProcessor,
    CyclicBarrierFilter,
    StreamJoiner,
)


class Stream:
    def __init__(
        self,
        broker: MessageBroker,
        name: str,
        source: str,
        pipeline_head: MessageProcessor,
        output_topic: str | None,
    ):
        self.broker = broker
        self.name = name
        self.source = source
        self.pipeline_head = pipeline_head
        self.output_topic = output_topic

    def run(self):
        try:
            self.pipeline_head.init_chain()

            while self.broker.is_running():
                message = self.broker.read_from(self.source)
                if message is None:  # read until a tombstone message
                    break
                self.pipeline_head.process(message)

            if self.output_topic is not None:
                # notify downstream consumers to gracefully stop
                self.broker.write_to(self.output_topic, None)

            self.pipeline_head.cleanup_chain()
        except Exception as e:
            logging.error(f"Exception occurred in {self.name}: {e}")
            traceback.print_exception(e)

        self.broker.unsubscribe_from(self.source)


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
        self.broker.subscribe_to(topic)
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
        return self.stream(topic).named(name)

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
