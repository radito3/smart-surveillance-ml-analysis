import logging
import traceback
from threading import Event
from typing import Self, Callable

from messaging.message_broker import MessageBroker
from messaging.processor import MessageProcessor, BatchingProcessor, FilteringProcessor


class Stream:

    def __init__(self, broker: MessageBroker, name: str, sources: list[str], pipeline_head: MessageProcessor,
                 output_topic: str | None):
        self.broker = broker
        self.name = name
        self.sources = sources
        self.pipeline_head = pipeline_head
        self.output_topic = output_topic
        self.shutdown: Event = Event()

    def run(self):
        try:
            self.pipeline_head.init_chain()
            should_stop: bool = False
            while not self.shutdown.is_set():
                message: any | dict[str, any] = {}
                for source_topic in self.sources:
                    inner_message = self.broker.read_from(source_topic, self.name)
                    if inner_message is None:  # read until a tombstone message
                        should_stop = True
                        break
                    if len(self.sources) > 1:
                        message[source_topic] = inner_message
                    else:
                        message = inner_message

                if should_stop:
                    break
                self.pipeline_head.process(message)

            if self.output_topic is not None:
                self.broker.write_to(self.output_topic, None)  # notify downstream consumers to gracefully stop
            self.pipeline_head.cleanup_chain()
        except Exception as e:
            logging.error(f"Exception occurred in {self.name}: {e}")
            traceback.print_exception(e)
            self.shutdown.set()

        for topic in self.sources:
            self.broker.unsubscribe_from(topic, self.name)


class StreamConfig:
    def __init__(self, name: str | None):
        self.name: str | None = name
        self.source_topics: list[str] = []
        self.pipeline: list[MessageProcessor] = []
        self.output_topic: str | None = None
        self.sink = None


class StreamsBuilder:

    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.configs: list[StreamConfig] = []
        self.current_config: StreamConfig = StreamConfig(None)

    def stream(self, *topics: str) -> Self:
        self.current_config.source_topics = [*topics]
        return self

    def named(self, name: str) -> Self:
        self.current_config.name = name
        # needs to be early because the producer might have published records by the time the streams are run
        # and in that case, the messages would be lost
        [self.broker.subscribe_to(topic, name) for topic in self.current_config.source_topics]
        return self

    def window(self, size, step=-1) -> Self:
        self.current_config.pipeline.append(BatchingProcessor(size, step))
        return self

    def filter(self, predicate: Callable[[any], bool]) -> Self:
        self.current_config.pipeline.append(FilteringProcessor(predicate))
        return self

    def process(self, transform_func: MessageProcessor) -> Self:
        self.current_config.pipeline.append(transform_func)
        return self

    # consider method through(topic) which will combine to(topic) and stream(topic) while keeping the stream name

    def to(self, topic: str):
        self.current_config.output_topic = topic
        self.current_config.sink = lambda msg: self.broker.write_to(topic, msg)
        self.configs.append(self.current_config)
        self.current_config = StreamConfig(None)

    def for_each(self, action: Callable[[any], None]):
        self.current_config.sink = action
        self.configs.append(self.current_config)
        self.current_config = StreamConfig(None)

    def build(self) -> list[Stream]:
        return [self.__build_single(config) for config in self.configs]

    def __build_single(self, config: StreamConfig) -> Stream:
        num_stages = len(config.pipeline)

        if num_stages > 1:
            for i in range(num_stages - 1):
                config.pipeline[i].set_next(config.pipeline[i + 1])

        if num_stages == 0:
            config.pipeline.append(MessageProcessor(lambda msg: config.sink(msg)))
        else:
            config.pipeline[-1].set_next(MessageProcessor(lambda msg: config.sink(msg)))

        return Stream(self.broker, config.name, config.source_topics, config.pipeline[0], config.output_topic)
