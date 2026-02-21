import logging
import traceback

from messaging.message_broker import MessageBroker
from messaging.processor import MessageProcessor


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
        self.broker.subscribe_to(self.source)
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
