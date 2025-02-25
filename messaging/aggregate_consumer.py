from collections import defaultdict, namedtuple

from messaging.consumer import Consumer

WindowConfig = namedtuple('WindowConfig', ['window_size', 'window_step'])

class AggregateConsumer(Consumer):

    def __init__(self, topics: list[str], **kwargs):
        self.num_topics = len(topics)
        self.buffer_configs = {topic: WindowConfig(int(kwargs[topic]), int(kwargs['step']) if 'step' in kwargs else -1)
                               for topic in topics if topic in kwargs}
        self.buffers: dict[str, list[any]] = defaultdict(list)
        self.aggregate_msg: dict[str, any] = {}

    def process_message(self, message: any):
        topic, payload = message
        if topic in self.buffer_configs:
            self.buffers[topic].append(payload)
            # what if the messages from the buffered topic overflow and overwrite the previous buffer value?
            # one solution is to have a ring buffer (collections.deque) that drops the oldest messages
            # ideally we don't want to drop messages...
            if len(self.buffers[topic]) == self.buffer_configs[topic].window_size:
                buffer_copy = self.buffers[topic].copy()  # we want a snapshot of the list
                self.aggregate_msg[topic] = buffer_copy
                self.buffers[topic] = self.buffers[topic][self.buffer_configs[topic].window_step:]
        else:
            self.aggregate_msg[topic] = payload

        if len(self.aggregate_msg) == self.num_topics:
            self.process_aggregated_message(self.aggregate_msg)
            self.aggregate_msg.clear()

    def process_aggregated_message(self, message: dict[str, any]):
        pass
