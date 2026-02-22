from threading import Thread
from itertools import chain

from messaging.consumer import Consumer
from messaging.message_broker import MessageBroker
from messaging.stream import Stream
from messaging.producer import Producer


class Topology:
    def __init__(self, broker: MessageBroker, streams: list[Stream]):
        self.broker = broker
        self.streams: list[Stream] = streams
        self.sources: dict[str, Producer] = {}
        self.sinks: dict[str, Consumer] = {}

    def add_source(self, topic: str, producer: Producer):
        self.sources[topic] = producer

    def add_sink(self, topic: str, consumer: Consumer):
        self.sinks[topic] = consumer


class KafkaStreams:
    def __init__(self, topology: Topology):
        self.topology: Topology = topology
        self.threads: list[Thread] = []

    def start(self):
        topics = chain.from_iterable([
            map(lambda s: s.source, self.topology.streams),
            self.topology.sources.keys(),
            self.topology.sinks.keys()
        ])
        for topic in topics:
            self.topology.broker.create_topic(topic)

        for producer in self.topology.sources.values():
            self.threads.append(Thread(name=producer.name()+"-thread", target=producer.run))

        self.threads.extend([Thread(name=stream.name, target=stream.run) for stream in self.topology.streams])

        for topic, consumer in self.topology.sinks.items():
            self.threads.append(Thread(name=topic+"-consumer-thread", target=self.__consumer_runner, args=(topic, consumer,)))

        for thread in self.threads:
            thread.start()

    def __consumer_runner(self, topic: str, consumer: Consumer):
        self.topology.broker.subscribe_to(topic)
        while self.topology.broker.is_running():
            message = self.topology.broker.read_from(topic)
            if message is None:
                break
            consumer.process(message)
        self.topology.broker.unsubscribe_from(topic)

    def wait(self):
        for thread in self.threads:
            thread.join()
